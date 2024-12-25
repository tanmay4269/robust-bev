from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from mmengine.utils import is_list_of

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from .ops import Voxelization

from mmdet.models import DetrTransformerDecoder

@MODELS.register_module()
class BEVFusion(Base3DDetector):

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        view_recon_cfg: Optional[dict] = None,
        tmp_cfg: Optional[dict] = None,
        **kwargs,
    ) -> None:
        voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # TODO: Make this another module
        if view_recon_cfg is None:
            view_recon_cfg = dict(
                enabled=False, 
                pretraining=False, 
                use_reconstructed_view=False)
        self.view_recon_enabled = view_recon_cfg.pop('enabled')
        self.view_recon_pretraining = view_recon_cfg.pop('pretraining')
        self.use_reconstructed_view = view_recon_cfg.pop('use_reconstructed_view')
        self.num_views_drop = view_recon_cfg.pop('num_views_drop')

        # Temporary config for analysis
        self.drop_pts_feature = tmp_cfg.pop('drop_pts_feature')

        assert self.view_recon_enabled or not self.view_recon_pretraining, \
            "Error: view_recon_enabled is False while view_recon_pretraining is True"

        if not self.view_recon_pretraining:
            self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
            self.pts_voxel_layer = Voxelization(**voxelize_cfg)

            self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)

        self.img_backbone = MODELS.build(
            img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(
            img_neck) if img_neck is not None else None
        
        if self.view_recon_enabled:
            self.num_views = 6
            embed_dims = view_recon_cfg.pop('embed_dims')
            self.view_embed = nn.Embedding(
                self.num_views, embed_dims)
            self.learned_query = nn.Parameter(
                torch.randn(1, embed_dims))

            C, H, W = 256, 32, 88  # TODO: Find a better way of getting these values
            self.dropped_view_detector = nn.Sequential(
                nn.Conv2d(in_channels= C, out_channels= 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(start_dim=1),
                nn.Linear(16 * H * W, 1),
                nn.Sigmoid()
            )
         
            # TODO: Add this to config
            self.view_recon_net = DetrTransformerDecoder(
                num_layers=view_recon_cfg.pop('num_layers'),
                layer_cfg=view_recon_cfg.pop('layer_cfg'),
            )

            checkpoint = view_recon_cfg.pop('checkpoint')
            # TODO: Figure out a better way of doing this
            if checkpoint is not None:
                state_dict = torch.load(checkpoint)['state_dict']
                def _helper(module_name):
                    ret = dict()
                    for k, v in state_dict.items():
                        if k.startswith(module_name):
                            ret[k[len(module_name) + 1:]] = v
                    return ret
                self.view_embed.load_state_dict(_helper('view_embed'))
                self.learned_query.data = state_dict['learned_query']
                self.view_recon_net.load_state_dict(_helper('view_recon_net'))

        if not self.view_recon_pretraining:
            self.view_transform = MODELS.build(
                view_transform) if view_transform is not None else None
            self.pts_middle_encoder = MODELS.build(pts_middle_encoder)

            self.fusion_layer = MODELS.build(
                fusion_layer) if fusion_layer is not None else None

            self.pts_backbone = MODELS.build(pts_backbone)
            self.pts_neck = MODELS.build(pts_neck)

            self.bbox_head = MODELS.build(bbox_head)

        self.init_weights()

        if self.view_recon_pretraining:
            # Freeze all parameters except the view reconstruction network
            for param in self.parameters():
                param.requires_grad = False
            for param in self.view_recon_net.parameters():
                param.requires_grad = True
            self.learned_query.requires_grad = True

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a segmentation head.
        """
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        compute_loss=False
    ) -> torch.Tensor:
        losses = dict()

        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        if self.num_views_drop > 0:
            # ! Check if this works
            dropped_idxs = torch.randint(0, x.shape[1], (B, self.num_views_drop), device=x.device)
            mask = torch.ones_like(x)
            for i in range(B):
                mask[i, dropped_idxs[i], ...] = 0
            x = x * mask

        if self.dropped_view_detector is not None:
            gt_dropped_idxs = dropped_idxs.clone()  # (B, self.num_views_drop)
            views = x.clone().view(-1, C, H, W)

            dropped_probs = self.dropped_view_detector(views)  # (B * self.num_views, )
            bce_loss = F.binary_cross_entropy(dropped_probs, gt_dropped_idxs.view(-1))
            losses['loss/dropped_view_detector'] = bce_loss

        if self.view_recon_enabled:
            x, recon_loss = self.view_reconstruction(
                x, compute_loss, dropped_idxs)
            losses['loss/view_recon'] = recon_loss
            if self.view_recon_pretraining:
                return None, losses

        # with torch.autocast(device_type='cuda', dtype=torch.float32):
        with torch.cuda.amp.autocast(enabled=False):
            x = self.view_transform(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            )
        return (x, losses) if self.view_recon_enabled else (x, None)
    
    # TODO: Add this to utils.py
    def _build_2d_sin_emb(self, H, W, C, device):
        """A simple 2D sinusoidal encoding for pixel coords (u/H, v/W)."""
        u = torch.arange(H, device=device, dtype=torch.float).unsqueeze(1).repeat(1, W)
        v = torch.arange(W, device=device, dtype=torch.float).unsqueeze(0).repeat(H, 1)
        
        # Simple sine/cosine
        pos_u = torch.stack([torch.sin(u / H * np.pi), torch.cos(u / H * np.pi)], dim=-1)
        pos_v = torch.stack([torch.sin(v / W * np.pi), torch.cos(v / W * np.pi)], dim=-1)
        pos = torch.cat([pos_u, pos_v], dim=-1)  # shape (H, W, 4)
        
        # Project up to C if needed (simple linear for brevity)
        linear = nn.Linear(4, C, bias=False).to(device)
        return linear(pos.view(-1, 4)).view(H, W, C)

    def view_reconstruction(self, x, compute_loss=False, dropped_idxs=None):
        """
        Args:
            x (torch.Tensor): Multi-view image features right out 
                of the backbone. Shape (B, N, C, H, W)
            compute_loss (bool): Whether to compute the reconstruction loss.
        Returns:
            torch.Tensor: Reconstructed multi-view image features.
                Shape (B, N, C, H, W)
            torch.Tensor: Reconstruction loss if computed.
        TODOs:
            - [ ] generalize to multi view reconstruction
            - [ ] identify missing view automatically
            - [ ] make this another module
        """
        B, N, C, H, W = x.shape
        # dropped_idx = torch.randint(0, N, (1,), device=x.device).item()
        assert dropped_idxs.shape[0] == 1, \
            "Does not support multiple views drop"
        dropped_idx = dropped_idxs
        dropped_view = x[:, dropped_idx, ...].clone()
        remaining_views = torch.cat([x[:, :dropped_idx, ...], x[:, dropped_idx+1:, ...]], dim=1)  # shape (B, N-1, C, H, W)

        query = self.learned_query.expand(H * W, -1).unsqueeze(0).expand(B, -1, -1)
        key = remaining_views.reshape(B, (N -1) * H * W, C)
        value = key.clone()

        # Position embeddings
        sin_emb = self._build_2d_sin_emb(H, W, C, x.device).view(H * W, C)

        # Dropped view
        dropped_v_emb = self.view_embed(torch.tensor(dropped_idx, device=x.device))
        query_pos_enc = sin_emb + dropped_v_emb
        query_pos_enc = query_pos_enc.unsqueeze(0).expand(B, -1, -1)

        # Remaining views
        view_ids = torch.arange(N - 1, device=x.device).unsqueeze(-1).expand(-1, H * W)
        key_pos_enc = []
        for i in range(N - 1):
            v_emb = self.view_embed(view_ids[i])  # shape (H*W, C)
            key_pos_enc.append(sin_emb + v_emb)
        key_pos_enc = torch.stack(key_pos_enc, dim=0).view((N - 1) * H * W, C)
        key_pos_enc = key_pos_enc.unsqueeze(0).expand(B, -1, -1)  # shape (B, (N-1)*H*W, C)

        reconstructed = self.view_recon_net(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos_enc,
            key_pos=key_pos_enc,
            key_padding_mask=None
        )
        reconstructed_view = reconstructed[-1].reshape(B, 1, C, H, W)

        if self.use_reconstructed_view:
            mask = torch.ones_like(x)
            mask[:, dropped_idx:dropped_idx+1, ...] = 0
            x = x * mask + reconstructed_view * (1 - mask)

        # ? Can this be done based on model's train or eval mode?
        recon_loss = F.mse_loss(reconstructed_view, dropped_view.unsqueeze(1)) if compute_loss else None

        return x, recon_loss

    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        # with torch.autocast('cuda', enabled=False):
        with torch.cuda.amp.autocast(enabled=False):
            points = [point.float() for point in points]
            feats, coords, sizes = self.voxelize(points)
            batch_size = coords[-1, 0] + 1
        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        if self.view_recon_pretraining:
            return []   # Jugad that doesn't work, but forces the 
                        # model to train just for one epoch

        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats, _ = self.extract_feat(batch_inputs_dict, batch_input_metas)

        if self.with_bbox_head:
            outputs = self.bbox_head.predict(feats, batch_input_metas)

        res = self.add_pred_to_datasample(batch_data_samples, outputs)

        return res

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        losses = dict()
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            img_feature, img_losses = self.extract_img_feat(
                imgs,
                deepcopy(points),
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                batch_input_metas,
                compute_loss=True
            )
            features.append(img_feature)
            losses.update(img_losses)
        
        if self.view_recon_pretraining:
            return None, losses
        
        pts_feature = self.extract_pts_feat(batch_inputs_dict)
        if self.drop_pts_feature:
            mask = torch.zeros_like(pts_feature)
            pts_feature = pts_feature * mask
        features.append(pts_feature)

        if self.fusion_layer is not None:
            x = self.fusion_layer(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.pts_backbone(x)
        x = self.pts_neck(x)

        return (x, losses)

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats, losses = self.extract_feat(batch_inputs_dict, batch_input_metas)

        if not self.view_recon_pretraining and self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(feats, batch_data_samples)
            losses.update(bbox_loss)

        return losses
