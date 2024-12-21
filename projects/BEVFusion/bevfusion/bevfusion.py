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
        **kwargs,
    ) -> None:
        voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
        self.pts_voxel_layer = Voxelization(**voxelize_cfg)

        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)

        self.img_backbone = MODELS.build(
            img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(
            img_neck) if img_neck is not None else None
        
        # TODO: Add this to config
        self.view_recon_net = DetrTransformerDecoder(
            num_layers=3,
            layer_cfg=dict(),  # use default config
        )
        # Learned query for reconstruction
        self.learned_query = nn.Parameter(torch.randn(1, 256))

        self.num_views = 6
        self.view_embed_dim = 256
        self.view_embed = nn.Embedding(
            self.num_views, self.view_embed_dim)

        self.view_transform = MODELS.build(
            view_transform) if view_transform is not None else None
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)

        self.fusion_layer = MODELS.build(
            fusion_layer) if fusion_layer is not None else None

        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)

        self.bbox_head = MODELS.build(bbox_head)

        self.init_weights()

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze view reconstruction network
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
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x, recon_loss = self.view_reconstruction(x, compute_loss)

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
        return x, recon_loss
    
    def _build_2d_sin_emb(self, H, W, C, device):
        """Create a simple 2D sinusoidal encoding for pixel coords (u/H, v/W)."""
        u = torch.arange(H, device=device, dtype=torch.float).unsqueeze(1).repeat(1, W)
        v = torch.arange(W, device=device, dtype=torch.float).unsqueeze(0).repeat(H, 1)
        
        # Simple sine/cosine
        pos_u = torch.stack([torch.sin(u / H * np.pi), torch.cos(u / H * np.pi)], dim=-1)
        pos_v = torch.stack([torch.sin(v / W * np.pi), torch.cos(v / W * np.pi)], dim=-1)
        pos = torch.cat([pos_u, pos_v], dim=-1)  # shape (H, W, 4)
        
        # Project up to C if needed (simple linear for brevity)
        linear = nn.Linear(4, C, bias=False).to(device)
        return linear(pos.view(-1, 4)).view(H, W, C)

    def view_reconstruction(self, x, compute_loss=False):
        """
        Args:
            x (torch.Tensor): Multi-view image features right out 
                of the backbone. Shape (B, N, C, H, W)
            compute_loss (bool): Whether to compute the reconstruction loss.
        Returns:
            torch.Tensor: Reconstructed multi-view image features.
                Shape (B, N, C, H, W)
            torch.Tensor: Reconstruction loss if computed.
        Notes:
            Only reconstructs one view right now. Will support multiple views soon.
        """
        B, N, C, H, W = x.shape
        dropped_idx = torch.randint(0, N, (1,), device=x.device).item()
        dropped_view = x[:, dropped_idx, ...].clone()

        # Exclude the dropped view when building key and value
        remain_views = torch.cat([x[:, :dropped_idx, ...], x[:, dropped_idx+1:, ...]], dim=1)  # shape (B, N-1, C, H, W)

        key = remain_views.reshape(B, (N -1) * H * W, C)
        value = key.clone()

        query = self.learned_query.expand(H * W, -1).unsqueeze(0).expand(B, -1, -1)

        # Build 2D sinusoidal embedding for pixels
        sin_emb = self._build_2d_sin_emb(H, W, C, x.device).view(H * W, C)

        # Build view embedding for each remaining view
        view_ids = torch.arange(N - 1, device=x.device).unsqueeze(-1).expand(-1, H * W)
        key_pos_enc = []
        for i in range(N - 1):
            v_emb = self.view_embed(view_ids[i])  # shape (H*W, C)
            key_pos_enc.append(sin_emb + v_emb)
        key_pos_enc = torch.stack(key_pos_enc, dim=0).view((N - 1) * H * W, C)
        key_pos_enc = key_pos_enc.unsqueeze(0).expand(B, -1, -1)  # shape (B, (N-1)*H*W, C)

        # Build for dropped view
        dropped_v_emb = self.view_embed(torch.tensor(dropped_idx, device=x.device))
        query_pos_enc = sin_emb + dropped_v_emb
        query_pos_enc = query_pos_enc.unsqueeze(0).expand(B, -1, -1)

        reconstructed = self.view_recon_net(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos_enc,
            key_pos=key_pos_enc,
            key_padding_mask=None
        )
        reconstructed_view = reconstructed[-1].reshape(B, 1, C, H, W)
        x[:, dropped_idx : dropped_idx+1, ...] = reconstructed_view

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
            img_feature, recon_loss = self.extract_img_feat(
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
        pts_feature = self.extract_pts_feat(batch_inputs_dict)
        features.append(pts_feature)

        if self.fusion_layer is not None:
            x = self.fusion_layer(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.pts_backbone(x)
        x = self.pts_neck(x)

        return (x, recon_loss)

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats, recon_loss = self.extract_feat(batch_inputs_dict, batch_input_metas)

        losses = dict()
        if recon_loss is not None:
            losses['loss/view_recon'] = recon_loss
        if self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(feats, batch_data_samples)

        losses.update(bbox_loss)

        return losses
