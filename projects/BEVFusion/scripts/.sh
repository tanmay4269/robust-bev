#!/bin/bash

GPUS=1

cd /tmp/mmdetection_dev/mmdetection3d

# --- Lidar-only training ---
# bash tools/dist_train.sh \
#     projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py\
#     $GPUS

# --- Full training ---
CKPT1=/tmp/mmdetection_dev/mmdetection3d/projects/BEVFusion/checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth
CKPT2=/workspace/projects/BEVFusion/checkpoints/bevfusion_converted.pth
CKPT3=/workspace/projects/BEVFusion/checkpoints/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth
CKPT4=/tmp/mmdetection_dev/mmdetection3d/projects/BEVFusion/checkpoints/bevfusion-trained.pth

# LIDAR_PRETRAINED_CHECKPOINT=/workspace/projects/BEVFusion/checkpoints/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth
# IMAGE_PRETRAINED_BACKBONE=/workspace/projects/BEVFusion/checkpoints/swint-nuimages-pretrained.pth
# bash tools/dist_train.sh \
#     projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
#     $GPUS \
#     --cfg-options \
#     load_from=${LIDAR_PRETRAINED_CHECKPOINT} \
#     model.img_backbone.init_cfg.checkpoint=${IMAGE_PRETRAINED_BACKBONE}

PYTHONPATH="tools/..":$PYTHONPATH \
python tools/train.py \
    projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
    --cfg-options load_from=$CKPT4

# NOTES: 
# - CKPT1 has size mismatch for pts_middle_encoder.* => unusable
# - CKPT2 needs manual change of vtransform to view_transform in the state_dict