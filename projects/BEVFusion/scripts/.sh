#!/bin/bash

GPUS=1

cd /workspace

bash tools/dist_train.sh \
    projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py\
    $GPUS

# LIDAR_PRETRAINED_CHECKPOINT=/workspace/projects/BEVFusion/checkpoints/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth
# IMAGE_PRETRAINED_BACKBONE=/workspace/projects/BEVFusion/checkpoints/swint-nuimages-pretrained.pth
# bash tools/dist_train.sh \
#     projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
#     $GPUS \
#     --cfg-options \
#     load_from=${LIDAR_PRETRAINED_CHECKPOINT} \
#     model.img_backbone.init_cfg.checkpoint=${IMAGE_PRETRAINED_BACKBONE}