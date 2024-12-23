#!/bin/bash

WORKSPACE_PATH=/workspace
CONFIG_PATH=projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
CKPT_PATH=/workspace/projects/BEVFusion/checkpoints/bevfusion-trained.pth

BATCH_SIZE=3

cd $WORKSPACE_PATH
export PYTHONPATH=$PYTHONPATH:$WORKSPACE_PATH

# echo "1. No view drop testing"
# python tools/test.py \
#     $CONFIG_PATH \
#     $CKPT_PATH \
#     --cfg-options \
#         test_dataloader.batch_size=$BATCH_SIZE \
#         model.view_recon_cfg.enabled=False
# echo ===============================================

echo "2. View drop testing - without view reconstruction"
python tools/test.py \
    $CONFIG_PATH \
    $CKPT_PATH \
    --cfg-options \
        test_dataloader.batch_size=$BATCH_SIZE \
        model.view_recon_cfg.tmp_drop_view_only=True \
        model.view_recon_cfg.enabled=False
echo ===============================================

# echo "3. View drop testing - with view reconstruction"
# python tools/test.py \
#     $CONFIG_PATH \
#     $CKPT_PATH \
#     --cfg-options \
#         test_dataloader.batch_size=$BATCH_SIZE
# echo ===============================================
