#!/bin/bash

WORKSPACE_PATH=/workspace

LIDAR_CAM_CONFIG_PATH=projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
VIEW_RECON_CONFIG_PATH=projects/BEVFusion/configs/view_recon.py

CKPT_0_PATH=projects/BEVFusion/checkpoints/bevfusion-trained.pth
CKPT_1_PATH=projects/BEVFusion/checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth
CKPT_2_PATH=projects/BEVFusion/checkpoints/bevfusion-2.pth

BATCH_SIZE=14

cd $WORKSPACE_PATH
export PYTHONPATH=$PYTHONPATH:$WORKSPACE_PATH

# echo "1. No view drop testing"
# python tools/test.py \
#     $VIEW_RECON_CONFIG_PATH \
#     $CKPT_2_PATH \
#     --cfg-options \
#         test_dataloader.batch_size=$BATCH_SIZE \
#         model.view_recon_cfg.enabled=False
# echo ===============================================

# echo "2. View drop testing - without view reconstruction"
# python tools/test.py \
#     $VIEW_RECON_CONFIG_PATH \
#     $CKPT_2_PATH \
#     --cfg-options \
#         test_dataloader.batch_size=$BATCH_SIZE \
#         model.view_recon_cfg.enabled=False
# echo ===============================================

# echo "3. View drop testing - with view reconstruction"
# python tools/test.py \
#     $VIEW_RECON_CONFIG_PATH \
#     $CKPT_2_PATH \
#     --cfg-options \
#         test_dataloader.batch_size=$BATCH_SIZE
# echo ===============================================

# echo "4. All views dropped testing - without view reconstruction"
# python tools/test.py \
#     $VIEW_RECON_CONFIG_PATH \
#     $CKPT_2_PATH \
#     --cfg-options \
#         test_dataloader.batch_size=$BATCH_SIZE \
#         model.view_recon_cfg.num_views_drop=6 \
#         model.view_recon_cfg.enabled=False
# echo ===============================================

# echo "5. Camera only testing - without view reconstruction"
# python tools/test.py \
#     $VIEW_RECON_CONFIG_PATH \
#     $CKPT_2_PATH \
#     --cfg-options \
#         test_dataloader.batch_size=$BATCH_SIZE \
#         model.view_recon_cfg.num_views_drop=0 \
#         model.view_recon_cfg.enabled=False \
#         model.tmp_cfg.drop_pts_feature=True
# echo ===============================================


for i in {1..5}
do
    python tools/test.py \
        $VIEW_RECON_CONFIG_PATH \
        $CKPT_2_PATH \
        --cfg-options \
            test_dataloader.batch_size=$BATCH_SIZE \
            model.view_recon_cfg.num_views_drop=$i \
            model.view_recon_cfg.enabled=False
    echo ===============================================
done
