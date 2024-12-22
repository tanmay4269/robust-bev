#!/bin/bash

MODE=training

WORKSPACE_PATH=/workspace
CONFIG_PATH=projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
CKPT_PATH=/workspace/projects/BEVFusion/checkpoints/bevfusion-trained.pth
# CKPT_PATH=/workspace/projects/BEVFusion/checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth
TRAINING_BATCH_SIZE=1
TESTING_BATCH_SIZE=6

cd $WORKSPACE_PATH
export PYTHONPATH=$PYTHONPATH:$WORKSPACE_PATH
if [ $MODE == "training" ]; then
    python tools/train.py \
        $CONFIG_PATH \
        --cfg-options \
            load_from=$CKPT_PATH \
            train_dataloader.batch_size=$TRAINING_BATCH_SIZE
elif [ $MODE == "testing" ]; then
    python tools/test.py \
        $CONFIG_PATH \
        $CKPT_PATH \
        --cfg-options \
            test_dataloader.batch_size=$TESTING_BATCH_SIZE
else
    echo "Invalid mode, choose between 'training' or 'testing'"
fi