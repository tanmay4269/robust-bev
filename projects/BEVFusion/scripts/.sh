#!/bin/bash

MODE=training

WORKSPACE_PATH=/workspace
CONFIG_PATH=/workspace/projects/BEVFusion/configs/dropping_lidar.py
# CKPT_PATH=projects/BEVFusion/checkpoints/bevfusion-1.pth
CKPT_PATH=projects/BEVFusion/checkpoints/bevfusion-2.pth
TRAINING_BATCH_SIZE=6
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