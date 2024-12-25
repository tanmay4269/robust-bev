#!/bin/bash

BATCH_SIZE=14

WORKSPACE_PATH=/workspace
CONFIG_PATH=projects/BEVFusion/configs/dropping_lidar.py
CKPT_PATH=projects/BEVFusion/checkpoints/bevfusion-2.pth

cd $WORKSPACE_PATH
export PYTHONPATH=$PYTHONPATH:$WORKSPACE_PATH
for val in 0.25 0.5 0.75; do
    python tools/test.py \
        $CONFIG_PATH \
        $CKPT_PATH \
        --cfg-options \
            model.pts_dropout_probs.val=$val \
            test_dataloader.batch_size=$BATCH_SIZE
    echo ============================
done