_base_ = [
    './bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

model = dict(
    pts_dropout=dict(
        type="1d",  # 1d or 2d
        probs=dict(
            train=1.0,
            val=1.0
        )
    )
)

train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=1)
val_cfg = dict()
test_cfg = dict()
