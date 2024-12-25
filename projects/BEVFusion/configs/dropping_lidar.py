_base_ = [
    './bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

model = dict(
    pts_dropout=dict(
        type="2d",  # 1d or 2d
        probs=dict(
            train=0.3,
            val=0.3
        )
    )
)