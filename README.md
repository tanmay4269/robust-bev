This work isn't a fork of mmdetection3d because I wanted this repo to be private. 
Here's the [official README.md](https://github.com/open-mmlab/mmdetection3d/blob/main/README.md) for mmdetection3d.

To build the image, run this: `docker build -t mmdetection3d docker/` \
To spin up a container, run this: `./docker/run.sh`

# Developement 
## TODO
- [ ] `data/nuscenes/nuscenes_infos_test.pkl` has not been created well, issue: \
    (no issues given on github, likely data issue on my side)
    ```
    Writing to output file: ./data/nuscenes/nuscenes_infos_test.pkl.
    Traceback (most recent call last):
    File "tools/create_data.py", line 359, in <module>
        max_sweeps=args.max_sweeps)
    File "tools/create_data.py", line 81, in nuscenes_data_prep
        update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_test_path)
    File "/mmdetection3d/tools/dataset_converters/update_infos_to_v2.py", line 1148, in update_pkl_infos
        update_nuscenes_infos(pkl_path=pkl_path, out_dir=out_dir)
    File "/mmdetection3d/tools/dataset_converters/update_infos_to_v2.py", line 376, in update_nuscenes_infos
        print(f'ignore classes: {ignore_class_name}')
    UnboundLocalError: local variable 'ignore_class_name' referenced before assignment
    ```