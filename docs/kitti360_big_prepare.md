## Prepare KITTI360-Big Scene

### Step 1: Downloading

Firstly, create an account at the [KITTI360](https://www.cvlibs.net/datasets/kitti-360/download.php) website for downloading. Then, download the following three components:

- [Perspective Images for Train & Val (128G)](https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/a1d81d9f7fc7195c937f9ad12e2a2c66441ecb4e/download_2d_perspective.zip)

- [Calibrations (3K)](https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/384509ed5413ccc81328cf8c55cc6af078b8c444/calibration.zip)

- [Vechicle Poses (8.9M)](https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/89a6bae3c8a6f789e12de4807fc1e8fdcf182cf4/data_poses.zip)

### Step 2: Prepare the Partition

Specifically, we use the stereo images of sequence `1951-2900` in the `2013_05_28_drive_0009_sync` (a total of 1900 images) for training and evaluation. The extraction is only applied on the `Perspective Images for Train & Val (128G)`. The unzipped `calibration` and `vehicle poses` stay unchanged. 

### Step 3: Organize the Files

Please organize all the ingredients as following in the `data` folder and start the training. 
```bash
./
|-- data
|   |-- KITTI360
|   |   |-- data_2d_nvs_long_challenge
|   |   |   |-- train_00.txt                    # already provided
|   |   |   |-- train_00
|   |   |   |   |-- 2013_05_28_drive_0009_sync
|   |   |   |   |   |-- image_00                # extracted cam0 image sequence
|   |   |   |   |   |   |-- data_rect           
|   |   |   |   |   |   |   |-- 0000001951.png
|   |   |   |   |   |   |   |-- ...
|   |   |   |   |   |-- image_01                # extracted cam1 image sequence
|   |   |   |   |   |   |-- data_rect           
|   |   |   |   |   |   |   |-- 0000001951.png
|   |   |   |   |   |   |   |-- ...
|   |   |-- calibration                         # files from 'Vechicle Poses (8.9M)'
|   |   |   |-- calib_cam_to_pose.txt
|   |   |   |-- calib_cam_to_velo.txt
|   |   |   |-- ...
|   |   |-- data_poses                          # files from `Calibrations (3K)`
|   |   |   |-- 2013_05_28_drive_0009_sync
|   |   |   |-- ...
|-- ...
```