import numpy as np
from pyboreas import BoreasDataset
# import sys
# import multiprocessing
#
# multiprocessing.set_start_method('spawn', force=True)


# root = '/Users/timhansen/Documents/dataFolder/radar_boreas/'
root = '/home/tim-external/dataFolder/radar_boreas/'
# root = '/Users/timhansen/Documents/dataFolder/radar_boreas/boreas-2021-09-02-11-42/'
# root = '/home/deeprobotics/dataFolder/radar_data/radar_boreas/'

bd = BoreasDataset(root)

# Note: The Boreas dataset differs from others (KITTI) in that camera,
# lidar, and radar measurements are not synchronous. However, each
# sensor message has an accurate timestamp and pose instead.
# See our tutorials for how to work with multiple sensors.

# Loop through each frame in order (odometry)
for seq in bd.sequences:
    # Iterator examples:
    # for camera_frame in seq.camera:
    #     img = camera_frame.img  # np.ndarray
    #     # do something
    #     camera_frame.unload_data() # Memory reqs will keep increasing without this
    # for lidar_frame in seq.lidar:
    #     pts = lidar_frame.points  # np.ndarray (x,y,z,i,r,t)
    #     # do something
    #     lidar_frame.unload_data() # Memory reqs will keep increasing without this
    # Retrieve frames based on their index:
    N = len(seq.radar_frames)
    for i in range(N):
        radar_frame = seq.get_radar(i)
        # do something
        radar_frame.unload_data() # Memory reqs will keep increasing without this

# Iterator example:
# cam_iter = bd.sequences[0].get_camera_iter()
# cam0 = next(cam_iter)  # First camera frame
# cam1 = next(cam_iter)  # Second camera frame

# Randomly access frames (deep learning, localization):
# N = len(bd.lidar_frames)
# indices = np.random.permutation(N)
# for idx in indices:
#     lidar_frame = bd.get_lidar(idx)
#     do something
    # lidar_frame.unload_data() # Memory reqs will keep increasing without this

# Each sequence contains a calibration object:
# calib = bd.sequences[0].calib
# point_lidar = np.array([1, 0, 0, 1]).reshape(4, 1)
# point_camera = np.matmul(calib.T_camera_lidar, point_lidar)

# Each sensor frame has a timestamp, groundtruth pose
# (4x4 homogeneous transform) wrt a global coordinate frame (ENU),
# and groundtruth velocity information. Unless it's a part of the test set,
# in that case, ground truth poses will be missing. However we still provide IMU
# data (in the applanix frame) through the imu.csv files.
lidar_frame = bd.get_lidar(0)
t = lidar_frame.timestamp  # timestamp in seconds
T_enu_lidar = lidar_frame.pose  # 4x4 homogenous transform [R t; 0 0 0 1]
vbar = lidar_frame.velocity  # 6x1 vel in ENU frame [v_se_in_e; w_se_in_e]
varpi = lidar_frame.body_rate  # 6x1 vel in sensor frame [v_se_in_s; w_se_in_s]