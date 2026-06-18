import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# clc, clear - skipped

folder_name = "00_512_25_2"

gt_pose_absolute_list = pd.read_csv(f"{folder_name}/absolutePoseListGT.csv").values

how_many_images = 200

absolute_poses_gt = np.zeros((how_many_images + 1, 3))
for i in range(2, how_many_images + 1):
    x = gt_pose_absolute_list[i - 1, 0]
    y = gt_pose_absolute_list[i - 1, 1]
    yaw = gt_pose_absolute_list[i - 1, 2]

    absolute_poses_gt[i, :] = [x, y, yaw]

plt.figure(1)
plt.plot(absolute_poses_gt[1:, 0], absolute_poses_gt[1:, 1])
plt.axis('equal')

gt_scan_matching_pose_list = pd.read_csv(f"{folder_name}/gtPoseScanMatching.csv").values
absolute_poses_scan_matching_gt = np.zeros((how_many_images + 1, 3))
for i in range(2, how_many_images + 1):
    x = gt_scan_matching_pose_list[i - 1, 0]
    y = gt_scan_matching_pose_list[i - 1, 1]
    yaw = gt_scan_matching_pose_list[i - 1, 2]
    T_new = np.array([
        [np.cos(yaw), -np.sin(yaw), x],
        [np.sin(yaw), np.cos(yaw), y],
        [0, 0, 1]
    ])

    x_old = absolute_poses_scan_matching_gt[i - 1, 0]
    y_old = absolute_poses_scan_matching_gt[i - 1, 1]
    yaw_old = absolute_poses_scan_matching_gt[i - 1, 2]
    T_old = np.array([
        [np.cos(yaw_old), -np.sin(yaw_old), x_old],
        [np.sin(yaw_old), np.cos(yaw_old), y_old],
        [0, 0, 1]
    ])

    absolute_transformations = T_old @ T_new
    x_new = absolute_transformations[0, 2]
    y_new = absolute_transformations[1, 2]
    yaw_new = np.arctan2(absolute_transformations[1, 0], absolute_transformations[0, 0])
    absolute_poses_scan_matching_gt[i, :] = [x_new, y_new, yaw_new]

plt.figure(2)
plt.plot(absolute_poses_scan_matching_gt[:, 0], absolute_poses_scan_matching_gt[:, 1])
plt.axis('equal')
