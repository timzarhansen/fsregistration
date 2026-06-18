import numpy as np
import pandas as pd
import os


def xyzrpy_to_4d(x, y, z, r, p, yaw):
    # Build rotation matrix from Euler angles (XYZ intrinsic sequence)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(r), -np.sin(r)],
        [0, np.sin(r), np.cos(r)]
    ])
    R_y = np.array([
        [np.cos(p), 0, np.sin(p)],
        [0, 1, 0],
        [-np.sin(p), 0, np.cos(p)]
    ])
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    R = R_z @ R_y @ R_x

    # Create homogeneous transformation
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = [x, y, z]
    return T


def T_to_xyzrpy(T):
    # Extract rotation matrix
    R = T[0:3, 0:3]

    # Extract translation vector
    t = T[0:3, 3]

    # Compute Euler angles from rotation matrix (XYZ sequence)
    r = np.arctan2(R[2, 1], R[2, 2])
    p = np.arcsin(-R[2, 0])
    yaw = np.arctan2(R[1, 0], R[0, 0])

    x = t[0]
    y = t[1]
    z = t[2]
    return x, y, z, r, p, yaw


# clc, clear - skipped

margin_rot = 1.4062
margin_mag = 0.75

sequence = np.arange(0, 45)
exclude = [2, 6, 9, 11, 18, 24, 25, 31, 32, 34, 37, 39, 43]

result = sequence[~np.isin(sequence, exclude)]
sequence = result

number_of_all_rot_error = []
number_of_all_trans_error = []
number_of_filtered_rot_error = []
number_of_filtered_trans_error = []
mean_trans = []
std_trans = []
mean_rot = []
std_rot = []
weight = []

all_scans_together = 26598

for sequence_ in sequence:
    str_seq = f"{sequence_:02d}"
    current_folder_name = f"/Users/timhansen/Documents/ros_ws/src/fsregistration/pythonScripts/radarDataset/saveRandomImagesBoreas/{str_seq}_256_75_5"

    gt_scan_matching_pose_list = pd.read_csv(f"{current_folder_name}/gtPoseScanMatching.csv").values
    est_matching_pose_list = pd.read_csv(f"{current_folder_name}/estPoseScanMatching.csv").values

    how_many_images = len(gt_scan_matching_pose_list)

    rot_error_tmp = np.zeros((how_many_images, 3))
    mag_error_tmp = np.zeros((how_many_images, 3))

    for i in range(how_many_images):
        GT_trans = xyzrpy_to_4d(
            gt_scan_matching_pose_list[i, 0],
            gt_scan_matching_pose_list[i, 1],
            0, 0, 0,
            gt_scan_matching_pose_list[i, 2]
        )
        Est_trans = xyzrpy_to_4d(
            est_matching_pose_list[i, 0],
            est_matching_pose_list[i, 1],
            0, 0, 0,
            est_matching_pose_list[i, 2]
        )
        resulting_error_trans = np.linalg.inv(GT_trans) @ Est_trans
        print(i)
        x, y, z, r, p, yaw = T_to_xyzrpy(resulting_error_trans)
        rot_error_tmp[i] = yaw * 180 / np.pi
        mag_error_tmp[i] = np.linalg.norm([x, y])

    [M, I] = np.max(mag_error_tmp), np.argmax(mag_error_tmp)
    [M, I] = np.max(rot_error_tmp), np.argmax(rot_error_tmp)

    rot_errors = rot_error_tmp
    trans_mag = mag_error_tmp

    rot_errors_abs = np.abs(rot_errors)
    is_outlier = np.abs(rot_errors_abs - np.mean(rot_errors_abs)) > margin_rot * np.std(rot_errors_abs)
    outlier_indices = np.where(is_outlier)[0]
    filtered_rot_errors = rot_errors_abs[np.abs(rot_errors_abs - np.mean(rot_errors_abs)) < margin_rot * np.std(rot_errors_abs)]

    is_outlier = np.abs(trans_mag - np.mean(trans_mag)) > margin_mag * np.std(trans_mag)
    outlier_indices = np.where(is_outlier)[0]
    filtered_trans_mag = trans_mag[np.abs(trans_mag - np.mean(trans_mag)) < margin_mag * np.std(trans_mag)]

    number_of_all_trans_error = np.vstack([number_of_all_trans_error, trans_mag]) if len(number_of_all_trans_error) > 0 else trans_mag
    number_of_all_rot_error = np.vstack([number_of_all_rot_error, rot_errors_abs]) if len(number_of_all_rot_error) > 0 else rot_errors_abs

    number_of_filtered_trans_error = np.vstack([number_of_filtered_trans_error, filtered_trans_mag]) if len(number_of_filtered_trans_error) > 0 else filtered_trans_mag
    number_of_filtered_rot_error = np.vstack([number_of_filtered_rot_error, filtered_rot_errors]) if len(number_of_filtered_rot_error) > 0 else filtered_rot_errors

    mean_trans = np.vstack([mean_trans, np.mean(filtered_trans_mag)]) if len(mean_trans) > 0 else np.array([np.mean(filtered_trans_mag)])
    std_trans = np.vstack([std_trans, np.std(filtered_trans_mag)]) if len(std_trans) > 0 else np.array([np.std(filtered_trans_mag)])

    mean_rot = np.vstack([mean_rot, np.mean(filtered_rot_errors)]) if len(mean_rot) > 0 else np.array([np.mean(filtered_rot_errors)])
    std_rot = np.vstack([std_rot, np.std(filtered_rot_errors)]) if len(std_rot) > 0 else np.array([np.std(filtered_rot_errors)])

    weight = np.vstack([weight, how_many_images / all_scans_together]) if len(weight) > 0 else np.array([how_many_images / all_scans_together])

    print(f"\nName Dataset: {current_folder_name}")
    print(f"{round(len(rot_errors), 2):.2f} {round(np.mean(filtered_rot_errors), 2):.2f} {round(np.std(filtered_rot_errors), 2):.2f} {round(100 * (len(rot_errors) - len(filtered_rot_errors)) / len(rot_errors), 2):.2f} {round(np.mean(filtered_trans_mag), 2):.2f} {round(np.std(filtered_trans_mag), 2):.2f} {round(100 * (len(trans_mag) - len(filtered_trans_mag)) / len(trans_mag), 2):.2f}")

length_all_scans = len(number_of_all_trans_error)
outliers_rot = 100 * (len(number_of_all_rot_error) - len(number_of_filtered_rot_error)) / len(number_of_all_rot_error)
outliers_trans = 100 * (len(number_of_all_trans_error) - len(number_of_filtered_trans_error)) / len(number_of_all_trans_error)

mean_mean_trans = np.sum(mean_trans * weight) / np.sum(weight)
mean_std_trans = np.sum(std_trans * weight) / np.sum(weight)
mean_mean_rot = np.sum(mean_rot * weight) / np.sum(weight)
mean_std_rot = np.sum(std_rot * weight) / np.sum(weight)
