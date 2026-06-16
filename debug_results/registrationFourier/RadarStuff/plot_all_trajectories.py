import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess


def get_new_absolute_pose(pose_old, pose_new):
    x_old = pose_old[0]
    y_old = pose_old[1]
    yaw_old = pose_old[2]
    T_old = np.array([
        [np.cos(yaw_old), -np.sin(yaw_old), x_old],
        [np.sin(yaw_old), np.cos(yaw_old), y_old],
        [0, 0, 1]
    ])
    x = pose_new[0]
    y = pose_new[1]
    yaw = pose_new[2]
    T_new = np.array([
        [np.cos(yaw), -np.sin(yaw), x],
        [np.sin(yaw), np.cos(yaw), y],
        [0, 0, 1]
    ])
    absolute_transformations = T_old @ T_new
    x_new = absolute_transformations[0, 2]
    y_new = absolute_transformations[1, 2]
    yaw_new = np.arctan2(absolute_transformations[1, 0], absolute_transformations[0, 0])
    return_pose = [x_new, y_new, yaw_new]
    return return_pose


sequence = np.arange(0, 2)
dataset_name = "_256_75_5"

for sequence_ in sequence:
    str_seq = f"{sequence_:02d}"
    current_folder_name = f"/Users/timhansen/Documents/ros_ws/src/fsregistration/pythonScripts/radarDataset/saveRandomImagesBoreas/{str_seq}{dataset_name}"

    gt_scan_matching_pose_list = pd.read_csv(f"{current_folder_name}/gtPoseScanMatching.csv").values
    est_matching_pose_list = pd.read_csv(f"{current_folder_name}/estPoseScanMatching.csv").values

    how_many_images = len(gt_scan_matching_pose_list) - 1
    absolute_poses_scan_matching_est = np.zeros((how_many_images + 1, 3))
    absolute_poses_scan_matching_gt = np.zeros((how_many_images + 1, 3))

    for i in range(2, how_many_images + 1):
        absolute_poses_scan_matching_gt[i, :] = get_new_absolute_pose(
            absolute_poses_scan_matching_gt[i - 1, :],
            gt_scan_matching_pose_list[i - 1, :]
        )
        absolute_poses_scan_matching_est[i, :] = get_new_absolute_pose(
            absolute_poses_scan_matching_est[i - 1, :],
            est_matching_pose_list[i - 1, :]
        )

    plt.figure(33)
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(450/100, 500/100)

    plot_sequence_from = 1
    plot_sequence_to = how_many_images

    plt.hold(True)
    linewidth = 1.5
    plt.plot(
        absolute_poses_scan_matching_est[plot_sequence_from:plot_sequence_to, 0],
        -absolute_poses_scan_matching_est[plot_sequence_from:plot_sequence_to, 1],
        linewidth=linewidth
    )
    plt.plot(
        absolute_poses_scan_matching_gt[plot_sequence_from:plot_sequence_to, 0],
        -absolute_poses_scan_matching_gt[plot_sequence_from:plot_sequence_to, 1],
        linewidth=linewidth
    )

    plt.ylabel("m")
    plt.xlabel("m")
    plt.legend(['Estimate', 'GT'], loc='upper left')
    plt.axis('equal')
    plt.box(True)
    plt.grid(True)

    name_of_file = f"/Users/timhansen/Documents/MATLAB/matlabTestEnvironment/registrationFourier/RadarStuff/resultFigs/trajectory{str_seq}{dataset_name}"
    plt.savefig(f"{name_of_file}.pdf")
    system_command = f"pdfcrop {name_of_file}.pdf {name_of_file}.pdf"
    saving_command = f"echo '{system_command}' >> resultFigs/commands.sh"
    subprocess.run(saving_command, shell=True)
