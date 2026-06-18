import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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


def pcshowpair_alpha(pc1, pc2, background_color='white', projection='orthographic', view_plane='XY', alpha=1.0):
    if hasattr(pc1, 'points'):
        pc1 = np.asarray(pc1.points)
    if hasattr(pc2, 'points'):
        pc2 = np.asarray(pc2.points)

    pc1 = pc1[~np.isnan(pc1).any(axis=1)]
    pc2 = pc2[~np.isnan(pc2).any(axis=1)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], c='r', s=5, alpha=alpha)
    ax.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], c='b', s=5, alpha=alpha)

    ax.set_facecolor(background_color)
    ax.set_aspect('equal')

    if view_plane.upper() == 'XY':
        ax.view_init(elev=90, azim=-90)
    elif view_plane.upper() == 'XZ':
        ax.view_init(elev=0, azim=0)
    elif view_plane.upper() == 'YZ':
        ax.view_init(elev=0, azim=90)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def xyzrpy_to_4d(x, y, z, r, p, yaw):
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

    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = [x, y, z]
    return T


def pose_transformation_4d(pose1, pose2):
    x1 = pose1[0]
    y1 = pose1[1]
    t1 = pose1[2]
    x2 = pose2[0]
    y2 = pose2[1]
    t2 = pose2[2]

    T1 = np.eye(4)
    Rz_t1 = np.array([
        [np.cos(t1), -np.sin(t1), 0],
        [np.sin(t1), np.cos(t1), 0],
        [0, 0, 1]
    ])
    T1[0:3, 0:3] = Rz_t1
    T1[0, 3] = x1
    T1[1, 3] = y1

    T2 = np.eye(4)
    Rz_t2 = np.array([
        [np.cos(t2), -np.sin(t2), 0],
        [np.sin(t2), np.cos(t2), 0],
        [0, 0, 1]
    ])
    T2[0:3, 0:3] = Rz_t2
    T2[0, 3] = x2
    T2[1, 3] = y2

    T = np.linalg.inv(T1) @ T2
    return T


def radar_to_point_cloud(radar_scan, threshold, pixel_size, color):
    radar_scan = (radar_scan - radar_scan.min()) / (radar_scan.max() - radar_scan.min())
    rows, cols = np.where(radar_scan > threshold)

    points = []
    for r, c in zip(rows, cols):
        x = c * pixel_size
        y = r * pixel_size
        points.append([x, y, 0])

    points = np.array(points)
    if len(points) == 0:
        return np.array([])

    colors = np.tile(color, (len(points), 1))
    return np.hstack([points, colors])


# clc, clear - skipped

folder_name = "01_256_75_5"

gt_scan_matching_pose_list = pd.read_csv(f"{folder_name}/gtPoseScanMatching.csv").values
est_matching_pose_list = pd.read_csv(f"{folder_name}/estPoseScanMatching.csv").values

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

plt.figure(1)
plt.clf()
plt.plot(absolute_poses_scan_matching_est[:, 0], absolute_poses_scan_matching_est[:, 1])
plt.axis('equal')
plt.title("FS2D " + folder_name)

plt.figure(2)
plt.clf()
plt.plot(absolute_poses_scan_matching_gt[:, 0], absolute_poses_scan_matching_gt[:, 1])
plt.axis('equal')
plt.title("Ground Truth")

plt.figure(3)
plt.clf()
plot_sequence_from = 1
plot_sequence_to = how_many_images
plt.hold(True)
linewidth = 1.5
plt.plot(
    absolute_poses_scan_matching_est[plot_sequence_from:plot_sequence_to, 0],
    absolute_poses_scan_matching_est[plot_sequence_from:plot_sequence_to, 1],
    linewidth=linewidth
)
plt.plot(
    absolute_poses_scan_matching_gt[plot_sequence_from:plot_sequence_to, 0],
    absolute_poses_scan_matching_gt[plot_sequence_from:plot_sequence_to, 1],
    linewidth=linewidth
)

plt.ylabel("m")
plt.xlabel("m")
plt.legend(['Estimate', 'GT'], loc='upper left')
plt.axis('equal')
plt.title(folder_name)
plt.box(True)
plt.grid(True)
fig = plt.gcf()
fig.set_size_inches(800/100, 500/100)

name_of_file = f"/Users/timhansen/Documents/MATLAB/matlabTestEnvironment/registrationFourier/RadarStuff/resultFigs/trajectory{folder_name}"
plt.savefig(f"{name_of_file}.pdf")
system_command = f"pdfcrop {name_of_file}.pdf {name_of_file}.pdf"
saving_command = f"echo '{system_command}' >> resultFigs/commands.sh"
subprocess.run(saving_command, shell=True)

plt.figure(5)
plt.clf()
plot_sequence_from = 1
plot_sequence_to = 80
plt.hold(True)
plt.plot(
    absolute_poses_scan_matching_est[plot_sequence_from:plot_sequence_to, 0],
    absolute_poses_scan_matching_est[plot_sequence_from:plot_sequence_to, 1],
    linewidth=linewidth
)
plt.plot(
    absolute_poses_scan_matching_gt[plot_sequence_from:plot_sequence_to, 0],
    absolute_poses_scan_matching_gt[plot_sequence_from:plot_sequence_to, 1],
    linewidth=linewidth
)

plt.ylabel("m")
plt.xlabel("m")
plt.legend(['Estimate', 'GT'], loc='upper left')
plt.axis('equal')
plt.title(folder_name + " Zoom")
plt.box(True)
plt.grid(True)
fig = plt.gcf()
fig.set_size_inches(800/100, 500/100)

name_of_file = f"/Users/timhansen/Documents/MATLAB/matlabTestEnvironment/registrationFourier/RadarStuff/resultFigs/zoom{folder_name}"
plt.savefig(f"{name_of_file}.pdf")
system_command = f"pdfcrop {name_of_file}.pdf {name_of_file}.pdf"
saving_command = f"echo '{system_command}' >> resultFigs/commands.sh"
subprocess.run(saving_command, shell=True)

# Computing Box Plot
rot_errors = np.genfromtxt(f"{folder_name}/rotationErrorList.csv")

with open(f"{folder_name}/translationErrorList.csv", 'r') as f:
    lines = f.readlines()
    trans_data = [list(map(float, line.strip().split())) for line in lines if line.strip()]
    nx2_matrix = np.array(trans_data)

trans_mag = np.sqrt(np.sum(nx2_matrix ** 2, axis=1))

plt.figure(4)
plt.clf()

ax1 = plt.gca()
ax1.boxplot([trans_mag, np.full(len(trans_mag), np.nan)], labels=["", ""])
plt.title('error margins')
plt.xlabel(folder_name)
plt.ylabel('Error [m]')

ax2 = ax1.twinx()
ax2.boxplot([np.full(len(rot_errors), np.nan), np.abs(rot_errors)], labels=["trans", "rot"])
ax2.set_ylabel('Error [°]')

ax1.set_yscale('log')
ax2.set_yscale('log')

name_of_file = f"/Users/timhansen/Documents/MATLAB/matlabTestEnvironment/registrationFourier/RadarStuff/resultFigs/boxplot{folder_name}"
plt.savefig(f"{name_of_file}.pdf")
system_command = f"pdfcrop {name_of_file}.pdf {name_of_file}.pdf"
saving_command = f"echo '{system_command}' >> resultFigs/commands.sh"
subprocess.run(saving_command, shell=True)

# Blending Images
skip = 5
which_to_blend = 3480

threshold = 0.2
pixel_size = 0.75
image_1_number = f"{which_to_blend:04d}"
image_2_number = f"{which_to_blend + skip:04d}"

plt.figure(6)
resulting_string_radar1 = f"{folder_name}/original_image_{image_1_number}.png"
radar_scan1 = mpimg.imread(resulting_string_radar1)
radar_scan1 = (radar_scan1 - radar_scan1.min()) / (radar_scan1.max() - radar_scan1.min())
plt.imshow(radar_scan1)

resulting_string_radar2 = f"{folder_name}/original_image_{image_2_number}.png"
radar_scan2 = mpimg.imread(resulting_string_radar2)
radar_scan2 = (radar_scan2 - radar_scan2.min()) / (radar_scan2.max() - radar_scan2.min())

plt.figure(7)
plt.imshow(radar_scan2)

plt.figure(8)
plt.clf()
pc1 = radar_to_point_cloud(radar_scan1, threshold, pixel_size, [1, 0, 0])
pc2 = radar_to_point_cloud(radar_scan2, threshold, pixel_size, [0, 0, 1])

pose_diff = gt_scan_matching_pose_list[which_to_blend // skip + 1, :]
pose_diff_4d = np.linalg.inv(xyzrpy_to_4d(pose_diff[0], pose_diff[1], 0, 0, 0, pose_diff[2]))

pose_diff_4d[0:3, 3] = np.linalg.inv(pose_diff_4d[0:3, 0:3]) @ pose_diff_4d[0:3, 3]
x_tmp = pose_diff_4d[0, 3]
y_tmp = pose_diff_4d[1, 3]
pose_diff_4d[0, 3] = -y_tmp
pose_diff_4d[1, 3] = -x_tmp

pcshowpair_alpha(pc1, pc2, background_color="white", projection="orthographic", view_plane="XY", alpha=0.3)
plt.title("Difference Between Two Point Clouds GT")
plt.xlabel("X(m)")
plt.ylabel("Y(m)")

plt.axis('equal')
plt.box(True)
plt.grid(True)
fig = plt.gcf()
fig.set_size_inches(600/100, 500/100)

name_of_file = f"/Users/timhansen/Documents/MATLAB/matlabTestEnvironment/registrationFourier/RadarStuff/resultFigs/blendedPCL_GT{folder_name}_{which_to_blend}"
plt.savefig(f"{name_of_file}.pdf")
system_command = f"pdfcrop {name_of_file}.pdf {name_of_file}.pdf"
saving_command = f"echo '{system_command}' >> resultFigs/commands.sh"
subprocess.run(saving_command, shell=True)

plt.figure(9)
plt.clf()
pc1 = radar_to_point_cloud(radar_scan1, threshold, pixel_size, [1, 0, 0])
pc2 = radar_to_point_cloud(radar_scan2, threshold, pixel_size, [0, 0, 1])
pose_diff = est_matching_pose_list[which_to_blend // skip + 1, :]
pose_diff_4d = np.linalg.inv(xyzrpy_to_4d(pose_diff[0], pose_diff[1], 0, 0, 0, pose_diff[2]))

pose_diff_4d[0:3, 3] = np.linalg.inv(pose_diff_4d[0:3, 0:3]) @ pose_diff_4d[0:3, 3]
x_tmp = pose_diff_4d[0, 3]
y_tmp = pose_diff_4d[1, 3]
pose_diff_4d[0, 3] = -y_tmp
pose_diff_4d[1, 3] = -x_tmp

pcshowpair_alpha(pc1, pc2, background_color="white", projection="orthographic", view_plane="XY", alpha=0.3)
plt.title("Difference Between Two Point Clouds EST")
plt.xlabel("X(m)")
plt.ylabel("Y(m)")

plt.axis('equal')
plt.box(True)
plt.grid(True)
fig = plt.gcf()
fig.set_size_inches(600/100, 500/100)

name_of_file = f"/Users/timhansen/Documents/MATLAB/matlabTestEnvironment/registrationFourier/RadarStuff/resultFigs/blendedPCL_EST{folder_name}_{which_to_blend}"
plt.savefig(f"{name_of_file}.pdf")
system_command = f"pdfcrop {name_of_file}.pdf {name_of_file}.pdf"
saving_command = f"echo '{system_command}' >> resultFigs/commands.sh"
subprocess.run(saving_command, shell=True)

# Error computation for tabular Custom
margin_rot = 1.4062
margin_mag = 0.75
rot_errors_abs = np.abs(rot_errors)
is_outlier = np.abs(rot_errors_abs - np.mean(rot_errors_abs)) > margin_rot * np.std(rot_errors_abs)
filtered_rot_errors = rot_errors_abs[np.abs(rot_errors_abs - np.mean(rot_errors_abs)) < margin_rot * np.std(rot_errors_abs)]

is_outlier = np.abs(trans_mag - np.mean(trans_mag)) > margin_mag * np.std(trans_mag)
filtered_trans_mag = trans_mag[np.abs(trans_mag - np.mean(trans_mag)) < margin_mag * np.std(trans_mag)]

print(f'Name Dataset: {folder_name}')
print('Custom Method:')
print(f'Rotation Mean: {np.mean(filtered_rot_errors):.4f}')
print(f'Rotation Std: {np.std(filtered_rot_errors):.4f}')
print(f'Translation Mean: {np.mean(filtered_trans_mag):.4f}')
print(f'Translation Std: {np.std(filtered_trans_mag):.4f}')
print(f'numberOutlier Rotation: {len(rot_errors) - len(filtered_rot_errors):.4f}')
print(f'numberOutlier Translation: {len(trans_mag) - len(filtered_trans_mag):.4f}')
print(f'Number Of Scan Pairs: {len(rot_errors):.4f}')

# Read all sequences
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

for sequence_ in sequence:
    str_seq = f"{sequence_:02d}"
    current_folder_name = f"/Users/timhansen/Documents/ros_ws/src/fsregistration/pythonScripts/radarDataset/saveRandomImagesBoreas/{str_seq}_256_75_5"

    rot_errors = np.genfromtxt(f"{current_folder_name}/rotationErrorList.csv")

    with open(f"{current_folder_name}/translationErrorList.csv", 'r') as f:
        lines = f.readlines()
        trans_data = [list(map(float, line.strip().split())) for line in lines if line.strip()]
        nx2_matrix = np.array(trans_data)

    trans_mag = np.sqrt(np.sum(nx2_matrix ** 2, axis=1))

    rot_errors_abs = np.abs(rot_errors)
    is_outlier = np.abs(rot_errors_abs - np.mean(rot_errors_abs)) > margin_rot * np.std(rot_errors_abs)
    filtered_rot_errors = rot_errors_abs[np.abs(rot_errors_abs - np.mean(rot_errors_abs)) < margin_rot * np.std(rot_errors_abs)]

    is_outlier = np.abs(trans_mag - np.mean(trans_mag)) > margin_mag * np.std(trans_mag)
    filtered_trans_mag = trans_mag[np.abs(trans_mag - np.mean(trans_mag)) < margin_mag * np.std(trans_mag)]

    number_of_all_trans_error = np.vstack([number_of_all_trans_error, trans_mag]) if len(number_of_all_trans_error) > 0 else trans_mag
    number_of_all_rot_error = np.vstack([number_of_all_rot_error, rot_errors_abs]) if len(number_of_all_rot_error) > 0 else rot_errors_abs

    number_of_filtered_trans_error = np.vstack([number_of_filtered_trans_error, filtered_trans_mag]) if len(number_of_filtered_trans_error) > 0 else filtered_trans_mag
    number_of_filtered_rot_error = np.vstack([number_of_filtered_rot_error, filtered_rot_errors]) if len(number_of_filtered_rot_error) > 0 else filtered_rot_errors

    mean_trans = np.vstack([mean_trans, np.mean(filtered_trans_mag)]) if len(mean_trans) > 0 else np.array([np.mean(filtered_trans_mag)])
    std_trans = np.vstack([std_trans, np.std(filtered_trans_mag)]) if len(std_trans) > 0 else np.array([np.std(filtered_trans_mag)])

    mean_rot = np.vstack([mean_rot, np.mean(filtered_rot_errors)]) if len(mean_rot) > 0 else np.array([np.mean(filtered_rot_errors)])
    std_rot = np.vstack([std_rot, np.std(filtered_rot_errors)]) if len(std_rot) > 0 else np.array([np.std(filtered_rot_errors)])

    print(f'\nName Dataset: {current_folder_name}')
    print(f'{len(rot_errors):.4f} {np.mean(filtered_rot_errors):.4f} {np.std(filtered_rot_errors):.4f} {100 * (len(rot_errors) - len(filtered_rot_errors)) / len(rot_errors):.4f} {np.mean(filtered_trans_mag):.4f} {np.std(filtered_trans_mag):.4f} {100 * (len(trans_mag) - len(filtered_trans_mag)) / len(trans_mag):.4f}')

length_all_scans = len(number_of_all_trans_error)
outliers_rot = 100 * (len(number_of_all_rot_error) - len(number_of_filtered_rot_error)) / len(number_of_all_rot_error)
outliers_trans = 100 * (len(number_of_all_trans_error) - len(number_of_filtered_trans_error)) / len(number_of_all_trans_error)
mean_mean_trans = np.mean(mean_trans)
mean_std_trans = np.mean(std_trans)
mean_mean_rot = np.mean(mean_rot)
mean_std_rot = np.mean(std_rot)
