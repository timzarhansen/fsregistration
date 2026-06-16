import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def add_command_to_batchfile(fullpath, command, append=False):
    mode = 'a' if append else 'w'
    with open(fullpath, mode) as fid:
        fid.write(f"{command}\n")


def rotation_matrix(roll, pitch, yaw):
    roll_rad = np.radians(roll)
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0, np.sin(roll_rad), np.cos(roll_rad)]
    ])

    R_y = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])

    R_z = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])

    return R_z @ R_y @ R_x


def plot_pdfs(name_of_the_dataset, name_of_batch_file, name_of_the_titel_list, name_of_result_file=None):
    ground_truth_raw = np.genfromtxt(f"csvFiles/groundTruthOverTime{name_of_the_dataset}.csv")
    position_estimate_ekf_raw = np.genfromtxt(f"csvFiles/positionEstimationOverTime{name_of_the_dataset}.csv")
    angle_and_intensities_raw = np.genfromtxt(f"csvFiles/angleAndIntensities{name_of_the_dataset}.csv")
    number_of_markers = np.genfromtxt(f"csvFiles/numberOfMarkers{name_of_the_dataset}.csv")

    position_first_gt_matrix = 3
    while np.isnan(ground_truth_raw[4 * position_first_gt_matrix, 0]):
        position_first_gt_matrix += 1

    number_of_sonar_measurements = ground_truth_raw.shape[0] // 4

    first_gt_matrix = ground_truth_raw[4 * position_first_gt_matrix:4 * position_first_gt_matrix + 4, 0:4]
    estimate_ekf_first = position_estimate_ekf_raw[4 * position_first_gt_matrix:4 * position_first_gt_matrix + 4, 0:4]
    change_of_ekf_matrix = first_gt_matrix @ np.linalg.inv(estimate_ekf_first)

    x_gt = np.zeros(number_of_sonar_measurements)
    y_gt = np.zeros(number_of_sonar_measurements)
    x_ekf = np.zeros(number_of_sonar_measurements)
    y_ekf = np.zeros(number_of_sonar_measurements)
    rotation_matrix_ekf = np.zeros((number_of_sonar_measurements, 3, 3))
    rotation_matrix_egt = np.zeros((number_of_sonar_measurements, 3, 3))

    for i in range(number_of_sonar_measurements):
        x_gt[i] = ground_truth_raw[i * 4, 3]
        y_gt[i] = ground_truth_raw[i * 4 + 1, 3]
        tmp_ekf_matrix = change_of_ekf_matrix @ position_estimate_ekf_raw[i * 4:i * 4 + 4, 0:4]
        x_ekf[i] = tmp_ekf_matrix[0, 3]
        y_ekf[i] = tmp_ekf_matrix[1, 3]
        rotation_matrix_ekf[i, 0:3, 0:3] = tmp_ekf_matrix[0:3, 0:3]
        rotation_matrix_egt[i, 0:3, 0:3] = ground_truth_raw[i * 4:i * 4 + 3, 0:3]

    yaw_error = []
    error_x = []
    error_y = []
    norm_error = []

    for i in range(2, number_of_sonar_measurements):
        if (not np.isnan(x_gt[i]) and not np.isnan(y_gt[i]) and
            not np.isnan(x_ekf[i]) and not np.isnan(y_ekf[i]) and
            not np.isnan(rotation_matrix_ekf[i]).any() and
            not np.isnan(rotation_matrix_egt[i]).any()):
            error_rotation_matrix = rotation_matrix_ekf[i] @ np.linalg.inv(rotation_matrix_egt[i])
            yaw_error.append(np.arctan2(error_rotation_matrix[1, 0], error_rotation_matrix[0, 0]))
            error_x.append(x_gt[i] - x_ekf[i])
            error_y.append(y_gt[i] - y_ekf[i])
            norm_error.append(np.sqrt((x_gt[i] - x_ekf[i]) ** 2 + (y_gt[i] - y_ekf[i]) ** 2))

    error_x = np.array(error_x)
    error_y = np.array(error_y)
    norm_error = np.array(norm_error)
    yaw_error = np.array(yaw_error)

    print(name_of_the_dataset)
    print(np.mean(np.abs(error_x)))
    print(np.std(np.abs(error_x)))
    print(np.mean(np.abs(error_y)))
    print(np.std(np.abs(error_y)))
    print(np.mean(np.abs(norm_error)))
    print(np.std(np.abs(norm_error)))
    print(np.mean(np.abs(yaw_error)))
    print(np.std(np.abs(yaw_error)))
    print(norm_error[-1])

    plt.figure(1)
    plt.clf()
    plt.hold(True)
    plt.plot(x_gt, y_gt)
    plt.plot(x_ekf, y_ekf)
    plt.legend(['GT', 'EKF'], loc='upper left')
    plt.axis('equal')

    number_of_pixels = 256
    size_of_map = 45
    size_of_cell = size_of_map / number_of_pixels
    map_gt = np.zeros((number_of_pixels, number_of_pixels))
    map_index_gt = np.zeros((number_of_pixels, number_of_pixels))
    position_robot_save = np.zeros((number_of_sonar_measurements, 4, 4))

    for i in range(position_first_gt_matrix, number_of_sonar_measurements):
        current_sonar_measurement = angle_and_intensities_raw[i, :]
        angle_sonar = current_sonar_measurement[1]
        angle_sonar_matrix_tmp = rotation_matrix(0, 0, 0) @ rotation_matrix(0, 0, angle_sonar)
        angle_sonar_matrix = np.eye(4)
        angle_sonar_matrix[0:3, 0:3] = angle_sonar_matrix_tmp
        range_val = current_sonar_measurement[0]
        position_robot = first_gt_matrix @ np.linalg.inv(first_gt_matrix) @ ground_truth_raw[(i - 1) * 4:(i - 1) * 4 + 4, 0:4]

        rotation_matrix_90_degree = np.eye(4)
        rotation_matrix_90_degree[0:3, 0:3] = rotation_matrix(np.pi, 0, 0) @ rotation_matrix(0, 0, -np.pi / 2)
        position_robot_save[i, 0:4, 0:4] = rotation_matrix_90_degree @ position_robot

        if not np.isnan(position_robot[0, 0]):
            number_of_intensity_values = current_sonar_measurement.shape[0] - 2
            for j in range(10, number_of_intensity_values):
                position_of_intensity = position_robot @ angle_sonar_matrix @ np.array([j * range_val / number_of_intensity_values, 0, 0, 1])
                position_in_pixel = position_of_intensity / size_of_cell
                x_pos_in_map = round(position_in_pixel[0])
                y_pos_in_map = round(position_in_pixel[1])
                x_index_map = round(x_pos_in_map + number_of_pixels / 2)
                y_index_map = round(y_pos_in_map + number_of_pixels / 2)
                if 0 < x_index_map < number_of_pixels and 0 < y_index_map < number_of_pixels:
                    map_gt[x_index_map, y_index_map] += current_sonar_measurement[j + 1]
                    map_index_gt[x_index_map, y_index_map] += 1

    max_map = 0
    for i in range(number_of_pixels):
        for j in range(number_of_pixels):
            if map_index_gt[i, j] > 0:
                map_gt[i, j] /= map_index_gt[i, j]
                if map_gt[i, j] > max_map:
                    max_map = map_gt[i, j]

    for i in range(number_of_pixels):
        for j in range(number_of_pixels):
            map_gt[i, j] /= max_map

    fig, ax1 = plt.subplots()
    ax1.imshow(map_gt, extent=[-size_of_map/2, size_of_map/2, -size_of_map/2, size_of_map/2], origin='lower')
    ax1.hold(True)

    positions_of_cameras = np.array([[-8.54974, 8.68527], [-8.49616, 4.99751], [-8.39248, -3.68944],
                                     [2.08273, -6.38497], [-2.46475, -6.44979], [-7.45200, -6.50915],
                                     [9.57199, -5.44995], [9.50768, 0.69213], [9.55034, -3.12613],
                                     [9.45405, 3.97318], [-8.43350, 0.45310], [9.38556, 8.88193]])
    for i in range(positions_of_cameras.shape[0]):
        positions_of_cameras[i, 0], positions_of_cameras[i, 1] = positions_of_cameras[i, 1], positions_of_cameras[i, 0]
    ax1.plot(positions_of_cameras[:, 0], positions_of_cameras[:, 1], 'rx', linewidth=2)

    ax2 = fig.add_axes(ax1.get_position(), frameon=False)
    c = number_of_markers
    ax2.scatter(position_robot_save[:, 0, 3], position_robot_save[:, 1, 3], s=4, c=c, cmap='hot')
    ax2.set_visible(False)
    ax2.invert_yaxis()

    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)

    fig.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=ax1, orientation='vertical', position=[0.05, 0.11, 0.0675, 0.815])
    fig.colorbar(plt.cm.ScalarMappable(cmap='hot'), ax=ax2, orientation='vertical', position=[0.88, 0.11, 0.0675, 0.815])

    ax1.set_aspect('equal')
    ax1.set_title(f'Dataset: {name_of_the_titel_list}')

    name_of_pdf_file = f'/home/ws/matlab/registrationFourier/datasetReleaseJournal/PDFS/GTMap{name_of_the_dataset}'
    plt.savefig(f'{name_of_pdf_file}.pdf')
    add_command_to_batchfile(name_of_batch_file, f'pdfcrop {name_of_pdf_file}.pdf {name_of_pdf_file}.pdf &', True)

    plt.figure(3)
    plt.clf()

    map_ekf = np.zeros((number_of_pixels, number_of_pixels))
    map_index_ekf = np.zeros((number_of_pixels, number_of_pixels))
    position_robot_save_ekf = np.zeros((number_of_sonar_measurements, 4, 4))

    for i in range(position_first_gt_matrix, number_of_sonar_measurements):
        current_sonar_measurement = angle_and_intensities_raw[i, :]
        angle_sonar = current_sonar_measurement[1]
        angle_sonar_matrix_tmp = rotation_matrix(0, 0, 0) @ rotation_matrix(0, 0, angle_sonar)
        angle_sonar_matrix = np.eye(4)
        angle_sonar_matrix[0:3, 0:3] = angle_sonar_matrix_tmp
        range_val = current_sonar_measurement[0]
        position_robot = first_gt_matrix @ np.linalg.inv(first_gt_matrix) @ change_of_ekf_matrix @ position_estimate_ekf_raw[(i - 1) * 4:(i - 1) * 4 + 4, 0:4]

        rotation_matrix_90_degree = np.eye(4)
        rotation_matrix_90_degree[0:3, 0:3] = rotation_matrix(np.pi, 0, 0) @ rotation_matrix(0, 0, -np.pi / 2)
        position_robot_save_ekf[i, 0:4, 0:4] = rotation_matrix_90_degree @ position_robot

        if not np.isnan(position_robot[0, 0]):
            number_of_intensity_values = current_sonar_measurement.shape[0] - 2
            for j in range(10, number_of_intensity_values):
                position_of_intensity = position_robot @ angle_sonar_matrix @ np.array([j * range_val / number_of_intensity_values, 0, 0, 1])
                position_in_pixel = position_of_intensity / size_of_cell
                x_pos_in_map = round(position_in_pixel[0])
                y_pos_in_map = round(position_in_pixel[1])
                x_index_map = round(x_pos_in_map + number_of_pixels / 2)
                y_index_map = round(y_pos_in_map + number_of_pixels / 2)
                if 0 < x_index_map < number_of_pixels and 0 < y_index_map < number_of_pixels:
                    map_ekf[x_index_map, y_index_map] += current_sonar_measurement[j + 1]
                    map_index_ekf[x_index_map, y_index_map] += 1

    max_map = 0
    for i in range(number_of_pixels):
        for j in range(number_of_pixels):
            if map_index_ekf[i, j] > 0:
                map_ekf[i, j] /= map_index_ekf[i, j]
                if map_ekf[i, j] > max_map:
                    max_map = map_ekf[i, j]

    for i in range(number_of_pixels):
        for j in range(number_of_pixels):
            map_ekf[i, j] /= max_map

    fig, ax1 = plt.subplots()
    ax1.imshow(map_ekf, extent=[-size_of_map/2, size_of_map/2, -size_of_map/2, size_of_map/2], origin='lower')

    ax2 = fig.add_axes(ax1.get_position(), frameon=False)
    c = number_of_markers
    ax2.scatter(position_robot_save_ekf[:, 0, 3], position_robot_save_ekf[:, 1, 3], s=4, c=c, cmap='hot')
    ax2.set_visible(False)
    ax2.invert_yaxis()

    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)

    fig.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=ax1, orientation='vertical', position=[0.05, 0.11, 0.0675, 0.815])
    fig.colorbar(plt.cm.ScalarMappable(cmap='hot'), ax=ax2, orientation='vertical', position=[0.88, 0.11, 0.0675, 0.815])

    ax1.set_aspect('equal')
    ax1.set_title(f'Dataset: {name_of_the_titel_list}')

    name_of_pdf_file = f'/home/ws/matlab/registrationFourier/datasetReleaseJournal/PDFS/EKFMap{name_of_the_dataset}'
    plt.savefig(f'{name_of_pdf_file}.pdf')
    add_command_to_batchfile(name_of_batch_file, f'pdfcrop {name_of_pdf_file}.pdf {name_of_pdf_file}.pdf &', True)


# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

name_of_the_dataset = "tracking7m2step"

name_of_batch_file = "batchfile.sh"

name_of_the_dataset_list = [
    "15m4StepsLowSpeed",
    "15m4StepsFastSpeed",
    "7m1StepsSlowSpeed",
    "7m1StepsFastSpeed",
    "7m4StepsFastSpeed",
    "7m4StepsSlowSpeed",
    "15m1StepsSlowSpeed",
    "15m1StepsFastSpeed",
    "micron15m15m",
    "micron7m15m",
    "micron20m20m",
    "tracking15m2step",
    "tracking7m2step"
]

name_of_the_titel_list = [
    "3",
    "4",
    "5",
    "6",
    "8",
    "7",
    "1",
    "2",
    "9",
    "10",
    "11",
    "12",
    "13"
]

for k in range(len(name_of_the_dataset_list)):
    plot_pdfs(name_of_the_dataset_list[k], name_of_batch_file, name_of_the_titel_list[k])
