import numpy as np


def angle_diff(a, b):
    y = np.mod(b - a, 2 * np.pi)
    if y < -np.pi:
        y = y + 2 * np.pi
    elif y > np.pi:
        y = y - 2 * np.pi
    return y


def rotation_matrix(roll, pitch, yaw):
    rotation = np.array([
        [np.cos(yaw) * np.cos(pitch), np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll), np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)],
        [np.sin(yaw) * np.cos(pitch), np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll), np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)],
        [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)]
    ])
    return rotation


import matplotlib.pyplot as plt
from scipy.optimize import minimize


def calc_error(x, y, alpha, gt_matrix, slam_matrix):
    matrix_to_be_optimized = np.eye(4)
    matrix_to_be_optimized[0, 3] = x
    matrix_to_be_optimized[1, 3] = y
    matrix_to_be_optimized[0, 0] = np.cos(alpha)
    matrix_to_be_optimized[1, 1] = np.cos(alpha)
    matrix_to_be_optimized[0, 1] = -np.sin(alpha)
    matrix_to_be_optimized[1, 0] = np.sin(alpha)
    current_size = gt_matrix.shape[0]
    norm_tmp = np.zeros((current_size, 1))
    for i in range(current_size):
        tmp_matrix1 = matrix_to_be_optimized @ gt_matrix[i, :, :]
        tmp_matrix2 = slam_matrix[i, :, :]
        norm_tmp[i, 0] = np.linalg.norm([tmp_matrix1[0, 3] - tmp_matrix2[0, 3], tmp_matrix1[1, 3] - tmp_matrix2[1, 3]])
    norm_result = np.mean(np.abs(norm_tmp))
    return norm_result


def calc_std(x, y, alpha, gt_matrix, slam_matrix):
    matrix_to_be_optimized = np.eye(4)
    matrix_to_be_optimized[0, 3] = x
    matrix_to_be_optimized[1, 3] = y
    matrix_to_be_optimized[0, 0] = np.cos(alpha)
    matrix_to_be_optimized[1, 1] = np.cos(alpha)
    matrix_to_be_optimized[0, 1] = -np.sin(alpha)
    matrix_to_be_optimized[1, 0] = np.sin(alpha)
    current_size = gt_matrix.shape[0]
    norm_tmp = np.zeros((current_size, 1))
    for i in range(current_size):
        tmp_matrix1 = matrix_to_be_optimized @ gt_matrix[i, :, :]
        tmp_matrix2 = slam_matrix[i, :, :]
        norm_tmp[i, 0] = np.linalg.norm([tmp_matrix1[0, 3] - tmp_matrix2[0, 3], tmp_matrix1[1, 3] - tmp_matrix2[1, 3]])
    norm_result = np.std(np.abs(norm_tmp))
    return norm_result


name_of_method_list = ["_classical_slam_"]
name_of_dataset_list = ["_simulation"]

for name_of_method in name_of_method_list:
    for name_of_dataset in name_of_dataset_list:
        input_position = np.genfromtxt("csvFiles/IROSResults/positionEstimationOverTime" + name_of_dataset + name_of_method + ".csv", delimiter=",")
        input_position = input_position[:-4*70, :4]
        input_ground_truth = np.genfromtxt("csvFiles/IROSResults/groundTruthOverTime" + name_of_dataset + name_of_method + ".csv", delimiter=",")
        input_ground_truth = input_ground_truth[:-4*70, :4]

        how_many_skip = 800
        input_position = input_position[4*how_many_skip:, :]
        input_ground_truth = input_ground_truth[4*how_many_skip:, :]

        first_matrix_gt = input_ground_truth[0:4, 0:4]
        first_matrix_slam = input_position[0:4, 0:4]

        number_of_vertices = input_ground_truth.shape[0] // 4
        gt_matrix = np.zeros((number_of_vertices, 4, 4))
        slam_matrix = np.zeros((number_of_vertices, 4, 4))

        for i in range(number_of_vertices):
            gt_matrix[i, :, :] = np.linalg.solve(first_matrix_gt, input_ground_truth[i*4:i*4+4, 0:4])
            slam_matrix[i, :, :] = np.linalg.solve(first_matrix_slam, input_position[i*4:i*4+4, 0:4])
            if name_of_dataset == "_circle" or name_of_dataset == "_s_curve" or name_of_dataset == "_squares":
                gt_matrix[i, 0, 3] = -gt_matrix[i, 0, 3]

            x_pos_slam = np.zeros(number_of_vertices)
            y_pos_slam = np.zeros(number_of_vertices)
            angle_slam = np.zeros(number_of_vertices)
            x_pos_gt = np.zeros(number_of_vertices)
            y_pos_gt = np.zeros(number_of_vertices)
            angle_gt = np.zeros(number_of_vertices)

            x_pos_slam[i] = slam_matrix[i, 0, 3]
            y_pos_slam[i] = slam_matrix[i, 1, 3]
            angle_slam[i] = np.arctan2(slam_matrix[i, 1, 0], slam_matrix[i, 0, 0])

            x_pos_gt[i] = gt_matrix[i, 0, 3]
            y_pos_gt[i] = gt_matrix[i, 1, 3]
            angle_gt[i] = np.arctan2(gt_matrix[i, 1, 0], gt_matrix[i, 0, 0])

        optimize_parameters = [0, 0, 0]
        def objective(v):
            return calc_error(v[0], v[1], v[2], gt_matrix, slam_matrix)

        result = minimize(objective, optimize_parameters, method='Nelder-Mead')
        Z = result.x

        print(calc_error(Z[0], Z[1], Z[2], gt_matrix, slam_matrix))
        print(calc_error(0, 0, Z[2], gt_matrix, slam_matrix))

        matrix_for_relative_error = np.eye(4)
        matrix_for_relative_error[0, 3] = Z[0]
        matrix_for_relative_error[1, 3] = Z[1]
        matrix_for_relative_error[0, 0] = np.cos(Z[2])
        matrix_for_relative_error[1, 1] = np.cos(Z[2])
        matrix_for_relative_error[0, 1] = -np.sin(Z[2])
        matrix_for_relative_error[1, 0] = np.sin(Z[2])

        for i in range(number_of_vertices):
            tmp_matrix = matrix_for_relative_error @ gt_matrix[i, :, :]
            x_pos_slam[i] = slam_matrix[i, 0, 3]
            y_pos_slam[i] = slam_matrix[i, 1, 3]
            x_pos_gt[i] = tmp_matrix[0, 3]
            y_pos_gt[i] = tmp_matrix[1, 3]

        k = 1
        window_size = 400
        relative_angle_error = np.zeros(number_of_vertices - window_size)
        for i in range(window_size//2, number_of_vertices - window_size//2):
            correction_i = window_size//2 - 1
            x_pos_slam1 = slam_matrix[i - correction_i, 0, 3]
            y_pos_slam1 = slam_matrix[i - correction_i, 1, 3]
            x_pos_slam2 = slam_matrix[i + correction_i, 0, 3]
            y_pos_slam2 = slam_matrix[i + correction_i, 1, 3]

            tmp_matrix1 = matrix_for_relative_error @ gt_matrix[i - correction_i, :, :]
            tmp_matrix2 = matrix_for_relative_error @ gt_matrix[i + correction_i, :, :]

            x_pos_gt1 = tmp_matrix1[0, 3]
            y_pos_gt1 = tmp_matrix1[1, 3]
            x_pos_gt2 = tmp_matrix2[0, 3]
            y_pos_gt2 = tmp_matrix2[1, 3]

            error_angle1 = np.arctan2(y_pos_slam1 - y_pos_slam2, x_pos_slam1 - x_pos_slam2)
            error_angle2 = np.arctan2(y_pos_gt1 - y_pos_gt2, x_pos_gt1 - x_pos_gt2)
            relative_angle_error[k - 1] = angle_diff(error_angle1, error_angle2)
            k += 1

        matrix_for_absolute_error = np.eye(4)
        matrix_for_absolute_error[0, 3] = 0
        matrix_for_absolute_error[1, 3] = 0
        matrix_for_absolute_error[0, 0] = np.cos(Z[2])
        matrix_for_absolute_error[1, 1] = np.cos(Z[2])
        matrix_for_absolute_error[0, 1] = -np.sin(Z[2])
        matrix_for_absolute_error[1, 0] = np.sin(Z[2])
        k = 1
        absolute_angle_error = np.zeros(number_of_vertices - window_size)
        for i in range(window_size//2, number_of_vertices - window_size//2):
            correction_i = window_size//2 - 1
            x_pos_slam1 = slam_matrix[i - correction_i, 0, 3]
            y_pos_slam1 = slam_matrix[i - correction_i, 1, 3]
            x_pos_slam2 = slam_matrix[i + correction_i, 0, 3]
            y_pos_slam2 = slam_matrix[i + correction_i, 1, 3]

            tmp_matrix1 = matrix_for_absolute_error @ gt_matrix[i - correction_i, :, :]
            tmp_matrix2 = matrix_for_absolute_error @ gt_matrix[i + correction_i, :, :]

            x_pos_gt1 = tmp_matrix1[0, 3]
            y_pos_gt1 = tmp_matrix1[1, 3]
            x_pos_gt2 = tmp_matrix2[0, 3]
            y_pos_gt2 = tmp_matrix2[1, 3]

            error_angle1 = np.arctan2(y_pos_slam1 - y_pos_slam2, x_pos_slam1 - x_pos_slam2)
            error_angle2 = np.arctan2(y_pos_gt1 - y_pos_gt2, x_pos_gt1 - x_pos_gt2)
            absolute_angle_error[k - 1] = angle_diff(error_angle1, error_angle2)
            k += 1

        plt.figure(3)
        plt.clf()
        magnitude_map_input = np.genfromtxt("csvFiles/IROSResults/currentMap" + name_of_dataset + name_of_method + ".csv", delimiter=",")
        n = int(np.sqrt(magnitude_map_input.shape[0]))
        magnitude_map = np.zeros((n, n))
        for j in range(n):
            for i in range(n):
                magnitude_map[i, j] = magnitude_map_input[(i - 1) * n + j]

        rotation_angle_image = 0
        rotation_gt_system = 0

        size_pixel = 10 / 512
        pos_x_zero = 95
        pos_y_zero = 85
        start_number = 100
        end_number = 440
        image_shift_x = 0
        image_shift_y = 0

        if name_of_dataset == "_valentinKeller":
            size_pixel = 45 / 512

        if name_of_dataset == "_simulation":
            size_pixel = 80 / 512
            pos_x_zero = 190
            pos_y_zero = 398
            start_number = 1
            end_number = 420
            rotation_angle_image = 3

            if name_of_method == "_dynamic_slam_4_0_":
                image_shift_x = 30
                image_shift_y = 3

            if name_of_method == "_dead_reckoning_":
                pos_y_zero = 415

        if name_of_dataset == "_circle":
            image_shift_x = 30
            image_shift_y = 3
            start_number = 140
            end_number = 430
            pos_x_zero = 75
            pos_y_zero = 58

            if name_of_method == "_dead_reckoning_":
                image_shift_x = 20
                image_shift_y = 35
                pos_x_zero = 50
                pos_y_zero = 58

            if name_of_method == "_classical_slam_":
                pos_x_zero = 70
                pos_y_zero = 36

            if name_of_method == "_dynamic_slam_4_0_":
                pos_x_zero = 72
                pos_y_zero = 37

        if name_of_dataset == "_s_curve":
            image_shift_x = 30
            image_shift_y = -45
            start_number = 90
            end_number = 420
            pos_x_zero = 88
            pos_y_zero = 60

            if name_of_method == "_dead_reckoning_":
                rotation_gt_system = 1.12

        if name_of_dataset == "_squares":
            start_number = 110
            end_number = 440
            image_shift_x = 20
            image_shift_y = 20

            if name_of_method == "_dynamic_slam_4_0_":
                pos_x_zero = 100
                pos_y_zero = 98

            if name_of_method == "_dead_reckoning_":
                rotation_gt_system = 0.775

        test_image = magnitude_map
        x_pos_gt = first_matrix_gt[0, 3]
        y_pos_gt = first_matrix_gt[1, 3]

        x_des_mid_point = 1.8
        y_des_mid_point = 1.8
        diff_x = x_pos_gt - x_des_mid_point
        diff_y = y_pos_gt - y_des_mid_point
        diff_x = diff_x / size_pixel
        diff_y = diff_y / size_pixel

        from scipy.ndimage import rotate
        test_image = rotate(test_image, rotation_angle_image, reshape=False)
        test_image = np.roll(test_image, round(diff_x) + image_shift_x, axis=1)
        test_image = np.roll(test_image, round(diff_y) + image_shift_y, axis=0)
        test_image = test_image[start_number:end_number, int(start_number*1.1):int(end_number*0.9)]

        quater_degree_matrix = np.eye(4)
        quater_degree_matrix[0:3, 0:3] = rotation_matrix(0, 0, rotation_gt_system) @ rotation_matrix(0, np.pi, 0) @ rotation_matrix(0, 0, np.pi/2)

        picture_gt_plot_x = np.zeros(number_of_vertices)
        picture_gt_plot_y = np.zeros(number_of_vertices)
        picture_slam_plot_x = np.zeros(number_of_vertices)
        picture_slam_plot_y = np.zeros(number_of_vertices)

        for i in range(number_of_vertices):
            tmp_matrix_gt = quater_degree_matrix @ matrix_for_relative_error @ gt_matrix[i, :, :]
            tmp_matrix_slam = quater_degree_matrix @ slam_matrix[i, :, :]

            picture_gt_plot_x[i] = tmp_matrix_gt[0, 3]
            picture_gt_plot_y[i] = tmp_matrix_gt[1, 3]
            picture_slam_plot_x[i] = tmp_matrix_slam[0, 3]
            picture_slam_plot_y[i] = tmp_matrix_slam[1, 3]

            picture_slam_plot_x[i] = picture_slam_plot_x[i] / size_pixel + pos_x_zero + x_pos_gt / size_pixel
            picture_slam_plot_y[i] = picture_slam_plot_y[i] / size_pixel + pos_y_zero + y_pos_gt / size_pixel

            picture_gt_plot_x[i] = picture_gt_plot_x[i] / size_pixel + pos_x_zero + x_pos_gt / size_pixel
            picture_gt_plot_y[i] = picture_gt_plot_y[i] / size_pixel + pos_y_zero + y_pos_gt / size_pixel

        remove_last_rows = 300
        picture_slam_plot_x = picture_slam_plot_x[:-remove_last_rows]
        picture_slam_plot_y = picture_slam_plot_y[:-remove_last_rows]
        picture_gt_plot_x = picture_gt_plot_x[:-remove_last_rows]
        picture_gt_plot_y = picture_gt_plot_y[:-remove_last_rows]

        plt.clf()
        plt.hold(True)
        plt.imshow(test_image)
        plt.plot(picture_gt_plot_x, picture_gt_plot_y, color='r')
        plt.plot(picture_slam_plot_x, picture_slam_plot_y, color='g')
        plt.axis('equal')
        plt.axis('image')

        plt.xlim([1, test_image.shape[0]])
        plt.ylim([1, test_image.shape[1]])

        if name_of_dataset == "_circle":
            plt.xlim([1, test_image.shape[1]])
            plt.ylim([1, test_image.shape[0]])
        if name_of_dataset == "_squares":
            plt.xlim([1, test_image.shape[1]])
            plt.ylim([1, test_image.shape[0]])
        if name_of_dataset == "_s_curve":
            plt.xlim([1, test_image.shape[1]])
            plt.ylim([1, test_image.shape[0]])
        if name_of_dataset == "_simulation":
            plt.xlim([1, test_image.shape[1]])
            plt.ylim([1, test_image.shape[0]])

        if name_of_dataset == "_simulation":
            ax = plt.gca()
            tmp = ax.get_xticks()
            ax.set_xticklabels([f"{t/5}" for t in tmp])
            tmp = ax.get_yticks()
            ax.set_yticklabels([f"{t/5}" for t in tmp])
        else:
            ax = plt.gca()
            tmp = ax.get_xticks()
            ax.set_xticklabels([f"{t/50}" for t in tmp])
            tmp = ax.get_yticks()
            ax.set_yticklabels([f"{t/50}" for t in tmp])

        plt.ylabel("m")
        plt.xlabel("m")

        name_of_pdf_file = '/home/ws/matlab/registrationFourier/csvFiles/IROSResults/images/' + name_of_dataset + name_of_method
        plt.savefig(name_of_pdf_file + '.pdf')

        import subprocess
        subprocess.run(['pdfcrop', name_of_pdf_file + '.pdf', name_of_pdf_file + '.pdf'])

        fid = open('errorValues.txt', 'a+')
        fid.write(name_of_dataset + name_of_method + '\n')
        fid.write(f"Absolute mean angle error: {np.mean(np.abs(absolute_angle_error))}\n")
        fid.write(f"Absolute std angle error: {np.std(np.abs(absolute_angle_error))}\n")
        fid.write(f"Absolute mean l2 error: {calc_error(0, 0, Z[2], gt_matrix, slam_matrix)}\n")
        fid.write(f"Absolute std l2 error: {calc_std(0, 0, Z[2], gt_matrix, slam_matrix)}\n")
        fid.write(f"Relative mean angle error: {np.mean(np.abs(relative_angle_error))}\n")
        fid.write(f"Relative std angle error: {np.std(np.abs(relative_angle_error))}\n")
        fid.write(f"Relative mean l2 error: {calc_error(Z[0], Z[1], Z[2], gt_matrix, slam_matrix)}\n")
        fid.write(f"Relative std l2 error: {calc_std(Z[0], Z[1], Z[2], gt_matrix, slam_matrix)}\n")
        fid.write(f"Angle mean error: {np.mean(np.abs(angle_slam))}\n")
        fid.write(f"Angle std error: {np.std(np.abs(angle_slam))}\n")
        fid.close()
