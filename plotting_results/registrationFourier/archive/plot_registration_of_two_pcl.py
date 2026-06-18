import numpy as np
import matplotlib.pyplot as plt
import subprocess
import open3d as o3d
from scipy.signal import find_peaks


def rotation_matrix(roll, pitch, yaw):
    rotation = np.array([
        [np.cos(yaw) * np.cos(pitch), np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll), np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)],
        [np.sin(yaw) * np.cos(pitch), np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll), np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)],
        [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)]
    ])
    return rotation


def angles_r(R, str_input):
    theta = np.zeros(3)

    By = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
    Ry90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    C = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    signy = 1

    if str_input[1] == 'x':
        R = C @ R @ np.linalg.inv(C)
    elif str_input[1] == 'z':
        R = np.linalg.inv(C) @ R @ C

    if all(str_input == 'xzy') or all(str_input == 'yxz') or all(str_input == 'zyx'):
        R = By @ R @ np.linalg.inv(By)
        signy = -1

    if all(str_input == 'xzx') or all(str_input == 'yxy') or all(str_input == 'zyz'):
        R = Ry90 @ R @ np.linalg.inv(Ry90)

    if str_input[0] != str_input[2]:
        theta[1] = signy * np.arcsin(R[0, 2]) * 180 / np.pi
        theta[0] = np.arctan2(-R[1, 2], R[2, 2]) * 180 / np.pi
        theta[2] = np.arctan2(-R[0, 1], R[0, 0]) * 180 / np.pi
    else:
        theta[1] = np.arccos(R[0, 0]) * 180 / np.pi
        theta[0] = np.arctan2(R[1, 0], -R[2, 0]) * 180 / np.pi
        theta[2] = np.arctan2(R[0, 1], R[0, 2]) * 180 / np.pi

    return theta


# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

from_idx = 0
to_idx = 257
resulting_yaw_diff_best_matching = np.zeros((to_idx - from_idx + 1, 1))
translation_diff_best_matching = np.zeros((to_idx - from_idx + 1, 2))
resulting_yaw_diff_initial_guess = np.zeros((to_idx - from_idx + 1, 1))
translation_diff_initial_guess = np.zeros((to_idx - from_idx + 1, 2))

for index_current_frame in range(from_idx, to_idx + 1):
    which_keyframe = index_current_frame
    name_of_current_testing_set = 'gazeboCorrectedEvenAnglesPCLs_2_75'
    name_of_folder = f'/home/tim-external/dataFolder/{name_of_current_testing_set}/'
    first_scan = f'pclKeyFrame{which_keyframe}.pcd'
    second_scan = f'pclKeyFrame{which_keyframe + 1}.pcd'

    command = f'rosrun underwaterslam registrationOfTwoPCLs {name_of_folder}{first_scan} {name_of_folder}{second_scan}'
    subprocess.call(command, shell=True)

    magnitude_fftw1 = np.genfromtxt("csvFiles/magnitudeFFTW1.csv", delimiter=",")
    n = int(np.sqrt(magnitude_fftw1.shape[0]))

    phase_fftw1 = np.genfromtxt("csvFiles/phaseFFTW1.csv", delimiter=",")
    voxel_data_used1 = np.genfromtxt("csvFiles/voxelDataFFTW1.csv", delimiter=",")

    magnitude_fftw2 = np.genfromtxt("csvFiles/magnitudeFFTW2.csv", delimiter=",")
    phase_fftw2 = np.genfromtxt("csvFiles/phaseFFTW2.csv", delimiter=",")
    voxel_data_used2 = np.genfromtxt("csvFiles/voxelDataFFTW2.csv", delimiter=",")

    magnitude1 = np.zeros((n, n))
    phase1 = np.zeros((n, n))
    voxel_data1 = np.zeros((n, n))
    magnitude2 = np.zeros((n, n))
    phase2 = np.zeros((n, n))
    voxel_data2 = np.zeros((n, n))

    for j in range(1, n + 1):
        for k in range(1, n + 1):
            magnitude1[j - 1, k - 1] = magnitude_fftw1[j * n - n + k - 1]
            phase1[j - 1, k - 1] = phase_fftw1[j * n - n + k - 1]
            voxel_data1[j - 1, k - 1] = voxel_data_used1[(j - 1) * n + k - 1]
            magnitude2[j - 1, k - 1] = magnitude_fftw2[j * n - n + k - 1]
            phase2[j - 1, k - 1] = phase_fftw2[j * n - n + k - 1]
            voxel_data2[j - 1, k - 1] = voxel_data_used2[(j - 1) * n + k - 1]

    magnitude1 = np.fft.fftshift(magnitude1)
    phase1 = np.fft.fftshift(phase1)
    magnitude2 = np.fft.fftshift(magnitude2)
    phase2 = np.fft.fftshift(phase2)

    plt.figure(1)
    plt.clf()
    plt.imshow(magnitude1)
    plt.axis('image')
    plt.box(True)

    plt.figure(2)
    plt.clf()
    plt.imshow(voxel_data1)
    plt.axis('image')

    plt.figure(3)
    plt.imshow(magnitude2)
    plt.axis('image')

    plt.figure(4)
    plt.imshow(voxel_data2)
    plt.axis('image')

    resampled_data_for_sphere1 = np.genfromtxt("csvFiles/resampledVoxel1.csv", delimiter=",")
    resampled_data_for_sphere2 = np.genfromtxt("csvFiles/resampledVoxel2.csv", delimiter=",")

    resampled_data_for_sphere_result1 = np.zeros((n, n))
    resampled_data_for_sphere_result2 = np.zeros((n, n))
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            resampled_data_for_sphere_result1[i - 1, j - 1] = resampled_data_for_sphere1[(i - 1) * n + j - 1]
            resampled_data_for_sphere_result2[i - 1, j - 1] = resampled_data_for_sphere2[(i - 1) * n + j - 1]

    plt.figure(5)
    plt.clf()
    plt.imshow(resampled_data_for_sphere_result2)
    plt.axis('image')

    plt.figure(6)
    plt.clf()
    plt.imshow(resampled_data_for_sphere_result1)
    plt.axis('image')

    plt.figure(7)
    correlation_of_angles = np.genfromtxt("csvFiles/resultingCorrelation1D.csv", delimiter=",")

    angles_x = np.linspace(0, 2 * np.pi, len(correlation_of_angles))
    psor, lsor = find_peaks(correlation_of_angles)
    plt.plot(angles_x, correlation_of_angles)
    for i in range(len(psor)):
        plt.text(angles_x[lsor[i]] + 0.15, psor[i] + 1, f'{round(angles_x[lsor[i]], 2)}')
    plt.grid(True)
    plt.xlim([-0.2, 2 * np.pi + 0.2])
    plt.ylim([np.min(correlation_of_angles) - 10, np.max(correlation_of_angles) + 10])
    plt.xlabel("angle in rad")
    plt.ylabel("Correlation Value")

    data_information = np.genfromtxt("csvFiles/dataForReadIn.csv", delimiter=",")

    number_of_solutions = int(data_information[0])
    best_solution = int(data_information[1])

    correlation_matrix_shift_1d = np.genfromtxt(f'csvFiles/resultingCorrelationShift{best_solution}.csv', delimiter=",")
    result_size = int(np.sqrt(len(correlation_matrix_shift_1d)))
    correlation_matrix_shift_2d = correlation_matrix_shift_1d.reshape(result_size, result_size)

    plt.figure(8)
    plt.clf()
    plt.imshow(correlation_matrix_shift_2d)
    plt.box(True)

    plt.figure(9)
    plt.clf()

    for i in range(1, number_of_solutions + 1):
        plt.subplot(2, 2, i)
        point_cloud_result1 = o3d.io.read_point_cloud(f'csvFiles/resulting{i - 1}PCL1.pcd')
        point_cloud_result2 = o3d.io.read_point_cloud(f'csvFiles/resulting{i - 1}PCL2.pcd')

        points1 = np.array(point_cloud_result1.points)
        points2 = np.array(point_cloud_result2.points)

        plt.scatter(points1[:, 0], points1[:, 1], c='blue', marker='.', s=1)
        plt.scatter(points2[:, 0], points2[:, 1], c='red', marker='.', s=1)
        plt.axis('equal')
        plt.grid(True)

    transformation1 = np.genfromtxt(f'{name_of_folder}position{which_keyframe}.csv', delimiter=",")
    transformation2 = np.genfromtxt(f'{name_of_folder}position{which_keyframe + 1}.csv', delimiter=",")
    translation1 = transformation1[0:3]
    translation2 = transformation2[0:3]
    rotation1 = transformation1[3:].reshape(3, 3)
    rotation2 = transformation2[3:].reshape(3, 3)
    resulting_translation_diff = -transformation1[0:3]
    resulting_rotation_diff = transformation2[3:6] - transformation1[3:6]

    complete_transformation1 = np.eye(4)
    complete_transformation2 = np.eye(4)

    complete_transformation1[0:3, 0:3] = rotation1
    complete_transformation2[0:3, 0:3] = rotation2
    complete_transformation1[0, 3] = transformation1[0]
    complete_transformation1[1, 3] = transformation1[1]
    complete_transformation2[0, 3] = transformation2[0]
    complete_transformation2[1, 3] = transformation2[1]

    gt_transformation_diff = np.linalg.inv(complete_transformation1) @ complete_transformation2
    rpy_tmp = angles_r(gt_transformation_diff[0:3, 0:3], 'xyz') / 180 * np.pi

    test_transformation = np.eye(4)
    test_transformation[0:3, 0:3] = rotation_matrix(np.pi, 0, 0)
    resulting_gt_transformation = test_transformation @ np.linalg.inv(gt_transformation_diff)

    resulting_yaw_diff = np.zeros(number_of_solutions)
    for i in range(number_of_solutions):
        resulting_transformation_of_scan = np.genfromtxt(f'csvFiles/resultingTransformation{i}.csv', delimiter=",")
        resulting_yaw_diff[i] = np.abs(np.arctan2(np.sin(rpy_tmp[2] - resulting_transformation_of_scan[5]), np.cos(rpy_tmp[2] - resulting_transformation_of_scan[5])))

    minimum_initial_guess, index_initial_guess = np.min(resulting_yaw_diff), np.argmin(resulting_yaw_diff)

    resulting_transformation_of_scan = np.genfromtxt(f'csvFiles/resultingTransformation{index_initial_guess}.csv', delimiter=",")
    resulting_yaw_diff_initial_guess[index_current_frame - from_idx] = np.arctan2(np.sin(rpy_tmp[2] - resulting_transformation_of_scan[5]), np.cos(rpy_tmp[2] - resulting_transformation_of_scan[5]))

    translation_diff_initial_guess[index_current_frame - from_idx, :] = resulting_transformation_of_scan[0:2] - resulting_gt_transformation[0:2, 3]

    resulting_transformation_of_scan = np.genfromtxt(f'csvFiles/resultingTransformation{best_solution}.csv', delimiter=",")
    resulting_yaw_diff_best_matching[index_current_frame - from_idx] = np.arctan2(np.sin(rpy_tmp[2] - resulting_transformation_of_scan[5]), np.cos(rpy_tmp[2] - resulting_transformation_of_scan[5]))
    translation_diff_best_matching[index_current_frame - from_idx, :] = resulting_transformation_of_scan[0:2] - resulting_gt_transformation[0:2, 3]

    np.savez(f'resultsOfManyMatching/{name_of_current_testing_set}64.npz',
             resulting_yaw_diff_initial_guess=resulting_yaw_diff_initial_guess,
             translation_diff_initial_guess=translation_diff_initial_guess,
             resulting_yaw_diff_best_matching=resulting_yaw_diff_best_matching,
             translation_diff_best_matching=translation_diff_best_matching)
