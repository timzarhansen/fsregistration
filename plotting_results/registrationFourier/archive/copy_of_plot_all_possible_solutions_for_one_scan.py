import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.signal import find_peaks
import subprocess


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

# classical success 2
which_keyframe = 24
name_of_folder = '/home/tim-external/dataFolder/gazeboCorrectedUnevenAnglesPCLs_4_100/'

first_scan = f'pclKeyFrame{which_keyframe}.pcd'
second_scan = f'pclKeyFrame{which_keyframe + 1}.pcd'

command = f'rosrun underwaterslam registrationOfTwoPCLs {name_of_folder}{first_scan} {name_of_folder}{second_scan}'
subprocess.run(command, shell=True)

plt.rcParams['text.usetex'] = True

magnitude_fftw1 = np.genfromtxt("csvFiles/magnitudeFFTW1.csv", delimiter=",")
N = int(np.sqrt(magnitude_fftw1.shape[0]))

phase_fftw1 = np.genfromtxt("csvFiles/phaseFFTW1.csv", delimiter=",")
voxel_data_used1 = np.genfromtxt("csvFiles/voxelDataFFTW1.csv", delimiter=",")

magnitude_fftw2 = np.genfromtxt("csvFiles/magnitudeFFTW2.csv", delimiter=",")
phase_fftw2 = np.genfromtxt("csvFiles/phaseFFTW2.csv", delimiter=",")
voxel_data_used2 = np.genfromtxt("csvFiles/voxelDataFFTW2.csv", delimiter=",")

magnitude1 = np.zeros((N, N))
phase1 = np.zeros((N, N))
voxel_data1 = np.zeros((N, N))
magnitude2 = np.zeros((N, N))
phase2 = np.zeros((N, N))
voxel_data2 = np.zeros((N, N))

for j in range(1, N + 1):
    for k in range(1, N + 1):
        magnitude1[j - 1, k - 1] = magnitude_fftw1[j * N - N + k - 1]
        phase1[j - 1, k - 1] = phase_fftw1[j * N - N + k - 1]
        voxel_data1[j - 1, k - 1] = voxel_data_used1[(j - 1) * N + k - 1]
        magnitude2[j - 1, k - 1] = magnitude_fftw2[j * N - N + k - 1]
        phase2[j - 1, k - 1] = phase_fftw2[j * N - N + k - 1]
        voxel_data2[j - 1, k - 1] = voxel_data_used2[(j - 1) * N + k - 1]

magnitude1 = np.fft.fftshift(magnitude1)
phase1 = np.fft.fftshift(phase1)
magnitude2 = np.fft.fftshift(magnitude2)
phase2 = np.fft.fftshift(phase2)

plt.figure(1)
plt.clf()
plt.imshow(magnitude1)
plt.axis('image')
plt.box(on=True)
plt.gca().set_box_aspect([1, 1, 1])
plt.savefig('outputPDFs/magnitudeScan1.pdf')
subprocess.run(['pdfcrop', 'outputPDFs/magnitudeScan1.pdf', 'outputPDFs/magnitudeScan1.pdf'])

plt.figure(2)
plt.clf()
plt.imshow(np.rot90(voxel_data1))
plt.axis('image')
plt.gca().set_box_aspect([1, 1, 1])
plt.savefig('outputPDFs/voxelData1.pdf')
subprocess.run(['pdfcrop', 'outputPDFs/voxelData1.pdf', 'outputPDFs/voxelData1.pdf'])

plt.figure(3)
plt.imshow(magnitude2)
plt.axis('image')
plt.gca().set_box_aspect([1, 1, 1])
plt.savefig('outputPDFs/magnitudeScan2.pdf')
subprocess.run(['pdfcrop', 'outputPDFs/magnitudeScan2.pdf', 'outputPDFs/magnitudeScan2.pdf'])

plt.figure(4)
plt.imshow(np.rot90(voxel_data2))
plt.axis('image')
plt.gca().set_box_aspect([1, 1, 1])
plt.savefig('outputPDFs/voxelData2.pdf')
subprocess.run(['pdfcrop', 'outputPDFs/voxelData2.pdf', 'outputPDFs/voxelData2.pdf'])

resampled_data_for_sphere1 = np.genfromtxt("csvFiles/resampledVoxel1.csv", delimiter=",")
resampled_data_for_sphere2 = np.genfromtxt("csvFiles/resampledVoxel2.csv", delimiter=",")

resampled_data_for_sphere_result1 = np.zeros((N, N))
resampled_data_for_sphere_result2 = np.zeros((N, N))
for i in range(1, N + 1):
    for j in range(1, N + 1):
        resampled_data_for_sphere_result1[i - 1, j - 1] = resampled_data_for_sphere1[(i - 1) * N + j - 1]
        resampled_data_for_sphere_result2[i - 1, j - 1] = resampled_data_for_sphere2[(i - 1) * N + j - 1]

if True:
    fig = plt.figure(5)
    ax = fig.add_subplot(111, projection='3d')
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(resampled_data_for_sphere_result2 / resampled_data_for_sphere_result2.max()), edgecolor='none')
    plt.axis('equal')
    plt.savefig('outputPDFs/resampledDataForSphereResult2.pdf')
    subprocess.run(['pdfcrop', 'outputPDFs/resampledDataForSphereResult2.pdf', 'outputPDFs/resampledDataForSphereResult2.pdf'])

    fig = plt.figure(6)
    ax = fig.add_subplot(111, projection='3d')
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(resampled_data_for_sphere_result1 / resampled_data_for_sphere_result1.max()), edgecolor='none')
    plt.axis('equal')
    plt.savefig('outputPDFs/resampledDataForSphereResult1.pdf')
    subprocess.run(['pdfcrop', 'outputPDFs/resampledDataForSphereResult1.pdf', 'outputPDFs/resampledDataForSphereResult1.pdf'])

plt.figure(7)
correlation_of_angles = np.genfromtxt("csvFiles/resultingCorrelation1D.csv", delimiter=",")
angles_x = np.linspace(0, 2 * np.pi, correlation_of_angles.shape[0])
psor, lsor = find_peaks(correlation_of_angles)
plt.plot(angles_x, correlation_of_angles)
for i in range(len(psor)):
    plt.text(angles_x[lsor[i]] + 0.15, psor[i] + 1, f'{round(angles_x[lsor[i]], 2)}')
plt.grid(True)
plt.xlim([-0.2, 2 * np.pi + 0.2])
plt.ylim([np.min(correlation_of_angles) - 10, np.max(correlation_of_angles) + 10])
plt.xlabel("angle in rad")
plt.ylabel("Correlation Value")
plt.box(on=True)
plt.gca().set_box_aspect([1, 1, 1])
plt.savefig('outputPDFs/resultingCorrelation1DAngle.pdf')
subprocess.run(['pdfcrop', 'outputPDFs/resultingCorrelation1DAngle.pdf', 'outputPDFs/resultingCorrelation1DAngle.pdf'])

data_information = np.genfromtxt("csvFiles/dataForReadIn.csv", delimiter=",")
number_of_solutions = int(data_information[0])
best_solution = int(data_information[1])

plt.figure(8)
plt.clf()
for i in range(1, number_of_solutions + 1):
    plt.subplot(2, 2, i)
    correlation_matrix_shift_1d = np.genfromtxt(f'csvFiles/resultingCorrelationShift{i - 1}.csv', delimiter=",")
    result_size = int(np.sqrt(len(correlation_matrix_shift_1d)))
    correlation_matrix_shift_2d = correlation_matrix_shift_1d.reshape(result_size, result_size)
    plt.imshow(correlation_matrix_shift_2d)
    plt.title(str(i))
    plt.box(on=True)
    plt.gca().set_box_aspect([1, 1, 1])

plt.savefig('outputPDFs/2dCorrelation.pdf')
subprocess.run(['pdfcrop', 'outputPDFs/2dCorrelation.pdf', 'outputPDFs/2dCorrelation.pdf'])

plt.figure(9)
plt.clf()

index_map = {1: 3, 2: 4, 3: 1, 4: 2}

for j in range(1, number_of_solutions + 1):
    i = index_map[j]
    plt.subplot(2, 2, j)

    point_cloud_result1 = o3d.io.read_point_cloud(f'csvFiles/resulting{i - 1}PCL1.pcd')
    point_cloud_result2 = o3d.io.read_point_cloud(f'csvFiles/resulting{i - 1}PCL2.pcd')

    tform = np.eye(4)
    tform[0:3, 0:3] = rotation_matrix(0, 0, -0.1)
    tform[0, 3] = 0
    tform[1, 3] = 0

    points1 = np.asarray(point_cloud_result1.points)
    points1_transformed = (tform @ np.hstack((points1, np.ones((points1.shape[0], 1)))).T).T[:, 0:3]
    point_cloud_result1.points = o3d.utility.Vector3dVector(points1_transformed)

    points2 = np.asarray(point_cloud_result2.points)
    points2_transformed = (tform @ np.hstack((points2, np.ones((points2.shape[0], 1)))).T).T[:, 0:3]
    point_cloud_result2.points = o3d.utility.Vector3dVector(points2_transformed)

    fig = plt.gcf()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points1_transformed[:, 0], points1_transformed[:, 1], points1_transformed[:, 2], c='r', marker='.', s=1)
    ax.scatter(points2_transformed[:, 0], points2_transformed[:, 1], points2_transformed[:, 2], c='b', marker='.', s=1)
    ax.view_init(90, -90)
    ax.set_facecolor('w')
    plt.title(str(j), color='k')

plt.savefig('outputPDFs/matchedPointcloudsAll.pdf')
subprocess.run(['pdfcrop', 'outputPDFs/matchedPointcloudsAll.pdf', 'outputPDFs/matchedPointcloudsAll.pdf'])

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

for i in range(1, number_of_solutions + 1):
    resulting_transformation_of_scan = np.genfromtxt(f'csvFiles/resultingTransformation{i - 1}.csv', delimiter=",")
    resulting_gt_transformation = test_transformation @ np.linalg.inv(gt_transformation_diff)
    resulting_yaw_diff = np.arctan2(np.sin(rpy_tmp[2] - resulting_transformation_of_scan[5]), np.cos(rpy_tmp[2] - resulting_transformation_of_scan[5]))
    translation_diff = resulting_transformation_of_scan[0:2] - resulting_gt_transformation[0:2, 3]

if True:
    resulting_yaw_diff_array = np.zeros(number_of_solutions)
    for i in range(1, number_of_solutions + 1):
        resulting_transformation_of_scan = np.genfromtxt(f'csvFiles/resultingTransformation{i - 1}.csv', delimiter=",")
        resulting_yaw_diff_array[i - 1] = abs(np.arctan2(np.sin(rpy_tmp[2] - resulting_transformation_of_scan[5]), np.cos(rpy_tmp[2] - resulting_transformation_of_scan[5])))

    minimum_initial_guess, index_initial_guess = np.min(resulting_yaw_diff_array), np.argmin(resulting_yaw_diff_array)

    fig = plt.figure(10)
    ax = fig.add_subplot(111, projection='3d')

    point_cloud_result1 = o3d.io.read_point_cloud(f'csvFiles/resulting{index_initial_guess}PCL1.pcd')
    point_cloud_result2 = o3d.io.read_point_cloud(f'csvFiles/resulting{index_initial_guess}PCL2.pcd')

    tform = np.eye(4)
    tform[0:3, 0:3] = rotation_matrix(0, 0, 1.6)
    tform[0, 3] = 0
    tform[1, 3] = 0

    points1 = np.asarray(point_cloud_result1.points)
    points1_transformed = (tform @ np.hstack((points1, np.ones((points1.shape[0], 1)))).T).T[:, 0:3]

    points2 = np.asarray(point_cloud_result2.points)
    points2_transformed = (tform @ np.hstack((points2, np.ones((points2.shape[0], 1)))).T).T[:, 0:3]

    ax.scatter(points1_transformed[:, 0], points1_transformed[:, 1], points1_transformed[:, 2], c='r', marker='.', s=1)
    ax.scatter(points2_transformed[:, 0], points2_transformed[:, 1], points2_transformed[:, 2], c='b', marker='.', s=1)
    ax.view_init(90, -90)
    ax.set_facecolor('w')
    ax.set_xlabel('Y-Axis in m')
    ax.set_ylabel('X-Axis in m')
    plt.gca().set_box_aspect([1, 1, 1])
    plt.savefig('outputPDFs/matchedPointclouds.pdf')
    subprocess.run(['pdfcrop', 'outputPDFs/matchedPointclouds.pdf', 'outputPDFs/matchedPointclouds.pdf'])

    plt.figure(11)
    plt.clf()
    correlation_matrix_shift_1d = np.genfromtxt(f'csvFiles/resultingCorrelationShift{index_initial_guess}.csv', delimiter=",")
    result_size = int(np.sqrt(len(correlation_matrix_shift_1d)))
    correlation_matrix_shift_2d = correlation_matrix_shift_1d.reshape(result_size, result_size)
    plt.imshow(correlation_matrix_shift_2d)
    plt.box(on=True)
    plt.gca().set_box_aspect([1, 1, 1])
    plt.savefig('outputPDFs/2dCorrelation.pdf')
    subprocess.run(['pdfcrop', 'outputPDFs/2dCorrelation.pdf', 'outputPDFs/2dCorrelation.pdf'])
