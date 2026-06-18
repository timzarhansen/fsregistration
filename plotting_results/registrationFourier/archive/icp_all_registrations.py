import numpy as np
import matplotlib.pyplot as plt
import subprocess
import open3d as o3d


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


def rotation_matrix(roll, pitch, yaw):
    rotation = np.array([
        [np.cos(yaw) * np.cos(pitch), np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll), np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)],
        [np.sin(yaw) * np.cos(pitch), np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll), np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)],
        [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)]
    ])
    return rotation


# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

from_idx = 0
to_idx = 257
resulting_yaw_diff = np.zeros((to_idx - from_idx + 1, 1))
translation_diff = np.zeros((to_idx - from_idx + 1, 2))

for index_current_frame in range(from_idx, to_idx + 1):
    which_keyframe = index_current_frame
    name_of_current_testing_set = 'gazeboCorrectedEvenAnglesPCLs_2_75'
    name_of_folder = f'/home/tim-external/dataFolder/{name_of_current_testing_set}/'
    first_scan = f'pclKeyFrame{which_keyframe}.pcd'
    second_scan = f'pclKeyFrame{which_keyframe + 1}.pcd'

    transformation1 = np.genfromtxt(f'{name_of_folder}position{which_keyframe}.csv', delimiter=',')
    transformation2 = np.genfromtxt(f'{name_of_folder}position{which_keyframe + 1}.csv', delimiter=',')
    translation1 = transformation1[0:3]
    translation2 = transformation2[0:3]
    rotation1 = transformation1[3:].reshape((3, 3))
    rotation2 = transformation2[3:].reshape((3, 3))
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

    test_transformation = np.eye(4)
    test_transformation[0:3, 0:3] = rotation_matrix(np.pi, 0, 0)
    resulting_gt_transformation = test_transformation @ np.linalg.inv(gt_transformation_diff)

    rpy_gt = angles_r(resulting_gt_transformation[0:3, 0:3], 'xyz') / 180 * np.pi
    yaw_initial_guess = rpy_gt[2]

    command = f'rosrun underwaterslam registrationOfTwoPCLICP {name_of_folder}{first_scan} {name_of_folder}{second_scan} {yaw_initial_guess}'
    subprocess.run(command, shell=True)

    plt.figure(9)
    plt.clf()

    point_cloud_result1 = o3d.io.read_point_cloud('csvFiles/resulting0PCL1.pcd')
    point_cloud_result2 = o3d.io.read_point_cloud('csvFiles/resulting0PCL2.pcd')
    tform = np.eye(4)
    tform[0:3, 0:3] = rotation_matrix(0, 0, 0)
    tform[0, 3] = 0
    tform[1, 3] = 0

    points1 = np.array(point_cloud_result1.points)
    points1_homogeneous = np.hstack([points1, np.ones((points1.shape[0], 1))])
    transformed_points1 = (tform @ points1_homogeneous.T).T[:, 0:3]
    point_cloud_result1.points = o3d.utility.Vector3dVector(transformed_points1)

    plt.scatter(point_cloud_result1.points[:, 0], point_cloud_result1.points[:, 1], c='r', marker='.')
    plt.scatter(point_cloud_result2.points[:, 0], point_cloud_result2.points[:, 1], c='b', marker='.')

    plt.figure(9)
    plt.axis('equal')

    gt_transformation_diff = np.linalg.inv(complete_transformation1) @ complete_transformation2
    rpy_tmp = angles_r(gt_transformation_diff[0:3, 0:3], 'xyz') / 180 * np.pi

    test_transformation = np.eye(4)
    test_transformation[0:3, 0:3] = rotation_matrix(np.pi, 0, 0)
    resulting_gt_transformation = test_transformation @ np.linalg.inv(gt_transformation_diff)

    resulting_transformation_of_scan = np.genfromtxt('csvFiles/resultingTransformation0.csv', delimiter=',')
    resulting_yaw_diff[index_current_frame - from_idx] = np.arctan2(np.sin(rpy_tmp[2] - resulting_transformation_of_scan[5]), np.cos(rpy_tmp[2] - resulting_transformation_of_scan[5]))
    translation_diff[index_current_frame - from_idx, :] = resulting_transformation_of_scan[0:2] - resulting_gt_transformation[0:2, 3]

    np.savez(f'resultsOfManyMatching/{name_of_current_testing_set}ICP.npz', resulting_yaw_diff=resulting_yaw_diff, translation_diff=translation_diff)
