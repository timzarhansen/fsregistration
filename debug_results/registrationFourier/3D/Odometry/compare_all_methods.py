import numpy as np
import pandas as pd
import os


def compute_mean_dataset(gt_pose_array, estimated_pose_array):
    def minimize_z_rotation_error(t1, t2):
        n = t1.shape[0]

        p1 = t1[:, 0:3, 3]
        p2 = t2[:, 0:3, 3]

        valid_indices = np.all(~np.isnan(p1), axis=1) & np.all(~np.isnan(p2), axis=1)
        p1 = p1[valid_indices, :]
        p2 = p2[valid_indices, :]

        if p1.size == 0 or p2.size == 0:
            raise ValueError('All data contains NaN values. Cannot compute the error.')

        def error_func(theta):
            return compute_error(p1, p2, theta)

        result = minimize_scalar(error_func, bounds=(0, 2 * np.pi), method='bounded')
        theta_min = result.x
        min_error = result.fun

        r_z = np.array([
            [np.cos(theta_min), -np.sin(theta_min), 0],
            [np.sin(theta_min), np.cos(theta_min), 0],
            [0, 0, 1]
        ])

        return r_z, min_error

    def compute_error(p1, p2, theta):
        r_z = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        p2_rotated = p2 @ r_z.T

        error = np.sum((p1 - p2_rotated) ** 2) / p1.size
        return error

    from scipy.optimize import minimize_scalar
    r_z, min_error = minimize_z_rotation_error(gt_pose_array, estimated_pose_array)

    return r_z, min_error


# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

np.set_printoptions(precision=12)

remove_first_n_entries = 1

files = os.listdir("data")
dir_flags = [os.path.isdir(os.path.join("data", f)) for f in files]
sub_folders = [files[i] for i in range(len(files)) if dir_flags[i]]
sub_folder_names = sub_folders[1:]

dataset_and_error = []

for k in range(len(sub_folder_names)):
    name_dataset = sub_folder_names[k]
    print(name_dataset)

    gt_dataset = pd.read_csv("data/" + name_dataset + "/gt.csv")
    pose_dataset = pd.read_csv("data/" + name_dataset + "/poses.csv")

    number_of_rows = len(gt_dataset)
    number_of_nodes = number_of_rows // 4

    gt_pose_array = np.zeros((number_of_nodes, 4, 4))
    estimated_pose_array = np.zeros((number_of_nodes, 4, 4))

    for i in range(number_of_nodes):
        gt_pose_array[i, 0:4, 0:4] = gt_dataset.iloc[i * 4 - 3:i * 4, 0:4].values
        current_pose_estimated = pose_dataset.iloc[i * 4 - 3:i * 4, 0:4].values
        x = current_pose_estimated[0, 3]
        y = current_pose_estimated[1, 3]
        current_pose_estimated[0, 3] = -x
        estimated_pose_array[i, 0:4, 0:4] = current_pose_estimated

    R_z, min_error = compute_mean_dataset(gt_pose_array, estimated_pose_array)
    print(min_error)

    l2_error_list = np.zeros(number_of_nodes - 1)
    for i in range(number_of_nodes - 1):
        gt_diff = np.linalg.inv(gt_pose_array[i, 0:4, 0:4]) @ gt_pose_array[i + 1, 0:4, 0:4]
        est_diff = np.linalg.inv(estimated_pose_array[i, 0:4, 0:4]) @ estimated_pose_array[i + 1, 0:4, 0:4]
        error_matrix = np.linalg.inv(est_diff) @ gt_diff
        l2_error_list[i] = np.linalg.norm(error_matrix[0:3, 3])

    dataset_and_error.append({
        'nameDataset': name_dataset,
        'min_error': min_error,
        'number_of_nodes': number_of_nodes,
        'mean_error_l2': np.nanmean(l2_error_list),
        'std_error_l2': np.nanstd(l2_error_list),
        'median_error_l2': np.nanmedian(l2_error_list)
    })

my_table = pd.DataFrame(dataset_and_error)

my_table.to_excel('comparisonAllMethods.xlsx', index=False)
