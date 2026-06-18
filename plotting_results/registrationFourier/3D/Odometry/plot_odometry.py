import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
# clc, clear, clf - skipped

name_dataset = "predator_30_15.000000_0.100000_Alpha"

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

gt_pose_array = gt_pose_array[1:-1, :, :]
estimated_pose_array = estimated_pose_array[1:-1, :, :]

r_z, min_error = compute_mean_dataset(gt_pose_array, estimated_pose_array)
print('Optimal rotation matrix around Z-axis:')
print(r_z)
print(f'Minimum squared mean error: {min_error}')

estimated_pose_array_rotated = estimated_pose_array.copy()
for i in range(estimated_pose_array_rotated.shape[0]):
    estimated_pose_array_rotated[i, 0:3, 3] = estimated_pose_array_rotated[i, 0:3, 3] @ r_z.T

plt.figure(1)
plt.clf()
plt.hold(True)
plt.plot(gt_pose_array[:, 0, 3], gt_pose_array[:, 1, 3])
plt.plot(estimated_pose_array[:, 0, 3], estimated_pose_array[:, 1, 3])
plt.box(True)
plt.axis('equal')

plt.figure(2)
plt.clf()
plt.hold(True)
plt.plot(gt_pose_array[:, 0, 3], gt_pose_array[:, 1, 3])
plt.plot(estimated_pose_array_rotated[:, 0, 3], estimated_pose_array_rotated[:, 1, 3])
plt.box(True)
plt.axis('equal')
