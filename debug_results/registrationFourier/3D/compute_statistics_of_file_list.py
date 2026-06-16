import numpy as np
import pandas as pd
import glob
import os


def rotm2quat(R):
    # Convert rotation matrix to quaternion (w, x, y, z)
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z])


def quatmultiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def quatconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


# listFiles = glob.glob("testFiles/o*")
list_files = glob.glob("testFiles/o*")
size_list_of_files = len(list_files)

output_list = np.zeros((size_list_of_files, 12))

for index_file in range(size_list_of_files):
    current_name = os.path.basename(list_files[index_file])
    current_folder = os.path.dirname(list_files[index_file])
    current_data = pd.read_csv(os.path.join(current_folder, current_name), sep='\t', header=None)
    number_of_solutions = current_data.iloc[4, 0]
    gt_solution = current_data.iloc[0:4, 0:4].values

    # get parameters of this matching Method
    split_string = current_name.split("_")

    output_list[index_file, 0] = float(split_string[0].replace("outfile", ""))
    output_list[index_file, 1] = float(split_string[1])
    output_list[index_file, 2] = float(split_string[2])
    output_list[index_file, 3] = float(split_string[3])
    output_list[index_file, 4] = float(split_string[4])
    output_list[index_file, 5] = float(split_string[5])
    output_list[index_file, 6] = float(split_string[6])
    output_list[index_file, 7] = float(split_string[7].replace(".txt", ""))

    highest_peak = 0
    highest_peak_index = 0
    l2_error_list = np.zeros(number_of_solutions)
    angle_error_list = np.zeros(number_of_solutions)

    for index_estimation in range(number_of_solutions):
        current_estimation_matrix = current_data.iloc[index_estimation * 5 + 1:(index_estimation + 1) * 5 - 1, 0:4].values
        current_peak = current_data.iloc[(index_estimation + 1) * 5, 0]
        # find highest Peak
        if current_peak > highest_peak:
            highest_peak = current_peak
            highest_peak_index = index_estimation
        quat_estimation = rotm2quat(current_estimation_matrix[0:3, 0:3])
        trans_estimation = current_estimation_matrix[0:3, 3]
        # compute error l2 norm
        l2_error_list[index_estimation] = np.linalg.norm(trans_estimation - gt_solution[0:3, 3])
        # compute error angle
        z = quatmultiply(quatconj(rotm2quat(gt_solution[0:3, 0:3])), quat_estimation)
        angle_error_list[index_estimation] = 2 * np.arccos(z[0]) * 180 / np.pi

    # find best Fit
    index_bestfit = np.argmin(l2_error_list * angle_error_list)

    # find highest Peak
    highest_peak_estimation_matrix = current_data.iloc[highest_peak_index * 5 + 1:(highest_peak_index + 1) * 5 - 1, 0:4].values
    quat_estimation = rotm2quat(highest_peak_estimation_matrix[0:3, 0:3])
    trans_estimation = highest_peak_estimation_matrix[0:3, 3]
    highest_peak_l2_error = np.linalg.norm(trans_estimation - gt_solution[0:3, 3])
    z = quatmultiply(quatconj(rotm2quat(gt_solution[0:3, 0:3])), quat_estimation)
    highest_peak_angle_error = 2 * np.arccos(z[0]) * 180 / np.pi

    output_list[index_file, 8] = l2_error_list[index_bestfit]
    output_list[index_file, 9] = angle_error_list[index_bestfit]
    output_list[index_file, 10] = highest_peak_l2_error
    output_list[index_file, 11] = highest_peak_angle_error

output_list_32 = output_list[output_list[:, 0] == 32, :]
output_list_64 = output_list[output_list[:, 0] == 64, :]
correlation_32 = np.corrcoef(output_list_32.T)
correlation_64 = np.corrcoef(output_list_64.T)

# Load saved data
# output_list_with_all = np.load('outputListWithAll.mat')  # Would need scipy.io.loadmat

output_list_32 = output_list[output_list[:, 0] == 32, :]
output_list_64 = output_list[output_list[:, 0] == 64, :]
output_list_128 = output_list[output_list[:, 0] == 128, :]

correlation_32 = np.corrcoef(output_list_32.T)
correlation_64 = np.corrcoef(output_list_64.T)
correlation_128 = np.corrcoef(output_list_128.T)
