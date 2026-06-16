import numpy as np
import pandas as pd
import os


def starts_with_outfile(str_input):
    prefix = 'outfile'
    return str_input.startswith(prefix)


def rotation_angle_difference(r1, r2):
    r = r1.T @ r2
    trace_r = np.trace(r)
    cos_theta = (trace_r - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)
    angle = np.arccos(cos_theta) * 180 / np.pi
    return angle


def transformations_matrix(roll, pitch, yaw, x, y, z):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    rotation = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ])

    transformation_matrix = np.array([
        [rotation[0, 0], rotation[0, 1], rotation[0, 2], x],
        [rotation[1, 0], rotation[1, 1], rotation[1, 2], y],
        [rotation[2, 0], rotation[2, 1], rotation[2, 2], z],
        [0, 0, 0, 1]
    ])

    return transformation_matrix


def evaluate_file_3d_predator_matching(folder, dataset_name, threshold_trans, threshold_rot):
    data_name = folder + "/" + dataset_name
    data = pd.read_csv(data_name).values
    data_set_length = len(data)
    resulting_data = np.zeros((data_set_length, 8))

    for i in range(data_set_length):
        if starts_with_outfile(dataset_name):
            gt_transformation = transformations_matrix(data[i, 2], data[i, 3], data[i, 4], data[i, 5], data[i, 6], data[i, 7])
            highest_peak_transformation = transformations_matrix(data[i, 8], data[i, 9], data[i, 10], data[i, 11], data[i, 12], data[i, 13])

            gt_trans = gt_transformation[0:3, 3]
            gt_rot = gt_transformation[0:3, 0:3]

            highest_trans = highest_peak_transformation[0:3, 3]
            highest_rot = highest_peak_transformation[0:3, 0:3]

            trans_high = np.linalg.norm(gt_trans - highest_trans)
            rot_high = rotation_angle_difference(gt_rot, highest_rot)

            within_trans = trans_high <= threshold_trans
            within_rot = rot_high <= threshold_rot
            within_both = within_trans and within_rot
            threshold_high = float(within_both)

            percentage = data[i, 1]

            resulting_data[i, 0] = percentage
            resulting_data[i, 1] = 0
            resulting_data[i, 2] = trans_high
            resulting_data[i, 3] = rot_high
            resulting_data[i, 4] = 0
            resulting_data[i, 5] = 0
            resulting_data[i, 6] = 0
            resulting_data[i, 7] = threshold_high

        else:
            gt_transformation = transformations_matrix(data[i, 3], data[i, 4], data[i, 5], data[i, 6], data[i, 7], data[i, 8])
            highest_peak_transformation = transformations_matrix(data[i, 9], data[i, 10], data[i, 11], data[i, 12], data[i, 13], data[i, 14])
            best_peak_transformation = transformations_matrix(data[i, 15], data[i, 16], data[i, 17], data[i, 18], data[i, 19], data[i, 20])

            gt_trans = gt_transformation[0:3, 3]
            gt_rot = gt_transformation[0:3, 0:3]

            highest_trans = highest_peak_transformation[0:3, 3]
            highest_rot = highest_peak_transformation[0:3, 0:3]

            best_trans = best_peak_transformation[0:3, 3]
            best_rot = best_peak_transformation[0:3, 0:3]

            trans_high = np.linalg.norm(gt_trans - highest_trans)
            trans_best = np.linalg.norm(gt_trans - best_trans)

            rot_high = rotation_angle_difference(gt_rot, highest_rot)
            rot_best = rotation_angle_difference(gt_rot, best_rot)

            within_trans = trans_high <= threshold_trans
            within_rot = rot_high <= threshold_rot
            within_both = within_trans and within_rot
            threshold_high = float(within_both)

            within_trans = trans_best <= threshold_trans
            within_rot = rot_best <= threshold_rot
            within_both = within_trans and within_rot
            threshold_best = float(within_both)

            percentage = data[i, 2]
            number_of_solutions = data[i, 1]

            resulting_data[i, 0] = percentage
            resulting_data[i, 1] = number_of_solutions
            resulting_data[i, 2] = trans_high
            resulting_data[i, 3] = rot_high
            resulting_data[i, 4] = trans_best
            resulting_data[i, 5] = rot_best
            resulting_data[i, 6] = threshold_best
            resulting_data[i, 7] = threshold_high

    mean_trans_high = np.mean(resulting_data[:, 2])
    mean_rot_high = np.mean(resulting_data[:, 3])
    mean_trans_best = np.mean(resulting_data[:, 4])
    mean_rot_best = np.mean(resulting_data[:, 5])

    std_trans_high = np.std(resulting_data[:, 2])
    std_rot_high = np.std(resulting_data[:, 3])
    std_trans_best = np.std(resulting_data[:, 4])
    std_rot_best = np.std(resulting_data[:, 5])

    median_trans_high = np.median(resulting_data[:, 2])
    median_rot_high = np.median(resulting_data[:, 3])
    median_trans_best = np.median(resulting_data[:, 4])
    median_rot_best = np.median(resulting_data[:, 5])

    print(f"mean trans high: {mean_trans_high} +- {std_trans_high}")
    print(f"mean rot high: {mean_rot_high} +- {std_rot_high}")
    print(f"mean trans best: {mean_trans_best} +- {std_trans_best}")
    print(f"mean rot best: {mean_rot_best} +- {std_rot_best}")

    print(f"median trans high: {median_trans_high}")
    print(f"median rot high: {median_rot_high}")
    print(f"median trans best: {median_trans_best}")
    print(f"median rot best: {median_rot_best}")

    correlation_best = np.corrcoef(resulting_data[:, 6], resulting_data[:, 0])[0, 1]
    percentage_best = np.sum(resulting_data[:, 6]) / len(resulting_data[:, 6]) * 100
    print(f"Percentage correct Best: {percentage_best}")

    correlation_high = np.corrcoef(resulting_data[:, 7], resulting_data[:, 0])[0, 1]
    percentage_high = np.sum(resulting_data[:, 7]) / len(resulting_data[:, 7]) * 100
    print(f"Percentage correct High: {percentage_high}")

    print(f'Correlation Best: {correlation_best}')
    print(f'Correlation High: {correlation_high}')

    fullresults = np.array([mean_trans_high, mean_rot_high, mean_trans_best, mean_rot_best,
                            std_trans_high, std_rot_high, std_trans_best, std_rot_best,
                            median_trans_high, median_rot_high, median_trans_best, median_rot_best,
                            correlation_best, correlation_high, percentage_best, percentage_high])

    return fullresults


# Clear workspace equivalent (not strictly needed in Python)

threshold_trans = 0.4
threshold_rot = 10

folder = "paperTests"

files = [f for f in os.listdir(folder) if f.endswith('.csv')]

if not files:
    raise FileNotFoundError('No CSV files found in the specified folder.')

results_list = []

for filename in files:
    fullresults = evaluate_file_3d_predator_matching(folder, filename, threshold_trans, threshold_rot)

    results_list.append({
        'FileName': filename,
        'Mean_Translation_High': fullresults[0],
        'Mean_Rotation_High': fullresults[1],
        'Mean_Translation_Best': fullresults[2],
        'Mean_Rotation_Best': fullresults[3],
        'Std_Translation_High': fullresults[4],
        'Std_Rotation_High': fullresults[5],
        'Std_Translation_Best': fullresults[6],
        'Std_Rotation_Best': fullresults[7],
        'Median_Translation_High': fullresults[8],
        'Median_Rotation_High': fullresults[9],
        'Median_Translation_Best': fullresults[10],
        'Median_Rotation_Best': fullresults[11],
        'correlation_best': fullresults[12],
        'correlation_high': fullresults[13],
        'threshold_best': fullresults[14],
        'threshold_high': fullresults[15]
    })

results_table = pd.DataFrame(results_list)

print('Results Table (FileName + 12 Statistics):')
print(results_table)

results_table.to_csv('results_analysis.csv', index=False)
