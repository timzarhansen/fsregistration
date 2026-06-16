import numpy as np
import matplotlib.pyplot as plt


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


def rotation_angle(r1, r2):
    r = r1.T @ r2
    trace_r = np.trace(r)
    cos_theta = (trace_r - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)
    angle = np.arccos(cos_theta) * 180 / np.pi
    return angle


threshold_trans = 0.4
threshold_rot = 10

folder = "resultFilesPredatorMatching/"
dataset_name = "outfile32_0_4_12_0.01_0.01__0.csv"
data_name = folder + dataset_name
data = np.genfromtxt(data_name, delimiter=",")

resulting_data = np.zeros((len(data), 8))

for i in range(len(data)):
    gt_transformation = transformations_matrix(data[i, 4], data[i, 5], data[i, 6], data[i, 7], data[i, 8], data[i, 9])
    highest_peak_transformation = transformations_matrix(data[i, 10], data[i, 11], data[i, 12], data[i, 13], data[i, 14], data[i, 15])
    best_peak_transformation = transformations_matrix(data[i, 16], data[i, 17], data[i, 18], data[i, 19], data[i, 20], data[i, 21])

    gt_trans = gt_transformation[0:3, 3]
    gt_rot = gt_transformation[0:3, 0:3]

    highest_trans = highest_peak_transformation[0:3, 3]
    highest_rot = highest_peak_transformation[0:3, 0:3]

    best_trans = best_peak_transformation[0:3, 3]
    best_rot = best_peak_transformation[0:3, 0:3]

    trans_high = np.linalg.norm(gt_trans - highest_trans)
    trans_best = np.linalg.norm(gt_trans - best_trans)

    rot_high = rotation_angle(gt_rot, highest_rot)
    rot_best = rotation_angle(gt_rot, best_rot)

    within_trans = trans_high <= threshold_trans
    within_rot = rot_high <= threshold_rot
    within_both = within_trans and within_rot
    threshold_high = int(within_both)

    within_trans = trans_best <= threshold_trans
    within_rot = rot_best <= threshold_rot
    within_both = within_trans and within_rot
    threshold_best = int(within_both)

    percentage = data[i, 3]
    numberofsolutions = data[i, 2]

    resulting_data[i, 0] = percentage
    resulting_data[i, 1] = numberofsolutions
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

print("mean trans high: " + str(mean_trans_high) + " +- " + str(std_trans_high))
print("mean rot high: " + str(mean_rot_high) + " +- " + str(std_rot_high))
print("mean trans best: " + str(mean_trans_best) + " +- " + str(std_trans_best))
print("mean rot best: " + str(mean_rot_best) + " +- " + str(std_rot_best))

print("median trans high: " + str(median_trans_high))
print("median rot high: " + str(median_rot_high))
print("median trans best: " + str(median_trans_best))
print("median rot best: " + str(median_rot_best))

correlation_best = np.corrcoef(resulting_data[:, 6], resulting_data[:, 0])[0, 1]
percentage_best = np.sum(resulting_data[:, 6]) / len(resulting_data[:, 6]) * 100
print("Percentage correct Best: " + str(percentage_best))
correlation_high = np.corrcoef(resulting_data[:, 7], resulting_data[:, 0])[0, 1]
percentage_high = np.sum(resulting_data[:, 7]) / len(resulting_data[:, 7]) * 100
print("Percentage correct High: " + str(percentage_high))

print("Correlation Best: " + str(correlation_best))
print("Correlation High: " + str(correlation_high))

plt.figure(1)
plt.hist(resulting_data[:, 0], 20)

plt.figure(2)
plt.hist(resulting_data[:, 1], 20)

plt.figure(3)
plt.boxplot([resulting_data[:, 2], resulting_data[:, 4]])

plt.figure(4)
plt.boxplot([resulting_data[:, 3], resulting_data[:, 5]])

fullresults = [dataset_name, mean_trans_high, mean_rot_high, mean_trans_best, mean_rot_best, std_trans_high, std_rot_high, std_trans_best, std_rot_best, median_trans_high, median_rot_high, median_trans_best, median_rot_best]
