import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess

# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

# whichDatasetFolder = "/home/tim-external/dataFolder/2022FinalPaperData/test/"
# whichDataset = "test"

# %% Valentin:
# whichDatasetFolder = "/home/tim-external/dataFolder/2022FinalPaperData/valentinNoNoise52/"
# whichDataset = "valentinBunkerNoNoise52"

# whichDatasetFolder = "/home/tim-external/dataFolder/2022FinalPaperData/valentinHighNoise52/"
# whichDataset = "valentinBunkerHighNoise52"

# whichDatasetFolder = "/home/tim-external/dataFolder/2022FinalPaperData/valentinNoNoise1510/"
# whichDataset = "valentinBunkerNoNoise1510"

# whichDatasetFolder = "/home/tim-external/dataFolder/2022FinalPaperData/valentinHighNoise52NEW/"
# whichDataset = "valentinHighNoise52NEW"

# %% StPere:

# whichDatasetFolder = "/home/tim-external/dataFolder/2022FinalPaperData/stPereNoNoise52/"
# whichDataset = "StPereNoNoise52"

# whichDatasetFolder = "/home/tim-external/dataFolder/2022FinalPaperData/stPereHighNoise52/"
# whichDataset = "StPereHighNoise52"

# whichDatasetFolder = "/home/tim-external/dataFolder/2022FinalPaperData/stPereHighNoise52/"
# whichDataset = "StPereHighNoise52NEW"

# whichDatasetFolder = "/home/tim-external/dataFolder/2022FinalPaperData/stPereNoNoise1510/"
# whichDataset = "StPereNoNoise1510"

# %% Simulation

# whichDatasetFolder = "/home/tim-external/dataFolder/2022FinalPaperData/SimulationResults/lowNoise305_52/"
# whichDataset = "consecutive"

# %% test

which_dataset_folder = "/home/tim-external/dataFolder/2022FinalPaperData/valentinNoNoise1510NEW/"
which_dataset = "valentinNoNoise1510NEW"

threshold = 1
time_matrix = pd.read_csv(which_dataset_folder + "calculationTime" + str(threshold) + ".csv", header=None).values
error_angle_matrix = pd.read_csv(which_dataset_folder + "errorAngle" + str(threshold) + ".csv", header=None).values
error_distance_matrix = pd.read_csv(which_dataset_folder + "errorDistance" + str(threshold) + ".csv", header=None).values

# calculate mean std div and median of computation time error angle and l2 norm error

current_method = [
    "GICP", "Super4PCS", "NDT D2D 2D", "NDT P2D", "Our 2D FMS 32",
    "Our 2D FMS 64", "Our 2D FMS 128", "Our FMS 2D 256", "Our Global 2D FMS 32",
    "Our Global 2D FMS 64", "Our Global 2D FMS 128", "Our Global FMS 2D 256", "no registration"
]

mean_calculation_time = np.zeros(13)
std_calculation_time = np.zeros(13)

for i in range(13):
    mean_calculation_time[i] = np.mean(time_matrix[:, i]) * 0.001
    std_calculation_time[i] = np.std(time_matrix[:, i]) * 0.001
    print(f"{current_method[i]} meanTime: {mean_calculation_time[i]}")
    print(f"{current_method[i]} stdDivTime: {std_calculation_time[i]}")

mean_angle = np.zeros(13)
std_angle = np.zeros(13)

for i in range(13):
    angle_tmp = error_angle_matrix[:, i].copy()
    angle_tmp[angle_tmp > 10000] = 0
    angle_tmp = angle_tmp[angle_tmp != 0]
    angle_tmp = angle_tmp[~np.isnan(angle_tmp)]
    mean_angle[i] = np.mean(angle_tmp) * 180 / np.pi
    std_angle[i] = np.std(angle_tmp) * 180 / np.pi
    print(f"{current_method[i]} meanAngle: {mean_angle[i]}")
    print(f"{current_method[i]} stdDivAngle: {std_angle[i]}")

mean_distance = np.zeros(13)
std_distance = np.zeros(13)

for i in range(13):
    dist_tmp = error_distance_matrix[:, i].copy()
    dist_tmp[dist_tmp > 100000] = 0
    dist_tmp = dist_tmp[dist_tmp != 0]
    dist_tmp = dist_tmp[~np.isnan(dist_tmp)]
    mean_distance[i] = np.mean(dist_tmp)
    std_distance[i] = np.std(dist_tmp)
    print(f"{current_method[i]} meanl2Norm: {mean_distance[i]}")
    print(f"{current_method[i]} stdDivl2Norm: {std_distance[i]}")

# Plot error angle
fig, ax = plt.subplots(figsize=(12, 6))
ax.boxplot(error_angle_matrix, whis=1)
ax.set_yscale('log')
labels = [
    f"GICP {mean_angle[0]:.4f}+-{std_angle[0]:.4f}",
    f"Super4PCS {mean_angle[1]:.4f}+-{std_angle[1]:.4f}",
    f"NDT D2D 2D {mean_angle[2]:.4f}+-{std_angle[2]:.4f}",
    f"NDT P2D {mean_angle[3]:.4f}+-{std_angle[3]:.4f}",
    f"Our 2D FMS 32 {mean_angle[4]:.4f}+-{std_angle[4]:.4f}",
    f"Our 2D FMS 64 {mean_angle[5]:.4f}+-{std_angle[5]:.4f}",
    f"Our 2D FMS 128 {mean_angle[6]:.4f}+-{std_angle[6]:.4f}",
    f"Our FMS 2D 256 {mean_angle[7]:.4f}+-{std_angle[7]:.4f}",
    f"Our Global 2D FMS 32 {mean_angle[8]:.4f}+-{std_angle[8]:.4f}",
    f"Our Global 2D FMS 64 {mean_angle[9]:.4f}+-{std_angle[9]:.4f}",
    f"Our Global 2D FMS 128 {mean_angle[10]:.4f}+-{std_angle[10]:.4f}",
    f"Our Global FMS 2D 256 {mean_angle[11]:.4f}+-{std_angle[11]:.4f}",
    f"initialGuess {mean_angle[12]:.4f}+-{std_angle[12]:.4f}"
]
ax.set_xticklabels(labels, rotation=90)
ax.set_title(f"error angle {which_dataset}")
ax.set_ylabel("error in rad")
name_of_file = "/home/tim-external/Documents/icra2023FMS/figures/boxplotMeanAngles" + which_dataset
plt.savefig(name_of_file + ".pdf", bbox_inches='tight')
subprocess.run(["pdfcrop", name_of_file + ".pdf", name_of_file + ".pdf"])

# Plot computation time
fig, ax = plt.subplots(figsize=(12, 6))
ax.boxplot(time_matrix, whis=1)
ax.set_yscale('log')
labels = [
    f"GICP {mean_calculation_time[0]:.4f}+-{std_calculation_time[0]:.4f}",
    f"Super4PCS {mean_calculation_time[1]:.4f}+-{std_calculation_time[1]:.4f}",
    f"NDT D2D 2D {mean_calculation_time[2]:.4f}+-{std_calculation_time[2]:.4f}",
    f"NDT P2D {mean_calculation_time[3]:.4f}+-{std_calculation_time[3]:.4f}",
    f"Our 2D FMS 32 {mean_calculation_time[4]:.4f}+-{std_calculation_time[4]:.4f}",
    f"Our 2D FMS 64 {mean_calculation_time[5]:.4f}+-{std_calculation_time[5]:.4f}",
    f"Our 2D FMS 128 {mean_calculation_time[6]:.4f}+-{std_calculation_time[6]:.4f}",
    f"Our FMS 2D 256 {mean_calculation_time[7]:.4f}+-{std_calculation_time[7]:.4f}",
    f"Our Global 2D FMS 32 {mean_calculation_time[8]:.4f}+-{std_calculation_time[8]:.4f}",
    f"Our Global 2D FMS 64 {mean_calculation_time[9]:.4f}+-{std_calculation_time[9]:.4f}",
    f"Our Global 2D FMS 128 {mean_calculation_time[10]:.4f}+-{std_calculation_time[10]:.4f}",
    f"Our Global FMS 2D 256 {mean_calculation_time[11]:.4f}+-{std_calculation_time[11]:.4f}",
    f"initialGuess {mean_calculation_time[12]:.4f}+-{std_calculation_time[12]:.4f}"
]
ax.set_xticklabels(labels, rotation=90)
ax.set_title(f"computation Time {which_dataset}")
ax.set_ylabel("computation Time in micro s")
name_of_file = "/home/tim-external/Documents/icra2023FMS/figures/boxplotMeanTime" + which_dataset
plt.savefig(name_of_file + ".pdf", bbox_inches='tight')
subprocess.run(["pdfcrop", name_of_file + ".pdf", name_of_file + ".pdf"])

# Plot error distance
dist_tmp = error_distance_matrix.copy()
dist_tmp[dist_tmp[:, 1] > 10000, 3] = 0

fig, ax = plt.subplots(figsize=(12, 6))
ax.boxplot(dist_tmp, whis=1)
ax.set_yscale('log')
labels = [
    f"GICP {mean_distance[0]:.4f}+-{std_distance[0]:.4f}",
    f"Super4PCS {mean_distance[1]:.4f}+-{std_distance[1]:.4f}",
    f"NDT D2D 2D {mean_distance[2]:.4f}+-{std_distance[2]:.4f}",
    f"NDT P2D {mean_distance[3]:.4f}+-{std_distance[3]:.4f}",
    f"Our 2D FMS 32 {mean_distance[4]:.4f}+-{std_distance[4]:.4f}",
    f"Our 2D FMS 64 {mean_distance[5]:.4f}+-{std_distance[5]:.4f}",
    f"Our 2D FMS 128 {mean_distance[6]:.4f}+-{std_distance[6]:.4f}",
    f"Our FMS 2D 256 {mean_distance[7]:.4f}+-{std_distance[7]:.4f}",
    f"Our Global 2D FMS 32 {mean_distance[8]:.4f}+-{std_distance[8]:.4f}",
    f"Our Global 2D FMS 64 {mean_distance[9]:.4f}+-{std_distance[9]:.4f}",
    f"Our Global 2D FMS 128 {mean_distance[10]:.4f}+-{std_distance[10]:.4f}",
    f"Our Global FMS 2D 256 {mean_distance[11]:.4f}+-{std_distance[11]:.4f}",
    f"initialGuess {mean_distance[12]:.4f}+-{std_distance[12]:.4f}"
]
ax.set_xticklabels(labels, rotation=90)
ax.set_title(f"error l2-norm {which_dataset}")
ax.set_ylabel("error l2-norm in m")
name_of_file = "/home/tim-external/Documents/icra2023FMS/figures/boxplotMeanDistance" + which_dataset
plt.savefig(name_of_file + ".pdf", bbox_inches='tight')
subprocess.run(["pdfcrop", name_of_file + ".pdf", name_of_file + ".pdf"])
