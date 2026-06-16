import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import matplotlib
matplotlib.use('Agg')

# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

N = 24
dataset_folder = "/home/tim-external/dataFolder/2022FinalPaperData/onlyAngleStPere/"
which_dataset = "StPere"

time_matrix = np.zeros((N, 12, 52))
error_angle_matrix = np.zeros((N, 12, 52))
error_distance_matrix = np.zeros((N, 12, 52))

for i in range(N):
    which_error = i * 5
    time_matrix[i, :, :] = pd.read_csv(dataset_folder + str(which_error) + "degree/calculationTime1.csv", header=None).values
    error_angle_matrix[i, :, :] = pd.read_csv(dataset_folder + str(which_error) + "degree/errorAngle1.csv", header=None).values
    error_distance_matrix[i, :, :] = pd.read_csv(dataset_folder + str(which_error) + "degree/errorDistance1.csv", header=None).values

error_angle_gicp = np.zeros((N, 52))
error_angle_super = np.zeros((N, 52))
error_angle_ndtd2d = np.zeros((N, 52))
error_angle_ndtp2d = np.zeros((N, 52))
error_angle_ourfms32 = np.zeros((N, 52))
error_angle_ourfms64 = np.zeros((N, 52))
error_angle_ourfms128 = np.zeros((N, 52))
error_angle_ourfms256 = np.zeros((N, 52))
error_angle_ourglobalfms32 = np.zeros((N, 52))
error_angle_ourglobalfms64 = np.zeros((N, 52))
error_angle_ourglobalfms128 = np.zeros((N, 52))
error_angle_ourglobalfms256 = np.zeros((N, 52))

error_distance_gicp = np.zeros((N, 52))
error_distance_super = np.zeros((N, 52))
error_distance_ndtd2d = np.zeros((N, 52))
error_distance_ndtp2d = np.zeros((N, 52))
error_distance_ourfms32 = np.zeros((N, 52))
error_distance_ourfms64 = np.zeros((N, 52))
error_distance_ourfms128 = np.zeros((N, 52))
error_distance_ourfms256 = np.zeros((N, 52))
error_distance_ourglobalfms32 = np.zeros((N, 52))
error_distance_ourglobalfms64 = np.zeros((N, 52))
error_distance_ourglobalfms128 = np.zeros((N, 52))
error_distance_ourglobalfms256 = np.zeros((N, 52))

time_gicp = np.zeros((N, 52))
time_super = np.zeros((N, 52))
time_ndtd2d = np.zeros((N, 52))
time_ndtp2d = np.zeros((N, 52))
time_ourfms32 = np.zeros((N, 52))
time_ourfms64 = np.zeros((N, 52))
time_ourfms128 = np.zeros((N, 52))
time_ourfms256 = np.zeros((N, 52))
time_ourglobalfms32 = np.zeros((N, 52))
time_ourglobalfms64 = np.zeros((N, 52))
time_ourglobalfms128 = np.zeros((N, 52))
time_ourglobalfms256 = np.zeros((N, 52))

for i in range(N):
    error_angle_gicp[i, :] = error_angle_matrix[i, :, 0]
    error_angle_super[i, :] = error_angle_matrix[i, :, 1]
    error_angle_ndtd2d[i, :] = error_angle_matrix[i, :, 2]
    error_angle_ndtp2d[i, :] = error_angle_matrix[i, :, 3]
    error_angle_ourfms32[i, :] = error_angle_matrix[i, :, 4]
    error_angle_ourfms64[i, :] = error_angle_matrix[i, :, 5]
    error_angle_ourfms128[i, :] = error_angle_matrix[i, :, 6]
    error_angle_ourfms256[i, :] = error_angle_matrix[i, :, 7]
    error_angle_ourglobalfms32[i, :] = error_angle_matrix[i, :, 8]
    error_angle_ourglobalfms64[i, :] = error_angle_matrix[i, :, 9]
    error_angle_ourglobalfms128[i, :] = error_angle_matrix[i, :, 10]
    error_angle_ourglobalfms256[i, :] = error_angle_matrix[i, :, 11]

    error_distance_gicp[i, :] = error_distance_matrix[i, :, 0]
    error_distance_super[i, :] = error_distance_matrix[i, :, 1]
    error_distance_ndtd2d[i, :] = error_distance_matrix[i, :, 2]
    error_distance_ndtp2d[i, :] = error_distance_matrix[i, :, 3]
    error_distance_ourfms32[i, :] = error_distance_matrix[i, :, 4]
    error_distance_ourfms64[i, :] = error_distance_matrix[i, :, 5]
    error_distance_ourfms128[i, :] = error_distance_matrix[i, :, 6]
    error_distance_ourfms256[i, :] = error_distance_matrix[i, :, 7]
    error_distance_ourglobalfms32[i, :] = error_distance_matrix[i, :, 8]
    error_distance_ourglobalfms64[i, :] = error_distance_matrix[i, :, 9]
    error_distance_ourglobalfms128[i, :] = error_distance_matrix[i, :, 10]
    error_distance_ourglobalfms256[i, :] = error_distance_matrix[i, :, 11]

    time_gicp[i, :] = time_matrix[i, :, 0]
    time_super[i, :] = time_matrix[i, :, 1]
    time_ndtd2d[i, :] = time_matrix[i, :, 2]
    time_ndtp2d[i, :] = time_matrix[i, :, 3]
    time_ourfms32[i, :] = time_matrix[i, :, 4]
    time_ourfms64[i, :] = time_matrix[i, :, 5]
    time_ourfms128[i, :] = time_matrix[i, :, 6]
    time_ourfms256[i, :] = time_matrix[i, :, 7]
    time_ourglobalfms32[i, :] = time_matrix[i, :, 8]
    time_ourglobalfms64[i, :] = time_matrix[i, :, 9]
    time_ourglobalfms128[i, :] = time_matrix[i, :, 10]
    time_ourglobalfms256[i, :] = time_matrix[i, :, 11]

under_limit = 45
upper_limit = 55
scaling = 1.0
xticks_save = np.arange(5, 121, 5)
scanling_to_degree = 180 / np.pi
general_line_width = 1.0

fig, ax = plt.subplots(figsize=(12, 6))
median_gicp = np.median(error_angle_gicp.T, axis=1)
yerr_gicp = np.array([
    median_gicp - np.percentile(error_angle_gicp.T, under_limit, axis=1),
    np.percentile(error_angle_gicp.T, upper_limit, axis=1) - median_gicp
]) * scanling_to_degree * scaling
ax.errorbar(xticks_save, median_gicp * scanling_to_degree, yerr=yerr_gicp, fmt='o', linewidth=general_line_width, label='GICP')

median_super = np.median(error_angle_super.T, axis=1)
yerr_super = np.array([
    median_super - np.percentile(error_angle_super.T, under_limit, axis=1),
    np.percentile(error_angle_gicp.T, upper_limit, axis=1) - median_super
]) * scanling_to_degree * scaling
ax.errorbar(xticks_save, median_super * scanling_to_degree, yerr=yerr_super, fmt='d', linewidth=general_line_width, label='Super4PCS')

median_ndtd2d = np.median(error_angle_ndtd2d.T, axis=1)
yerr_ndtd2d = np.array([
    median_ndtd2d - np.percentile(error_angle_ndtd2d.T, under_limit, axis=1),
    np.percentile(error_angle_ndtd2d.T, upper_limit, axis=1) - median_ndtd2d
]) * scanling_to_degree * scaling
ax.errorbar(xticks_save, median_ndtd2d * scanling_to_degree, yerr=yerr_ndtd2d, fmt='*', linewidth=general_line_width, label='NDT D2D 2D')

median_ndtp2d = np.median(error_angle_ndtp2d.T, axis=1)
yerr_ndtp2d = np.array([
    median_ndtp2d - np.percentile(error_angle_ndtp2d.T, under_limit, axis=1),
    np.percentile(error_angle_ndtp2d.T, upper_limit, axis=1) - median_ndtp2d
]) * scanling_to_degree * scaling
ax.errorbar(xticks_save, median_ndtp2d * scanling_to_degree, yerr=yerr_ndtp2d, fmt='.', linewidth=general_line_width, label='NDT P2D')

median_ourfms128 = np.median(error_angle_ourfms128.T, axis=1)
yerr_ourfms128 = np.array([
    median_ourfms128 - np.percentile(error_angle_ourfms128.T, under_limit, axis=1),
    np.percentile(error_angle_ourfms128.T, upper_limit, axis=1) - median_ourfms128
]) * scanling_to_degree * scaling
ax.errorbar(xticks_save, median_ourfms128 * scanling_to_degree, yerr=yerr_ourfms128, fmt='x', linewidth=general_line_width, label='Our 2D FMS 128')

median_ourglobalfms128 = np.median(error_angle_ourglobalfms128.T, axis=1)
yerr_ourglobalfms128 = np.array([
    median_ourglobalfms128 - np.percentile(error_angle_ourglobalfms128.T, under_limit, axis=1),
    np.percentile(error_angle_ourglobalfms128.T, upper_limit, axis=1) - median_ourglobalfms128
]) * scanling_to_degree * scaling
ax.errorbar(xticks_save, median_ourglobalfms128 * scanling_to_degree, yerr=yerr_ourglobalfms128, fmt='s', linewidth=general_line_width, label='Our Global FMS 2D 128')

ax.plot(xticks_save, xticks_save, linewidth=1.2, label='error of Initial Guess')
ax.legend(loc='upper left')
ax.set_ylabel("absolute angle error in degree")
ax.set_xlabel("scan rotated in degree")
ax.set_xlim([0, 125])
plt.box(on=True)
plt.grid(True)
name_of_file = "/home/tim-external/Documents/icra2023FMS/figures/error_angle_Rotating_scans" + which_dataset
plt.savefig(name_of_file + ".pdf", bbox_inches='tight')
subprocess.run(["pdfcrop", name_of_file + ".pdf", name_of_file + ".pdf"])

fig, ax = plt.subplots(figsize=(12, 6))
scaling = 0.2
xticks_save = np.arange(5, 121, 5)
general_line_width = 1.0

ax.errorbar(xticks_save, np.mean(error_distance_gicp.T, axis=1), yerr=scaling * np.std(error_distance_gicp.T, axis=1), fmt='o', linewidth=general_line_width, label='GICP')
ax.errorbar(xticks_save, np.mean(error_distance_super.T, axis=1), yerr=scaling * np.std(error_distance_super.T, axis=1), fmt='d', linewidth=general_line_width, label='Super4PCS')
ax.errorbar(xticks_save, np.mean(error_distance_ndtd2d.T, axis=1), yerr=scaling * np.std(error_distance_ndtd2d.T, axis=1), fmt='*', linewidth=general_line_width, label='NDT D2D 2D')
ax.errorbar(xticks_save, np.mean(error_distance_ndtp2d.T, axis=1), yerr=scaling * np.std(error_distance_ndtp2d.T, axis=1), fmt='.', linewidth=general_line_width, label='NDT P2D')
ax.errorbar(xticks_save, np.mean(error_distance_ourfms128.T, axis=1), yerr=np.std(error_distance_ourfms128.T, axis=1), fmt='x', linewidth=general_line_width, label='Our 2D FMS 128')
ax.errorbar(xticks_save, np.mean(error_distance_ourglobalfms128.T, axis=1), yerr=np.std(error_distance_ourglobalfms128.T, axis=1), fmt='s', linewidth=general_line_width, label='Our Global FMS 2D 128')

ax.legend(loc='upper left')
ax.set_ylabel("absolute distance error in m")
ax.set_xlabel("scan rotated by degree")
ax.set_xlim([0, 125])
plt.box(on=True)
plt.grid(True)
name_of_file = "/home/tim-external/Documents/icra2023FMS/figures/error_distance_Rotating_scans" + which_dataset
plt.savefig(name_of_file + ".pdf", bbox_inches='tight')
subprocess.run(["pdfcrop", name_of_file + ".pdf", name_of_file + ".pdf"])
