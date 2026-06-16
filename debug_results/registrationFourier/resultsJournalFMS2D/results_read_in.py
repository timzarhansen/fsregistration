import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

results_experiment = pd.read_csv("results.csv", header=None).values

city = ['GICP', 'GICP', 'GICP',
        'SUPER4PCS', 'SUPER4PCS', 'SUPER4PCS',
        'NDT D2D 2D', 'NDT D2D 2D', 'NDT D2D 2D',
        'NDT P2D', 'NDT P2D', 'NDT P2D',
        'FourierMellinTransform', 'FourierMellinTransform', 'FourierMellinTransform',
        'Our FMS 2D', 'Our FMS 2D', 'Our FMS 2D',
        'Our FMS 2D Global', 'Our FMS 2D Global', 'Our FMS 2D Global',
        'initial guess']

result_data = []
for i in range(22):
    result_data.append({
        'registrationMethod': city[i],
        'voxelSize': 0,
        'meanL2': 0,
        'stdL2': 0,
        'meanAngle': 0,
        'stdAngle': 0,
        'meanComputationTime': 0,
        'stdComputationTime': 0
    })

result_table = pd.DataFrame(result_data)

# initial guess
results_initial_guess = results_experiment[results_experiment[:, 0] == -1, :]
result_table.loc[21, 'voxelSize'] = 0
result_table.loc[21, 'meanL2'] = np.mean(np.abs(results_initial_guess[:, 3]))
result_table.loc[21, 'stdL2'] = np.std(results_initial_guess[:, 3])
result_table.loc[21, 'meanAngle'] = np.mean(np.abs(results_initial_guess[:, 4]))
result_table.loc[21, 'stdAngle'] = np.std(results_initial_guess[:, 4])
result_table.loc[21, 'meanComputationTime'] = np.mean(np.abs(results_initial_guess[:, 5]))
result_table.loc[21, 'stdComputationTime'] = np.std(results_initial_guess[:, 5])

# GICP
results_gicp = results_experiment[results_experiment[:, 0] == 1, :]
results_gicp_128 = results_gicp[results_gicp[:, 1] == 128, :]
result_table.loc[0, 'voxelSize'] = 128
result_table.loc[0, 'meanL2'] = np.mean(np.abs(results_gicp_128[:, 3]))
result_table.loc[0, 'stdL2'] = np.std(np.abs(results_gicp_128[:, 3]))
result_table.loc[0, 'meanAngle'] = np.mean(np.abs(results_gicp_128[:, 4]))
result_table.loc[0, 'stdAngle'] = np.std(np.abs(results_gicp_128[:, 4]))
result_table.loc[0, 'meanComputationTime'] = np.mean(np.abs(results_gicp_128[:, 5]))
result_table.loc[0, 'stdComputationTime'] = np.std(np.abs(results_gicp_128[:, 5]))

results_gicp_256 = results_gicp[results_gicp[:, 1] == 256, :]
result_table.loc[1, 'voxelSize'] = 256
result_table.loc[1, 'meanL2'] = np.mean(np.abs(results_gicp_256[:, 3]))
result_table.loc[1, 'stdL2'] = np.std(results_gicp_256[:, 3])
result_table.loc[1, 'meanAngle'] = np.mean(np.abs(results_gicp_256[:, 4]))
result_table.loc[1, 'stdAngle'] = np.std(results_gicp_256[:, 4])
result_table.loc[1, 'meanComputationTime'] = np.mean(np.abs(results_gicp_256[:, 5]))
result_table.loc[1, 'stdComputationTime'] = np.std(results_gicp_256[:, 5])

results_gicp_512 = results_gicp[results_gicp[:, 1] == 512, :]
result_table.loc[2, 'voxelSize'] = 512
result_table.loc[2, 'meanL2'] = np.mean(np.abs(results_gicp_512[:, 3]))
result_table.loc[2, 'stdL2'] = np.std(results_gicp_512[:, 3])
result_table.loc[2, 'meanAngle'] = np.mean(np.abs(results_gicp_512[:, 4]))
result_table.loc[2, 'stdAngle'] = np.std(results_gicp_512[:, 4])
result_table.loc[2, 'meanComputationTime'] = np.mean(np.abs(results_gicp_512[:, 5]))
result_table.loc[2, 'stdComputationTime'] = np.std(results_gicp_512[:, 5])

# SUPER 4PCS
results_super4pcs = results_experiment[results_experiment[:, 0] == 2, :]
results_super4pcs_128 = results_super4pcs[results_super4pcs[:, 1] == 128, :]
result_table.loc[3, 'voxelSize'] = 128
result_table.loc[3, 'meanL2'] = np.mean(np.abs(results_super4pcs_128[:, 3]))
result_table.loc[3, 'stdL2'] = np.std(results_super4pcs_128[:, 3])
result_table.loc[3, 'meanAngle'] = np.mean(np.abs(results_super4pcs_128[:, 4]))
result_table.loc[3, 'stdAngle'] = np.std(results_super4pcs_128[:, 4])
result_table.loc[3, 'meanComputationTime'] = np.mean(np.abs(results_super4pcs_128[:, 5]))
result_table.loc[3, 'stdComputationTime'] = np.std(results_super4pcs_128[:, 5])

results_super4pcs_256 = results_super4pcs[results_super4pcs[:, 1] == 256, :]
result_table.loc[4, 'voxelSize'] = 256
result_table.loc[4, 'meanL2'] = np.mean(np.abs(results_super4pcs_256[:, 3]))
result_table.loc[4, 'stdL2'] = np.std(results_super4pcs_256[:, 3])
result_table.loc[4, 'meanAngle'] = np.mean(np.abs(results_super4pcs_256[:, 4]))
result_table.loc[4, 'stdAngle'] = np.std(results_super4pcs_256[:, 4])
result_table.loc[4, 'meanComputationTime'] = np.mean(np.abs(results_super4pcs_256[:, 5]))
result_table.loc[4, 'stdComputationTime'] = np.std(results_super4pcs_256[:, 5])

results_super4pcs_512 = results_super4pcs[results_super4pcs[:, 1] == 512, :]
result_table.loc[5, 'voxelSize'] = 512
result_table.loc[5, 'meanL2'] = np.mean(np.abs(results_super4pcs_512[:, 3]))
result_table.loc[5, 'stdL2'] = np.std(results_super4pcs_512[:, 3])
result_table.loc[5, 'meanAngle'] = np.mean(np.abs(results_super4pcs_512[:, 4]))
result_table.loc[5, 'stdAngle'] = np.std(results_super4pcs_512[:, 4])
result_table.loc[5, 'meanComputationTime'] = np.mean(np.abs(results_super4pcs_512[:, 5]))
result_table.loc[5, 'stdComputationTime'] = np.std(results_super4pcs_512[:, 5])

# NDTD2D2D
results_ndtd2d2d = results_experiment[results_experiment[:, 0] == 3, :]
results_ndtd2d2d_128 = results_ndtd2d2d[results_ndtd2d2d[:, 1] == 128, :]
result_table.loc[6, 'voxelSize'] = 128
result_table.loc[6, 'meanL2'] = np.mean(np.abs(results_ndtd2d2d_128[:, 3]))
result_table.loc[6, 'stdL2'] = np.std(results_ndtd2d2d_128[:, 3])
result_table.loc[6, 'meanAngle'] = np.mean(np.abs(results_ndtd2d2d_128[:, 4]))
result_table.loc[6, 'stdAngle'] = np.std(results_ndtd2d2d_128[:, 4])
result_table.loc[6, 'meanComputationTime'] = np.mean(np.abs(results_ndtd2d2d_128[:, 5]))
result_table.loc[6, 'stdComputationTime'] = np.std(results_ndtd2d2d_128[:, 5])

results_ndtd2d2d_256 = results_ndtd2d2d[results_ndtd2d2d[:, 1] == 256, :]
result_table.loc[7, 'voxelSize'] = 256
result_table.loc[7, 'meanL2'] = np.mean(np.abs(results_ndtd2d2d_256[:, 3]))
result_table.loc[7, 'stdL2'] = np.std(results_ndtd2d2d_256[:, 3])
result_table.loc[7, 'meanAngle'] = np.mean(np.abs(results_ndtd2d2d_256[:, 4]))
result_table.loc[7, 'stdAngle'] = np.std(results_ndtd2d2d_256[:, 4])
result_table.loc[7, 'meanComputationTime'] = np.mean(np.abs(results_ndtd2d2d_256[:, 5]))
result_table.loc[7, 'stdComputationTime'] = np.std(results_ndtd2d2d_256[:, 5])

results_ndtd2d2d_512 = results_ndtd2d2d[results_ndtd2d2d[:, 1] == 512, :]
result_table.loc[8, 'voxelSize'] = 512
result_table.loc[8, 'meanL2'] = np.mean(np.abs(results_ndtd2d2d_512[:, 3]))
result_table.loc[8, 'stdL2'] = np.std(results_ndtd2d2d_512[:, 3])
result_table.loc[8, 'meanAngle'] = np.mean(np.abs(results_ndtd2d2d_512[:, 4]))
result_table.loc[8, 'stdAngle'] = np.std(results_ndtd2d2d_512[:, 4])
result_table.loc[8, 'meanComputationTime'] = np.mean(np.abs(results_ndtd2d2d_512[:, 5]))
result_table.loc[8, 'stdComputationTime'] = np.std(results_ndtd2d2d_512[:, 5])

# NDT P2D
results_ndtp2d = results_experiment[results_experiment[:, 0] == 5, :]
results_ndtp2d_128 = results_ndtp2d[results_ndtp2d[:, 1] == 128, :]
result_table.loc[9, 'voxelSize'] = 128
result_table.loc[9, 'meanL2'] = np.mean(np.abs(results_ndtp2d_128[:, 3]))
result_table.loc[9, 'stdL2'] = np.std(results_ndtp2d_128[:, 3])
result_table.loc[9, 'meanAngle'] = np.mean(np.abs(results_ndtp2d_128[:, 4]))
result_table.loc[9, 'stdAngle'] = np.std(results_ndtp2d_128[:, 4])
result_table.loc[9, 'meanComputationTime'] = np.mean(np.abs(results_ndtp2d_128[:, 5]))
result_table.loc[9, 'stdComputationTime'] = np.std(results_ndtp2d_128[:, 5])

results_ndtp2d_256 = results_ndtp2d[results_ndtp2d[:, 1] == 256, :]
result_table.loc[10, 'voxelSize'] = 256
result_table.loc[10, 'meanL2'] = np.mean(np.abs(results_ndtp2d_256[:, 3]))
result_table.loc[10, 'stdL2'] = np.std(results_ndtp2d_256[:, 3])
result_table.loc[10, 'meanAngle'] = np.mean(np.abs(results_ndtp2d_256[:, 4]))
result_table.loc[10, 'stdAngle'] = np.std(results_ndtp2d_256[:, 4])
result_table.loc[10, 'meanComputationTime'] = np.mean(np.abs(results_ndtp2d_256[:, 5]))
result_table.loc[10, 'stdComputationTime'] = np.std(results_ndtp2d_256[:, 5])

results_ndtp2d_512 = results_ndtp2d[results_ndtp2d[:, 1] == 512, :]
result_table.loc[11, 'voxelSize'] = 512
result_table.loc[11, 'meanL2'] = np.mean(np.abs(results_ndtp2d_512[:, 3]))
result_table.loc[11, 'stdL2'] = np.std(results_ndtp2d_512[:, 3])
result_table.loc[11, 'meanAngle'] = np.mean(np.abs(results_ndtp2d_512[:, 4]))
result_table.loc[11, 'stdAngle'] = np.std(results_ndtp2d_512[:, 4])
result_table.loc[11, 'meanComputationTime'] = np.mean(np.abs(results_ndtp2d_512[:, 5]))
result_table.loc[11, 'stdComputationTime'] = np.std(results_ndtp2d_512[:, 5])

# Fourier Mellin
results_fourier_mellin = results_experiment[results_experiment[:, 0] == 5, :]
results_fourier_mellin_128 = results_fourier_mellin[results_fourier_mellin[:, 1] == 128, :]
result_table.loc[12, 'voxelSize'] = 128
result_table.loc[12, 'meanL2'] = np.mean(np.abs(results_fourier_mellin_128[:, 3]))
result_table.loc[12, 'stdL2'] = np.std(results_fourier_mellin_128[:, 3])
result_table.loc[12, 'meanAngle'] = np.mean(np.abs(results_fourier_mellin_128[:, 4]))
result_table.loc[12, 'stdAngle'] = np.std(results_fourier_mellin_128[:, 4])
result_table.loc[12, 'meanComputationTime'] = np.mean(np.abs(results_fourier_mellin_128[:, 5]))
result_table.loc[12, 'stdComputationTime'] = np.std(results_fourier_mellin_128[:, 5])

results_fourier_mellin_256 = results_fourier_mellin[results_fourier_mellin[:, 1] == 256, :]
result_table.loc[13, 'voxelSize'] = 256
result_table.loc[13, 'meanL2'] = np.mean(np.abs(results_fourier_mellin_256[:, 3]))
result_table.loc[13, 'stdL2'] = np.std(results_fourier_mellin_256[:, 3])
result_table.loc[13, 'meanAngle'] = np.mean(np.abs(results_fourier_mellin_256[:, 4]))
result_table.loc[13, 'stdAngle'] = np.std(results_fourier_mellin_256[:, 4])
result_table.loc[13, 'meanComputationTime'] = np.mean(np.abs(results_fourier_mellin_256[:, 5]))
result_table.loc[13, 'stdComputationTime'] = np.std(results_fourier_mellin_256[:, 5])

results_fourier_mellin_512 = results_fourier_mellin[results_fourier_mellin[:, 1] == 512, :]
result_table.loc[14, 'voxelSize'] = 512
result_table.loc[14, 'meanL2'] = np.mean(np.abs(results_fourier_mellin_512[:, 3]))
result_table.loc[14, 'stdL2'] = np.std(results_fourier_mellin_512[:, 3])
result_table.loc[14, 'meanAngle'] = np.mean(np.abs(results_fourier_mellin_512[:, 4]))
result_table.loc[14, 'stdAngle'] = np.std(results_fourier_mellin_512[:, 4])
result_table.loc[14, 'meanComputationTime'] = np.mean(np.abs(results_fourier_mellin_512[:, 5]))
result_table.loc[14, 'stdComputationTime'] = np.std(results_fourier_mellin_512[:, 5])

# Our FMS 2D
results_our_fms_2d = results_experiment[results_experiment[:, 0] == 6, :]
results_our_fms_2d = results_our_fms_2d[results_our_fms_2d[:, 2] == 1, :]

results_our_fms_2d_ig_128 = results_our_fms_2d[results_our_fms_2d[:, 1] == 128, :]
result_table.loc[15, 'voxelSize'] = 128
result_table.loc[15, 'meanL2'] = np.mean(np.abs(results_our_fms_2d_ig_128[:, 3]))
result_table.loc[15, 'stdL2'] = np.std(results_our_fms_2d_ig_128[:, 3])
result_table.loc[15, 'meanAngle'] = np.mean(np.abs(results_our_fms_2d_ig_128[:, 4]))
result_table.loc[15, 'stdAngle'] = np.std(results_our_fms_2d_ig_128[:, 4])
result_table.loc[15, 'meanComputationTime'] = np.mean(np.abs(results_our_fms_2d_ig_128[:, 5]))
result_table.loc[15, 'stdComputationTime'] = np.std(results_our_fms_2d_ig_128[:, 5])

results_our_fms_2d_ig_256 = results_our_fms_2d[results_our_fms_2d[:, 1] == 256, :]
result_table.loc[16, 'voxelSize'] = 256
result_table.loc[16, 'meanL2'] = np.mean(np.abs(results_our_fms_2d_ig_256[:, 3]))
result_table.loc[16, 'stdL2'] = np.std(results_our_fms_2d_ig_256[:, 3])
result_table.loc[16, 'meanAngle'] = np.mean(np.abs(results_our_fms_2d_ig_256[:, 4]))
result_table.loc[16, 'stdAngle'] = np.std(results_our_fms_2d_ig_256[:, 4])
result_table.loc[16, 'meanComputationTime'] = np.mean(np.abs(results_our_fms_2d_ig_256[:, 5]))
result_table.loc[16, 'stdComputationTime'] = np.std(results_our_fms_2d_ig_256[:, 5])

results_our_fms_2d_ig_512 = results_our_fms_2d[results_our_fms_2d[:, 1] == 512, :]
result_table.loc[17, 'voxelSize'] = 512
result_table.loc[17, 'meanL2'] = np.mean(np.abs(results_our_fms_2d_ig_512[:, 3]))
result_table.loc[17, 'stdL2'] = np.std(results_our_fms_2d_ig_512[:, 3])
result_table.loc[17, 'meanAngle'] = np.mean(np.abs(results_our_fms_2d_ig_512[:, 4]))
result_table.loc[17, 'stdAngle'] = np.std(results_our_fms_2d_ig_512[:, 4])
result_table.loc[17, 'meanComputationTime'] = np.mean(np.abs(results_our_fms_2d_ig_512[:, 5]))
result_table.loc[17, 'stdComputationTime'] = np.std(results_our_fms_2d_ig_512[:, 5])

# Our FMS 2D global
results_our_fms_2d = results_experiment[results_experiment[:, 0] == 6, :]
results_our_fms_2d = results_our_fms_2d[results_our_fms_2d[:, 2] == 0, :]

results_our_fms_2d_gg_128 = results_our_fms_2d[results_our_fms_2d[:, 1] == 128, :]
result_table.loc[18, 'voxelSize'] = 128
result_table.loc[18, 'meanL2'] = np.mean(np.abs(results_our_fms_2d_gg_128[:, 3]))
result_table.loc[18, 'stdL2'] = np.std(results_our_fms_2d_gg_128[:, 3])
result_table.loc[18, 'meanAngle'] = np.mean(np.abs(results_our_fms_2d_gg_128[:, 4]))
result_table.loc[18, 'stdAngle'] = np.std(results_our_fms_2d_gg_128[:, 4])
result_table.loc[18, 'meanComputationTime'] = np.mean(np.abs(results_our_fms_2d_gg_128[:, 5]))
result_table.loc[18, 'stdComputationTime'] = np.std(results_our_fms_2d_gg_128[:, 5])

results_our_fms_2d_gg_256 = results_our_fms_2d[results_our_fms_2d[:, 1] == 256, :]
result_table.loc[19, 'voxelSize'] = 256
result_table.loc[19, 'meanL2'] = np.mean(np.abs(results_our_fms_2d_gg_256[:, 3]))
result_table.loc[19, 'stdL2'] = np.std(results_our_fms_2d_gg_256[:, 3])
result_table.loc[19, 'meanAngle'] = np.mean(np.abs(results_our_fms_2d_gg_256[:, 4]))
result_table.loc[19, 'stdAngle'] = np.std(results_our_fms_2d_gg_256[:, 4])
result_table.loc[19, 'meanComputationTime'] = np.mean(np.abs(results_our_fms_2d_gg_256[:, 5]))
result_table.loc[19, 'stdComputationTime'] = np.std(results_our_fms_2d_gg_256[:, 5])

results_our_fms_2d_gg_512 = results_our_fms_2d[results_our_fms_2d[:, 1] == 512, :]
result_table.loc[20, 'voxelSize'] = 512
result_table.loc[20, 'meanL2'] = np.mean(np.abs(results_our_fms_2d_gg_512[:, 3]))
result_table.loc[20, 'stdL2'] = np.std(results_our_fms_2d_gg_512[:, 3])
result_table.loc[20, 'meanAngle'] = np.mean(np.abs(results_our_fms_2d_gg_512[:, 4]))
result_table.loc[20, 'stdAngle'] = np.std(results_our_fms_2d_gg_512[:, 4])
result_table.loc[20, 'meanComputationTime'] = np.mean(np.abs(results_our_fms_2d_gg_512[:, 5]))
result_table.loc[20, 'stdComputationTime'] = np.std(results_our_fms_2d_gg_512[:, 5])

# Boxplots
plt.figure(1)
plt.clf()
l2_data = [results_gicp_128[:, 3], results_gicp_256[:, 3], results_gicp_512[:, 3],
           results_ndtd2d2d_128[:, 3], results_ndtd2d2d_256[:, 3], results_ndtd2d2d_512[:, 3],
           results_ndtp2d_128[:, 3], results_ndtp2d_256[:, 3], results_ndtp2d_512[:, 3],
           results_fourier_mellin_128[:, 3], results_fourier_mellin_256[:, 3], results_fourier_mellin_512[:, 3],
           results_our_fms_2d_ig_128[:, 3], results_our_fms_2d_ig_256[:, 3], results_our_fms_2d_ig_512[:, 3],
           results_our_fms_2d_gg_128[:, 3], results_our_fms_2d_gg_256[:, 3], results_our_fms_2d_gg_512[:, 3],
           results_initial_guess[:, 3]]
labels = ['GICP 128', 'GICP 256', 'GICP 512',
          'NDT D2D 2D 128', 'NDT D2D 2D 256', 'NDT D2D 2D 512',
          'NDT P2D 128', 'NDT P2D 256', 'NDT P2D 512',
          'FourierMellinTransform 128', 'FourierMellinTransform 256', 'FourierMellinTransform 512',
          'Our FMS 2D 128', 'Our FMS 2D 256', 'Our FMS 2D 512',
          'Our FMS 2D Global 128', 'Our FMS 2D Global 256', 'Our FMS 2D Global 512',
          'initial guess']
plt.boxplot(l2_data, labels=labels)
plt.yscale('log')
plt.title("L2 norm Error")

plt.figure(2)
plt.clf()
angle_data = [results_gicp_128[:, 4], results_gicp_256[:, 4], results_gicp_512[:, 4],
              results_ndtd2d2d_128[:, 4], results_ndtd2d2d_256[:, 4], results_ndtd2d2d_512[:, 4],
              results_ndtp2d_128[:, 4], results_ndtp2d_256[:, 4], results_ndtp2d_512[:, 4],
              results_fourier_mellin_128[:, 4], results_fourier_mellin_256[:, 4], results_fourier_mellin_512[:, 4],
              results_our_fms_2d_ig_128[:, 4], results_our_fms_2d_ig_256[:, 4], results_our_fms_2d_ig_512[:, 4],
              results_our_fms_2d_gg_128[:, 4], results_our_fms_2d_gg_256[:, 4], results_our_fms_2d_gg_512[:, 4],
              results_initial_guess[:, 4]]
plt.boxplot(angle_data, labels=labels)
plt.yscale('log')
plt.title("abs angle error")

plt.figure(3)
plt.clf()
time_data = [results_gicp_128[:, 5], results_gicp_256[:, 5], results_gicp_512[:, 5],
             results_ndtd2d2d_128[:, 5], results_ndtd2d2d_256[:, 5], results_ndtd2d2d_512[:, 5],
             results_ndtp2d_128[:, 5], results_ndtp2d_256[:, 5], results_ndtp2d_512[:, 5],
             results_fourier_mellin_128[:, 5], results_fourier_mellin_256[:, 5], results_fourier_mellin_512[:, 5],
             results_our_fms_2d_ig_128[:, 5], results_our_fms_2d_ig_256[:, 5], results_our_fms_2d_ig_512[:, 5],
             results_our_fms_2d_gg_128[:, 5], results_our_fms_2d_gg_256[:, 5], results_our_fms_2d_gg_512[:, 5],
             results_initial_guess[:, 5]]
plt.boxplot(time_data, labels=labels)
plt.yscale('log')
plt.title("computational time")

# Additional analysis
array_to_work_on = results_gicp_256
plt.plot(array_to_work_on[:, 6], array_to_work_on[:, 3], '.')
c = np.polyfit(array_to_work_on[:, 6], array_to_work_on[:, 3], 1)
y_est = np.polyval(c, array_to_work_on[:, 6])
plt.hold(True)
plt.plot(array_to_work_on[:, 6], y_est, 'r--', linewidth=2)
plt.hold(False)
