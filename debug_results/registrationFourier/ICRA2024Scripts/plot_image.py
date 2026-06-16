import numpy as np
import matplotlib.pyplot as plt

# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

number = 1

voxel_data_1_raw = np.genfromtxt(f"dataFolder/testFolder/correctedScan{number}.csv", delimiter=",")
voxel_data_2_raw = np.genfromtxt(f"dataFolder/testFolder/uncorrectedScan{number}.csv", delimiter=",")

plt.figure(1)
plt.imshow(voxel_data_1_raw, aspect='equal')

plt.figure(2)
plt.imshow(voxel_data_2_raw, aspect='equal')

# Clear workspace
# clc, clear - skipped

N = 200

input_example = np.zeros((N, N))

p = slice(97, 103)
k = 5
input_example[p, p] = np.ones((k, k))

result_fft = np.fft.fftshift(np.fft.fft2(input_example, N, N))

plt.imshow(np.angle(result_fft), aspect='equal')
