import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, correlate
from scipy.signal import find_peaks
from skimage import io

# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

voxel_data_used_1 = np.genfromtxt("csvFiles/voxelDataFFTW1.csv", delimiter=",")
voxel_data_used_2 = np.genfromtxt("csvFiles/voxelDataFFTW2.csv", delimiter=",")
n = 256
voxel_data_1 = np.zeros((n, n))
voxel_data_2 = np.zeros((n, n))

for j in range(n):
    for k in range(n):
        voxel_data_1[k, j] = voxel_data_used_1[(k - 1) * n + j]
        voxel_data_2[k, j] = voxel_data_used_2[(k - 1) * n + j]

a = io.imread('csvFiles/testImage1.png')
b = io.imread('csvFiles/testImage2.png')

b = b[:, :, 2]
maximum_b = np.max(b)
b = maximum_b - b
a = a[:, :, 2]
maximum_a = np.max(a)
a = maximum_a - a

b = b.astype(float)
a = a.astype(float)

b = gaussian_filter(b, sigma=2)
a = gaussian_filter(a, sigma=2)

plt.figure(1)
plt.imshow(a)
plt.figure(2)
plt.imshow(b)

maximum_overall = max([np.max(a), np.max(b)])
normalized_a = a / maximum_overall
normalized_b = b / maximum_overall
plt.figure(3)
size_tmp = normalized_a.shape[0]
c_new = correlate(normalized_a, normalized_b)

x_plot, y_plot = np.meshgrid(np.arange(1, c_new.shape[0] + 1), np.arange(1, c_new.shape[1] + 1))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_plot, y_plot, c_new)
plt.xlabel("x-axis")
plt.ylabel("y-axis")
ax.view_init(0, 90)

plt.figure(4)
c_simple = correlate(np.ones_like(normalized_a), np.ones_like(normalized_b))

x_plot, y_plot = np.meshgrid(np.arange(1, c_simple.shape[0] + 1), np.arange(1, c_simple.shape[1] + 1))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_plot, y_plot, c_simple, edgecolor='none')
plt.xlabel("x-axis")
plt.ylabel("y-axis")

plt.figure(5)
plt.clf()

resulting_corrected_correlation = (1. / c_simple) * c_new
x_plot, y_plot = np.meshgrid(np.arange(1, resulting_corrected_correlation.shape[0] + 1), np.arange(1, resulting_corrected_correlation.shape[1] + 1))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_plot, y_plot, resulting_corrected_correlation)
plt.xlabel("x-axis")
plt.ylabel("y-axis")

simple_copy = resulting_corrected_correlation
p = fast_2d_peak_find(resulting_corrected_correlation)
plt.hold(True)

z_for_plot = np.zeros(len(p[0:2:2]))
for k in range(len(p[0:2:2])):
    z_for_plot[k] = resulting_corrected_correlation[p[2 * k + 1] - 1, p[2 * k] - 1]

ax.plot3D(p[0:2:2], p[1:2:2], z_for_plot, 'r+')
ax.view_init(0, 90)


def fast_2d_peak_find(data):
    peaks, _ = find_peaks(data.flatten())
    peak_positions = np.unravel_index(peaks, data.shape)
    p = np.column_stack((peak_positions[1], peak_positions[0])).flatten()
    return p
