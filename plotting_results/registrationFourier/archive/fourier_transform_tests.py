import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks


def plot_ffts(voxel_data, figure_number):
    plt.figure(figure_number)
    plt.subplot(1, 3, 1)

    plt.imshow(voxel_data[:, :, voxel_data.shape[2] // 2], aspect='equal')
    plt.title('Voxel: ' + str(figure_number))
    plt.axis('image')

    fft_output = np.fft.fftshift(np.fft.fftn(voxel_data))

    plt.subplot(1, 3, 2)
    magnitude = np.abs(fft_output)
    plt.imshow(magnitude[:, :, magnitude.shape[2] // 2], aspect='equal')
    plt.title('Magnitude Voxel: ' + str(figure_number))
    plt.axis('image')

    imaginary_part = np.imag(fft_output)
    real_part = np.real(fft_output)
    phase = np.arctan2(imaginary_part, real_part)
    plt.subplot(1, 3, 3)
    plt.imshow(phase[:, :, phase.shape[2] // 2], aspect='equal')
    plt.title('Phase Voxel: ' + str(figure_number))
    plt.axis('image')

    return fft_output, magnitude, phase


# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

point_cloud = o3d.io.read_point_cloud("after_voxel_second.pcd")

from_to = 30
number_of_points = 64
voxel_data1 = np.zeros((number_of_points, number_of_points, number_of_points))

for j in range(len(point_cloud.points)):
    x_pos = point_cloud.points[j, 0]
    y_pos = point_cloud.points[j, 1]
    x_index = int((x_pos + from_to) / (from_to * 2) * number_of_points)
    y_index = int((y_pos + from_to) / (from_to * 2) * number_of_points)
    z_index = int((0 + from_to) / (from_to * 2) * number_of_points)
    voxel_data1[x_index, y_index, z_index] = 1

spectrum1, magnitude1, phase1 = plot_ffts(voxel_data1, 1)

voxel_data2 = np.zeros((number_of_points, number_of_points, number_of_points))
shift_first = np.array([-5, 10])

for j in range(len(point_cloud.points)):
    x_pos = point_cloud.points[j, 0] + shift_first[0]
    y_pos = point_cloud.points[j, 1] + shift_first[1]
    x_index = int((x_pos + from_to) / (from_to * 2) * number_of_points)
    y_index = int((y_pos + from_to) / (from_to * 2) * number_of_points)
    z_index = int((0 + from_to) / (from_to * 2) * number_of_points)
    voxel_data2[x_index, y_index, z_index] = 1

shift_second = np.array([-0, 0])
for j in range(len(point_cloud.points)):
    x_pos = point_cloud.points[j, 0] + shift_second[0]
    y_pos = point_cloud.points[j, 1] + shift_second[1]
    x_index = int((x_pos + from_to) / (from_to * 2) * number_of_points)
    y_index = int((y_pos + from_to) / (from_to * 2) * number_of_points)
    z_index = int((0 + from_to) / (from_to * 2) * number_of_points)
    voxel_data2[x_index, y_index] = 1

voxel_data2 = voxel_data2 + 0.000 * np.random.randn(*voxel_data2.shape)
spectrum2, magnitude2, phase2 = plot_ffts(voxel_data2, 2)

resulting_phase_difference = phase1 - phase2

inverse_fft_for_phase = np.fft.fftshift(np.fft.ifft2(np.exp(-1j * resulting_phase_difference)))

imaginary_part = np.imag(inverse_fft_for_phase)
real_part = np.real(inverse_fft_for_phase)
phase = np.arctan2(imaginary_part, real_part)

magnitude = np.abs(inverse_fft_for_phase)
magnitude = magnitude[:, :, magnitude.shape[0] // 2]

fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')
x_plot, y_plot = np.meshgrid(np.arange(1, number_of_points + 1), np.arange(1, number_of_points + 1))
ax.plot_surface(x_plot, y_plot, magnitude)
plt.title('magnitude of invFFT(arg(R)-arg(S))')

tf1, p = find_peaks(magnitude)

fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_plot, y_plot, magnitude)
plt.title('magnitude of invFFT(arg(R)-arg(S))')

p_reshaped = p.reshape(1, -1)

peaks_of_shift, i = np.sort(p_reshaped[0])[::-1], np.argsort(-p_reshaped[0])

diff_peaks = np.abs(np.diff(peaks_of_shift))
tf1, pos = np.max(diff_peaks), np.argmax(diff_peaks)

index_p_col = np.zeros(pos)
index_p_row = np.zeros(pos)
position = np.zeros((pos, 2))

for j in range(pos):
    index_p_col[j] = np.ceil(i[j] / number_of_points)
    index_p_row[j] = i[j] - number_of_points * (index_p_col[j] - 1)
    position[j, 0:2] = [index_p_row[j] - number_of_points / 2, index_p_col[j] - number_of_points / 2]

translation_shift1 = -position[0, :] / number_of_points * from_to * 2
translation_shift2 = -position[1, :] / number_of_points * from_to * 2

print(shift_second)
print(shift_first)
