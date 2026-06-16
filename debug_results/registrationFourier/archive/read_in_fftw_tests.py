import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import correlate2d
from mpl_toolkits.mplot3d import Axes3D

# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

magnitude_fftw1 = np.genfromtxt("csvFiles/magnitudeFFTW1.csv", delimiter=",")
n = int(np.sqrt(magnitude_fftw1.shape[0]))

phase_fftw1 = np.genfromtxt("csvFiles/phaseFFTW1.csv", delimiter=",")
voxel_data_used1 = np.genfromtxt("csvFiles/voxelDataFFTW1.csv", delimiter=",")

magnitude_fftw2 = np.genfromtxt("csvFiles/magnitudeFFTW2.csv", delimiter=",")
phase_fftw2 = np.genfromtxt("csvFiles/phaseFFTW2.csv", delimiter=",")
voxel_data_used2 = np.genfromtxt("csvFiles/voxelDataFFTW2.csv", delimiter=",")

magnitude1 = np.zeros((n, n))
phase1 = np.zeros((n, n))
voxel_data1 = np.zeros((n, n))
magnitude2 = np.zeros((n, n))
phase2 = np.zeros((n, n))
voxel_data2 = np.zeros((n, n))

for j in range(1, n + 1):
    for i in range(1, n + 1):
        magnitude1[i - 1, j - 1] = magnitude_fftw1[(i - 1) * n + j - 1]
        phase1[i - 1, j - 1] = phase_fftw1[(i - 1) * n + j - 1]
        voxel_data1[i - 1, j - 1] = voxel_data_used1[(i - 1) * n + j - 1]
        magnitude2[i - 1, j - 1] = magnitude_fftw2[(i - 1) * n + j - 1]
        phase2[i - 1, j - 1] = phase_fftw2[(i - 1) * n + j - 1]
        voxel_data2[i - 1, j - 1] = voxel_data_used2[(i - 1) * n + j - 1]

plt.figure(1)
plt.clf()
plt.subplot(1, 2, 2)
plt.imshow(magnitude1)
plt.title('Magnitude Voxel: ' + str(1))
plt.axis('image')

plt.subplot(1, 2, 1)
plt.imshow(voxel_data1)
plt.axis('image')

plt.figure(2)
plt.clf()
plt.subplot(1, 2, 2)
plt.imshow(magnitude2)
plt.title('Magnitude Voxel: ' + str(1))
plt.axis('image')

plt.subplot(1, 2, 1)
plt.imshow(voxel_data2)
plt.axis('image')

results = np.genfromtxt("csvFiles/resultCorrelation3D.csv", delimiter=",")

results = results / np.max(results)
result_size = int(np.sqrt(np.sqrt(len(results))))

a = results.reshape(result_size, result_size, result_size)

correlation_number_matrix = a[:, :, 0]

correlation_of_angles = np.genfromtxt("csvFiles/resultingCorrelation1D.csv", delimiter=",")

x_for_plot = np.arange(len(correlation_of_angles))
x_for_plot = x_for_plot / len(correlation_of_angles) * 360
plt.figure(6)
plt.clf()
plt.plot(x_for_plot, correlation_of_angles)

plt.figure(8)
plt.clf()

correlation_matrix_shift_1d = np.genfromtxt("csvFiles/resultingCorrelationShift.csv", delimiter=",")
result_size = int(np.sqrt(len(correlation_matrix_shift_1d)))
correlation_matrix_shift_2d = np.zeros((result_size, result_size))

for j in range(1, result_size + 1):
    for i in range(1, result_size + 1):
        correlation_matrix_shift_2d[i - 1, j - 1] = correlation_matrix_shift_1d[(i - 1) * result_size + j - 1]

test_image = correlation_matrix_shift_2d

x_plot, y_plot = np.meshgrid(np.arange(1, result_size + 1), np.arange(1, result_size + 1))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_plot, y_plot, test_image, edgecolor='none')
plt.xlabel("x-axis")
plt.ylabel("y-axis")

voxel_data1_trans_padding = voxel_data1
voxel_data2_trans_padding = voxel_data2

padding_size = 0
voxel_data1_trans_padding = np.pad(voxel_data1_trans_padding, ((padding_size, padding_size), (padding_size, padding_size)), 'constant')
voxel_data2_trans_padding = np.pad(voxel_data2_trans_padding, ((padding_size, padding_size), (padding_size, padding_size)), 'constant')

spectrum1_trans = np.fft.fftn(voxel_data1_trans_padding)
spectrum2_trans = np.fft.fftn(voxel_data2_trans_padding)

inverse_fft_for_phase = np.fft.fftshift(np.fft.ifft2(128 * 128 * spectrum1_trans * np.conj(spectrum2_trans)))

magnitude_translation = np.abs(inverse_fft_for_phase)

m, i = np.unravel_index(np.argmax(magnitude_translation), magnitude_translation.shape)
y_axis_index, x_axis_index = m, i

y = magnitude_translation[y_axis_index, :].flatten()
shift_xy = (x_axis_index - 1) * 0.6
x = np.linspace(0, 0.6 * (n - 1), n)

voxel_result1_fftw = np.genfromtxt("csvFiles/resultVoxel1.csv", delimiter=",")
n = int(np.sqrt(voxel_result1_fftw.shape[0]))

voxel_result2_fftw = np.genfromtxt("csvFiles/resultVoxel2.csv", delimiter=",")

voxel_result1 = np.zeros((n, n))
voxel_result2 = np.zeros((n, n))

for j in range(1, n + 1):
    for k in range(1, n + 1):
        voxel_result1[k - 1, j - 1] = voxel_result1_fftw[k * n - n + j - 1]
        voxel_result2[k - 1, j - 1] = voxel_result2_fftw[k * n - n + j - 1]

plt.figure(3)
plt.clf()

plt.subplot(1, 2, 1)
plt.imshow(voxel_result1)
plt.title('Voxel 1: ')
plt.axis('image')

plt.subplot(1, 2, 2)
alpha = 0.5
c = alpha * voxel_result1 + (1 - alpha) * voxel_result2
plt.imshow(c)
plt.title('Voxel 2: ')
plt.axis('image')

plt.figure(9)
a_fft = np.fft.fft2(voxel_data1)
plt.imshow(np.abs(a_fft))

b_fft = np.fft.fft2(voxel_data2)

c_new = np.fft.fftshift(np.fft.ifft2(a_fft * np.conj(b_fft)))

x_plot, y_plot = np.meshgrid(np.arange(1, c_new.shape[0] + 1), np.arange(1, c_new.shape[1] + 1))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_plot, y_plot, c_new, edgecolor='none')
plt.xlabel("x-axis")
plt.ylabel("y-axis")
