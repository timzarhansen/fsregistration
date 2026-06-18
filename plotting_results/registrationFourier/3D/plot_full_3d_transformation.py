import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def volume_viewer(volume_3d):
    # Simple 3D volume visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Show a few slices
    mid_slice = volume_3d.shape[0] // 2
    x, y = np.meshgrid(np.arange(volume_3d.shape[1]), np.arange(volume_3d.shape[2]))
    ax.plot_surface(x, y, volume_3d[mid_slice, :, :], cmap='viridis')
    plt.show()


which_rotation = 0

magnitude_fftw1 = np.genfromtxt("csvFiles/magnitudeFFTW1.csv", delimiter=",")
phase_fftw1 = np.genfromtxt("csvFiles/phaseFFTW1.csv", delimiter=",")
voxel_data_used1 = np.genfromtxt("csvFiles/voxelDataFFTW1.csv", delimiter=",")

magnitude_fftw2 = np.genfromtxt("csvFiles/magnitudeFFTW2.csv", delimiter=",")
phase_fftw2 = np.genfromtxt("csvFiles/phaseFFTW2.csv", delimiter=",")
voxel_data_used2 = np.genfromtxt("csvFiles/voxelDataFFTW2.csv", delimiter=",")

magnitude_fftw2_rotated = np.genfromtxt(f"csvFiles/magnitudeFFTW2Rotated{which_rotation}.csv", delimiter=",")
phase_fftw2_rotated = np.genfromtxt(f"csvFiles/phaseFFTW2Rotated{which_rotation}.csv", delimiter=",")
voxel_data_used2_rotated = np.genfromtxt(f"csvFiles/voxelDataFFTW2Rotated{which_rotation}.csv", delimiter=",")

N = int(np.cbrt(voxel_data_used1.shape[0]))
n_correlation = N * 2 - 1
magnitude1 = np.zeros((N, N, N))
phase1 = np.zeros((N, N, N))
voxel_data1 = np.zeros((N, N, N))
magnitude2 = np.zeros((N, N, N))
phase2 = np.zeros((N, N, N))
voxel_data2 = np.zeros((N, N, N))
magnitude2_rotated = np.zeros((N, N, N))
phase2_rotated = np.zeros((N, N, N))
voxel_data2_rotated = np.zeros((N, N, N))

for j in range(N):
    for i in range(N):
        for k in range(N):
            index = (i - 1) * N * N + (j - 1) * N + k - 1
            magnitude1[i, j, k] = magnitude_fftw1[index]
            phase1[i, j, k] = phase_fftw1[index]
            voxel_data1[i, j, k] = voxel_data_used1[index]
            magnitude2[i, j, k] = magnitude_fftw2[index]
            phase2[i, j, k] = phase_fftw2[index]
            voxel_data2[i, j, k] = voxel_data_used2[index]
            magnitude2_rotated[i, j, k] = magnitude_fftw2_rotated[index]
            phase2_rotated[i, j, k] = phase_fftw2_rotated[index]
            voxel_data2_rotated[i, j, k] = voxel_data_used2_rotated[index]

resampled_magnitude1 = np.genfromtxt("csvFiles/resampledMagnitudeSO3_1.csv", delimiter=",")
resampled_magnitude2 = np.genfromtxt("csvFiles/resampledMagnitudeSO3_2.csv", delimiter=",")

resampled_magnitude1_matrix = np.zeros((N, N))
resampled_magnitude2_matrix = np.zeros((N, N))

for j in range(N):
    for i in range(N):
        resampled_magnitude1_matrix[i, j] = resampled_magnitude1[(i - 1) * N + j - 1]
        resampled_magnitude2_matrix[i, j] = resampled_magnitude2[(i - 1) * N + j - 1]

plt.figure(1)
plt.imshow(resampled_magnitude1_matrix, aspect='equal')
plt.figure(2)
plt.imshow(resampled_magnitude2_matrix, aspect='equal')

plt.figure(3)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(resampled_magnitude1_matrix.flatten() / resampled_magnitude1_matrix.max()), edgecolor='none')
plt.axis('equal')

plt.figure(4)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(resampled_magnitude2_matrix.flatten() / resampled_magnitude2_matrix.max()), edgecolor='none')
plt.axis('equal')

correlation_values_real = np.genfromtxt("csvFiles/resultingCorrelationReal.csv", delimiter=",")
correlation_values_complex = np.genfromtxt("csvFiles/resultingCorrelationComplex.csv", delimiter=",")

correlation_values_real_matrix = np.zeros((N, N, N))
correlation_values_complex_matrix = np.zeros((N, N, N))
correlation_values_magnitude = np.zeros((N, N, N))

for j in range(N):
    for i in range(N):
        for k in range(N):
            index = (i - 1) * N + j + (k - 1) * N * N - 1
            correlation_values_real_matrix[i, j, k] = correlation_values_real[index]
            correlation_values_complex_matrix[i, j, k] = correlation_values_complex[index]
            correlation_values_magnitude[i, j, k] = np.sqrt(correlation_values_real_matrix[i, j, k] ** 2 + correlation_values_complex_matrix[i, j, k] ** 2)

volume_viewer(correlation_values_magnitude)

# correlation 3D
translation_correlation = np.genfromtxt(f"csvFiles/resultingCorrelationShift{which_rotation}.csv", delimiter=",")
N = int(np.cbrt(translation_correlation.shape[0]))
translation_correlation_3d = np.zeros((N, N, N))

for j in range(N):
    for i in range(N):
        for k in range(N):
            index = (i - 1) * N * N + (j - 1) * N + k - 1
            translation_correlation_3d[i, j, k] = translation_correlation[index]

volume_viewer(translation_correlation_3d)

c = np.max(translation_correlation_3d.flatten())
i = np.argmax(translation_correlation_3d.flatten())
print(c)
print(translation_correlation_3d.flatten()[i])
i1, i2, i3 = np.unravel_index(i, translation_correlation_3d.shape)
print(i1, i2, i3)
