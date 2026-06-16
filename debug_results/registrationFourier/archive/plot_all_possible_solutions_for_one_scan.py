import numpy as np
import matplotlib.pyplot as plt
import subprocess
from skimage.transform import resize

# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

which_keyframe = 24
name_of_folder = '/home/tim-external/dataFolder/gazeboCorrectedUnevenAnglesPCLs_4_100/'
first_scan = f'pclKeyFrame{which_keyframe}.pcd'
second_scan = f'pclKeyFrame{which_keyframe + 1}.pcd'

magnitude_fftw1 = np.genfromtxt("csvFiles/magnitudeFFTW1.csv", delimiter=",")
N = int(np.sqrt(magnitude_fftw1.shape[0]))

phase_fftw1 = np.genfromtxt("csvFiles/phaseFFTW1.csv", delimiter=",")
voxel_data_used1 = np.genfromtxt("csvFiles/voxelDataFFTW1.csv", delimiter=",")

magnitude_fftw2 = np.genfromtxt("csvFiles/magnitudeFFTW2.csv", delimiter=",")
phase_fftw2 = np.genfromtxt("csvFiles/phaseFFTW2.csv", delimiter=",")
voxel_data_used2 = np.genfromtxt("csvFiles/voxelDataFFTW2.csv", delimiter=",")

magnitude1 = np.zeros((N, N))
phase1 = np.zeros((N, N))
voxel_data1 = np.zeros((N, N))
magnitude2 = np.zeros((N, N))
phase2 = np.zeros((N, N))
voxel_data2 = np.zeros((N, N))

for j in range(N):
    for k in range(N):
        magnitude1[k, j] = magnitude_fftw1[k * N - N + j]
        phase1[k, j] = phase_fftw1[k * N - N + j]
        voxel_data1[k, j] = voxel_data_used1[(k - 1) * N + j]
        magnitude2[k, j] = magnitude_fftw2[k * N - N + j]
        phase2[k, j] = phase_fftw2[k * N - N + j]
        voxel_data2[k, j] = voxel_data_used2[(k - 1) * N + j]

magnitude1 = np.fft.fftshift(magnitude1)
phase1 = np.fft.fftshift(phase1)
magnitude2 = np.fft.fftshift(magnitude2)
phase2 = np.fft.fftshift(phase2)

plt.figure(1)
plt.clf()
plt.imshow(magnitude1, aspect='equal')
plt.axis('equal')
plt.tight_layout()
plt.savefig('/home/tim-external/Documents/icra2023FMS/figures/magnitudeScan1.pdf')
subprocess.run(['pdfcrop', '/home/tim-external/Documents/icra2023FMS/figures/magnitudeScan1.pdf', '/home/tim-external/Documents/icra2023FMS/figures/magnitudeScan1.pdf'])

plt.figure(2)
plt.clf()
plt.imshow(voxel_data1, aspect='equal')
plt.axis('equal')
plt.tight_layout()
plt.savefig('/home/tim-external/Documents/icra2023FMS/figures/voxelData1.pdf')
subprocess.run(['pdfcrop', '/home/tim-external/Documents/icra2023FMS/figures/voxelData1.pdf', '/home/tim-external/Documents/icra2023FMS/figures/voxelData1.pdf'])

plt.figure(3)
plt.clf()
plt.imshow(magnitude2, aspect='equal')
plt.axis('equal')
plt.tight_layout()
plt.savefig('/home/tim-external/Documents/icra2023FMS/figures/magnitudeScan2.pdf')
subprocess.run(['pdfcrop', '/home/tim-external/Documents/icra2023FMS/figures/magnitudeScan2.pdf', '/home/tim-external/Documents/icra2023FMS/figures/magnitudeScan2.pdf'])

plt.figure(4)
plt.imshow(voxel_data2, aspect='equal')
plt.axis('equal')
plt.tight_layout()
plt.savefig('/home/tim-external/Documents/icra2023FMS/figures/voxelData2.pdf')
subprocess.run(['pdfcrop', '/home/tim-external/Documents/icra2023FMS/figures/voxelData2.pdf', '/home/tim-external/Documents/icra2023FMS/figures/voxelData2.pdf'])

resampled_data_for_sphere1 = np.genfromtxt("csvFiles/resampledVoxel1.csv", delimiter=",")
resampled_data_for_sphere2 = np.genfromtxt("csvFiles/resampledVoxel2.csv", delimiter=",")

resampled_data_for_sphere_result1 = np.zeros((N, N))
resampled_data_for_sphere_result2 = np.zeros((N, N))
for j in range(1, N + 1):
    for i in range(1, N + 1):
        resampled_data_for_sphere_result1[j - 1, i - 1] = resampled_data_for_sphere1[(i - 1) * N + j]
        resampled_data_for_sphere_result2[j - 1, i - 1] = resampled_data_for_sphere2[(i - 1) * N + j]

if True:
    fig = plt.figure(5)
    ax = fig.add_subplot(111, projection='3d')
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, facecolors=plt.cm.gray(resampled_data_for_sphere_result2), edgecolor='none')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('/home/tim-external/Documents/icra2023FMS/figures/resampledDataForSphereResult2.pdf')
    subprocess.run(['pdfcrop', '/home/tim-external/Documents/icra2023FMS/figures/resampledDataForSphereResult2.pdf', '/home/tim-external/Documents/icra2023FMS/figures/resampledDataForSphereResult2.pdf'])

    fig = plt.figure(6)
    ax = fig.add_subplot(111, projection='3d')
    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, facecolors=plt.cm.gray(resampled_data_for_sphere_result1), edgecolor='none')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('/home/tim-external/Documents/icra2023FMS/figures/resampledDataForSphereResult1.pdf')
    subprocess.run(['pdfcrop', '/home/tim-external/Documents/icra2023FMS/figures/resampledDataForSphereResult1.pdf', '/home/tim-external/Documents/icra2023FMS/figures/resampledDataForSphereResult1.pdf'])
else:
    plt.figure(5)
    plt.clf()
    plt.imshow(resampled_data_for_sphere_result2, aspect='equal')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('/home/tim-external/Documents/icra2023FMS/figures/resampledDataForSphereResult2.pdf')
    subprocess.run(['pdfcrop', '/home/tim-external/Documents/icra2023FMS/figures/resampledDataForSphereResult2.pdf', '/home/tim-external/Documents/icra2023FMS/figures/resampledDataForSphereResult2.pdf'])

    plt.figure(6)
    plt.clf()
    plt.imshow(resampled_data_for_sphere_result1, aspect='equal')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('/home/tim-external/Documents/icra2023FMS/figures/resampledDataForSphereResult1.pdf')
    subprocess.run(['pdfcrop', '/home/tim-external/Documents/icra2023FMS/figures/resampledDataForSphereResult1.pdf', '/home/tim-external/Documents/icra2023FMS/figures/resampledDataForSphereResult1.pdf'])

plt.figure(7)
correlation_of_angles = np.genfromtxt("csvFiles/resultingCorrelation1D.csv", delimiter=",")

angles_x = np.linspace(0, 2 * np.pi, len(correlation_of_angles))
peaks, peaks_loc = scipy.signal.find_peaks(correlation_of_angles)
plt.plot(angles_x, correlation_of_angles)
for i in range(len(peaks)):
    plt.text(angles_x[peaks_loc[i]] + 0.15, peaks[i] + 1, f'{round(angles_x[peaks_loc[i]], 2)}')
plt.grid(True)
plt.xlim([-0.2, 2 * np.pi + 0.2])
plt.xlabel("angle in rad")
plt.ylabel("Correlation Value")
plt.tight_layout()
plt.savefig('/home/tim-external/Documents/icra2023FMS/figures/resultingCorrelation1DAngle.pdf')
subprocess.run(['pdfcrop', '/home/tim-external/Documents/icra2023FMS/figures/resultingCorrelation1DAngle.pdf', '/home/tim-external/Documents/icra2023FMS/figures/resultingCorrelation1DAngle.pdf'])

data_information = np.genfromtxt("csvFiles/dataForReadIn.csv", delimiter=",")

number_of_solutions = int(data_information[0])
best_solution = int(data_information[1])

plt.figure(8)
plt.clf()

for i in range(1, number_of_solutions + 1):
    plt.subplot(2, 2, i)
    correlation_matrix_shift_1d = np.genfromtxt(f'csvFiles/resultingCorrelationShift_{i - 1}_.csv', delimiter=",")
    result_size = int(np.sqrt(len(correlation_matrix_shift_1d)))
    correlation_matrix_shift_2d = correlation_matrix_shift_1d.reshape((result_size, result_size))
    plt.imshow(correlation_matrix_shift_2d)
    plt.title(str(i))
    plt.axis('equal')

plt.tight_layout()
plt.savefig('/home/tim-external/Documents/icra2023FMS/figures/2dCorrelation.pdf')
subprocess.run(['pdfcrop', '/home/tim-external/Documents/icra2023FMS/figures/2dCorrelation.pdf', '/home/tim-external/Documents/icra2023FMS/figures/2dCorrelation.pdf'])

f = plt.figure(9)
plt.clf()

for i in range(1, number_of_solutions + 1):
    result_voxel1_tmp = np.genfromtxt(f"csvFiles/resultVoxel1{i - 1}.csv", delimiter=",")
    result_voxel2_tmp = np.genfromtxt(f"csvFiles/resultVoxel2{i - 1}.csv", delimiter=",")

    voxel_result1 = np.zeros((N, N))
    voxel_result2 = np.zeros((N, N))
    for j in range(N):
        for k in range(N):
            voxel_result1[k, j] = result_voxel1_tmp[k * N - N + j]
            voxel_result2[k, j] = result_voxel2_tmp[k * N - N + j]

    plt.subplot(2, 2, i)
    blended = 0.5 * voxel_result1 + 0.5 * voxel_result2
    plt.imshow(blended)
    plt.title(str(i))

plt.tight_layout()
f.set_size_inches(5, 5)
plt.savefig('/home/tim-external/Documents/icra2023FMS/figures/matchedPointcloudsAll.pdf')
subprocess.run(['pdfcrop', '/home/tim-external/Documents/icra2023FMS/figures/matchedPointcloudsAll.pdf', '/home/tim-external/Documents/icra2023FMS/figures/matchedPointcloudsAll.pdf'])