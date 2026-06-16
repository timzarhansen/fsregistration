import numpy as np
import matplotlib.pyplot as plt
import subprocess
import open3d as o3d
from scipy.signal import find_peaks
from rotation_matrix import rotation_matrix
from mpl_toolkits.mplot3d import Axes3D

# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

which_keyframe = 182

name_of_folder = '/home/tim-external/dataFolder/newStPereDatasetCorrectionOnly/'
first_scan = f'pclKeyFrame{which_keyframe}.pcd'
second_scan = f'pclKeyFrame{which_keyframe + 1}.pcd'

command = f'rosrun underwaterslam registrationOfTwoPCLs {name_of_folder}{first_scan} {name_of_folder}{second_scan}'
subprocess.call(command, shell=True)

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
    for k in range(1, n + 1):
        magnitude1[j - 1, k - 1] = magnitude_fftw1[j * n - n + k - 1]
        phase1[j - 1, k - 1] = phase_fftw1[j * n - n + k - 1]
        voxel_data1[j - 1, k - 1] = voxel_data_used1[(j - 1) * n + k - 1]
        magnitude2[j - 1, k - 1] = magnitude_fftw2[j * n - n + k - 1]
        phase2[j - 1, k - 1] = phase_fftw2[j * n - n + k - 1]
        voxel_data2[j - 1, k - 1] = voxel_data_used2[(j - 1) * n + k - 1]

magnitude1 = np.fft.fftshift(magnitude1)
phase1 = np.fft.fftshift(phase1)
magnitude2 = np.fft.fftshift(magnitude2)
phase2 = np.fft.fftshift(phase2)

plt.figure(1)
plt.clf()
plt.imshow(magnitude1)
plt.axis('image')
plt.box(True)
plt.savefig('outputPDFs/magnitudeScan1.pdf', bbox_inches='tight')
subprocess.call('pdfcrop outputPDFs/magnitudeScan1.pdf outputPDFs/magnitudeScan1.pdf', shell=True)

plt.figure(2)
plt.clf()
plt.imshow(voxel_data1)
plt.axis('image')
plt.savefig('outputPDFs/voxelData1.pdf', bbox_inches='tight')
subprocess.call('pdfcrop outputPDFs/voxelData1.pdf outputPDFs/voxelData1.pdf', shell=True)

plt.figure(3)
plt.imshow(magnitude2)
plt.axis('image')
plt.savefig('outputPDFs/magnitudeScan2.pdf', bbox_inches='tight')
subprocess.call('pdfcrop outputPDFs/magnitudeScan2.pdf outputPDFs/magnitudeScan2.pdf', shell=True)

plt.figure(4)
plt.imshow(voxel_data2)
plt.axis('image')
plt.savefig('outputPDFs/voxelData2.pdf', bbox_inches='tight')
subprocess.call('pdfcrop outputPDFs/voxelData2.pdf outputPDFs/voxelData2.pdf', shell=True)

resampled_data_for_sphere1 = np.genfromtxt("csvFiles/resampledVoxel1.csv", delimiter=",")
resampled_data_for_sphere2 = np.genfromtxt("csvFiles/resampledVoxel2.csv", delimiter=",")

resampled_data_for_sphere_result1 = np.zeros((n, n))
resampled_data_for_sphere_result2 = np.zeros((n, n))
for i in range(1, n + 1):
    for j in range(1, n + 1):
        resampled_data_for_sphere_result1[i - 1, j - 1] = resampled_data_for_sphere1[(i - 1) * n + j - 1]
        resampled_data_for_sphere_result2[i - 1, j - 1] = resampled_data_for_sphere2[(i - 1) * n + j - 1]

fig = plt.figure(5)
ax = fig.add_subplot(111, projection='3d')
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(resampled_data_for_sphere_result2 / resampled_data_for_sphere_result2.max()), shade=False)
plt.savefig('outputPDFs/resampledDataForSphereResult2.pdf', bbox_inches='tight')
subprocess.call('pdfcrop outputPDFs/resampledDataForSphereResult2.pdf outputPDFs/resampledDataForSphereResult2.pdf', shell=True)

fig = plt.figure(6)
ax = fig.add_subplot(111, projection='3d')
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(resampled_data_for_sphere_result1 / resampled_data_for_sphere_result1.max()), shade=False)
plt.savefig('outputPDFs/resampledDataForSphereResult1.pdf', bbox_inches='tight')
subprocess.call('pdfcrop outputPDFs/resampledDataForSphereResult1.pdf outputPDFs/resampledDataForSphereResult1.pdf', shell=True)

plt.figure(7)
correlation_of_angles = np.genfromtxt("csvFiles/resultingCorrelation1D.csv", delimiter=",")

angles_x = np.linspace(0, 2 * np.pi, len(correlation_of_angles))
psor, lsor = find_peaks(correlation_of_angles)
plt.plot(angles_x, correlation_of_angles)
for i in range(len(psor)):
    plt.text(angles_x[lsor[i]] + 0.15, psor[i] + 1, f'{round(angles_x[lsor[i]], 2)}')
plt.grid(True)
plt.xlim([-0.2, 2 * np.pi + 0.2])
plt.ylim([np.min(correlation_of_angles) - 10, np.max(correlation_of_angles) + 10])
plt.xlabel("angle in rad")
plt.ylabel("Correlation Value")
plt.savefig('outputPDFs/resultingCorrelation1DAngle.pdf', bbox_inches='tight')
subprocess.call('pdfcrop outputPDFs/resultingCorrelation1DAngle.pdf outputPDFs/resultingCorrelation1DAngle.pdf', shell=True)

data_information = np.genfromtxt("csvFiles/dataForReadIn.csv", delimiter=",")

number_of_solutions = int(data_information[0])
best_solution = int(data_information[1])

plt.figure(8)
plt.clf()

for i in range(1, number_of_solutions + 1):
    plt.subplot(2, 2, i)
    correlation_matrix_shift_1d = np.genfromtxt(f'csvFiles/resultingCorrelationShift{i - 1}.csv', delimiter=",")
    result_size = int(np.sqrt(len(correlation_matrix_shift_1d)))
    correlation_matrix_shift_2d = correlation_matrix_shift_1d.reshape(result_size, result_size)
    plt.imshow(correlation_matrix_shift_2d)
    plt.box(True)
    plt.title(str(i))

plt.savefig('outputPDFs/2dCorrelation.pdf', bbox_inches='tight')
subprocess.call('pdfcrop outputPDFs/2dCorrelation.pdf outputPDFs/2dCorrelation.pdf', shell=True)

plt.figure(9)
plt.clf()

for i in range(1, number_of_solutions + 1):
    plt.subplot(2, 2, i)
    point_cloud_result1 = o3d.io.read_point_cloud(f'csvFiles/resulting{i - 1}PCL1.pcd')
    point_cloud_result2 = o3d.io.read_point_cloud(f'csvFiles/resulting{i - 1}PCL2.pcd')
    
    points1 = np.array(point_cloud_result1.points)
    points2 = np.array(point_cloud_result2.points)
    
    plt.scatter(points1[:, 0], points1[:, 1], c='blue', marker='.', s=1)
    plt.scatter(points2[:, 0], points2[:, 1], c='red', marker='.', s=1)
    plt.axis('equal')
    plt.grid(True)
    plt.title(str(i))
    plt.box(True)

plt.savefig('outputPDFs/matchedPointcloudsAll.pdf', bbox_inches='tight')
subprocess.call('pdfcrop outputPDFs/matchedPointcloudsAll.pdf outputPDFs/matchedPointcloudsAll.pdf', shell=True)

point_cloud_result1 = o3d.io.read_point_cloud(f'csvFiles/resulting{best_solution}PCL1.pcd')
point_cloud_result2 = o3d.io.read_point_cloud(f'csvFiles/resulting{best_solution}PCL2.pcd')

points1 = np.array(point_cloud_result1.points)
points2 = np.array(point_cloud_result2.points)

rot = rotation_matrix(0, 0, np.pi / 2)
points1_rotated = points1 @ rot.T
points2_rotated = points2 @ rot.T

plt.figure(10)
plt.clf()
plt.scatter(points2_rotated[:, 1], points2_rotated[:, 0], c='blue', marker='.', s=1)
plt.scatter(points1_rotated[:, 1], points1_rotated[:, 0], c='red', marker='.', s=1)
plt.grid(True)
plt.box(True)
plt.xlabel('Y-Axis in m')
plt.ylabel('X-Axis in m')
plt.savefig('outputPDFs/matchedPointclouds.pdf', bbox_inches='tight')
subprocess.call('pdfcrop outputPDFs/matchedPointclouds.pdf outputPDFs/matchedPointclouds.pdf', shell=True)

plt.figure(11)
plt.clf()
correlation_matrix_shift_1d = np.genfromtxt(f'csvFiles/resultingCorrelationShift{best_solution}.csv', delimiter=",")
result_size = int(np.sqrt(len(correlation_matrix_shift_1d)))
correlation_matrix_shift_2d = correlation_matrix_shift_1d.reshape(result_size, result_size)
plt.imshow(correlation_matrix_shift_2d)
plt.box(True)
plt.savefig('outputPDFs/2dCorrelation.pdf', bbox_inches='tight')
subprocess.call('pdfcrop outputPDFs/2dCorrelation.pdf outputPDFs/2dCorrelation.pdf', shell=True)
