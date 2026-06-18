import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import subprocess
from rotation_matrix import rotation_matrix

# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

for j in range(1, 230):
    which_keyframe = j
    name_of_folder = '/home/tim-external/dataFolder/gazeboCorrectedPCLs/'
    first_scan = f'pclKeyFrame{which_keyframe}.pcd'
    second_scan = f'pclKeyFrame{which_keyframe + 1}.pcd'

    command = f'rosrun underwaterslam registrationOfTwoPCLs {name_of_folder}{first_scan} {name_of_folder}{second_scan}'
    subprocess.run(command, shell=True)

    transformation1 = np.genfromtxt(f"{name_of_folder}position{which_keyframe}.csv", delimiter=",")
    transformation2 = np.genfromtxt(f"{name_of_folder}position{which_keyframe + 1}.csv", delimiter=",")
    resulting_translation_diff = transformation2[0:3] - transformation1[0:3]
    resulting_rotation_diff = transformation2[3:6] - transformation1[3:6]

    magnitude_fftw1 = np.genfromtxt("magnitudeFFTW1.csv", delimiter=",")
    n = int(np.sqrt(magnitude_fftw1.shape[0]))

    phase_fftw1 = np.genfromtxt("phaseFFTW1.csv", delimiter=",")
    voxel_data_used1 = np.genfromtxt("voxelDataFFTW1.csv", delimiter=",")

    magnitude_fftw2 = np.genfromtxt("magnitudeFFTW2.csv", delimiter=",")
    phase_fftw2 = np.genfromtxt("phaseFFTW2.csv", delimiter=",")
    voxel_data_used2 = np.genfromtxt("voxelDataFFTW2.csv", delimiter=",")

    magnitude1 = np.zeros((n, n))
    phase1 = np.zeros((n, n))
    voxel_data1 = np.zeros((n, n))
    magnitude2 = np.zeros((n, n))
    phase2 = np.zeros((n, n))
    voxel_data2 = np.zeros((n, n))

    for jj in range(n):
        for kk in range(n):
            magnitude1[jj, kk] = magnitude_fftw1[jj * n - n + kk]
            phase1[jj, kk] = phase_fftw1[jj * n - n + kk]
            voxel_data1[jj, kk] = voxel_data_used1[(jj - 1) * n + kk]
            magnitude2[jj, kk] = magnitude_fftw2[jj * n - n + kk]
            phase2[jj, kk] = phase_fftw2[jj * n - n + kk]
            voxel_data2[jj, kk] = voxel_data_used2[(jj - 1) * n + kk]

    magnitude1 = np.fft.fftshift(magnitude1)
    phase1 = np.fft.fftshift(phase1)
    magnitude2 = np.fft.fftshift(magnitude2)
    phase2 = np.fft.fftshift(phase2)

    plt.figure(1)
    plt.clf()
    plt.imshow(magnitude1, aspect='equal')
    plt.title('Magnitude first PCL Voxel:')

    plt.figure(2)
    plt.clf()
    plt.imshow(voxel_data1, aspect='equal')
    plt.title('First PointCloud as Voxel:')

    plt.figure(3)
    plt.imshow(magnitude2, aspect='equal')
    plt.title('Magnitude second PCL Voxel:')

    plt.figure(4)
    plt.imshow(voxel_data2, aspect='equal')
    plt.title('Second PointCloud as Voxel:')

    resampled_data_for_sphere1 = np.genfromtxt("resampledVoxel1.csv", delimiter=",")
    resampled_data_for_sphere2 = np.genfromtxt("resampledVoxel2.csv", delimiter=",")

    resampled_data_for_sphere_result1 = np.zeros((n, n))
    resampled_data_for_sphere_result2 = np.zeros((n, n))

    for i in range(1, n + 1):
        for jj in range(1, n + 1):
            resampled_data_for_sphere_result1[i - 1, jj - 1] = resampled_data_for_sphere1[(i - 1) * n + jj - 1]
            resampled_data_for_sphere_result2[i - 1, jj - 1] = resampled_data_for_sphere2[(i - 1) * n + jj - 1]

    if True:
        fig = plt.figure(5)
        ax = fig.add_subplot(111, projection='3d')
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = 15 * np.outer(np.cos(u), np.sin(v))
        y = 15 * np.outer(np.sin(u), np.sin(v))
        z = 15 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, facecolors=plt.cm.jet(resampled_data_for_sphere_result2 / resampled_data_for_sphere_result2.max()), rstride=2, cstride=2)
        plt.title('Second Resampled:')

        fig = plt.figure(6)
        ax = fig.add_subplot(111, projection='3d')
        x = 15 * np.outer(np.cos(u), np.sin(v))
        y = 15 * np.outer(np.sin(u), np.sin(v))
        z = 15 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, facecolors=plt.cm.jet(resampled_data_for_sphere_result1 / resampled_data_for_sphere_result1.max()), rstride=2, cstride=2)
        plt.title('first Resampled:')

    plt.figure(7)
    correlation_of_angles = np.genfromtxt("resultingCorrelation1D.csv", delimiter=",")
    plt.plot(correlation_of_angles)

    correlation_matrix_shift_1d = np.genfromtxt("resultingCorrelationShift.csv", delimiter=",")
    result_size = int(np.sqrt(len(correlation_matrix_shift_1d)))
    correlation_matrix_shift_2d = correlation_matrix_shift_1d.reshape(result_size, result_size)

    plt.figure(8)
    x_plot, y_plot = np.meshgrid(np.arange(1, result_size + 1), np.arange(1, result_size + 1))
    ax = plt.axes(projection='3d')
    ax.plot_surface(x_plot, y_plot, correlation_matrix_shift_2d)
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")

    plt.figure(9)
    point_cloud_result1 = o3d.io.read_point_cloud("resultingPCL1.pcd")
    point_cloud_result2 = o3d.io.read_point_cloud("resultingPCL2.pcd")

    tform = np.eye(4)
    tform[0:3, 0:3] = rotation_matrix(0, 0, 0)
    tform[0, 3] = 0
    tform[1, 3] = 0

    points = np.array(point_cloud_result1.points)
    transformed_points = (tform @ np.vstack([points, np.ones(len(points))]).T).T[:, 0:3]
    point_cloud_result1.points = o3d.utility.Vector3dVector(transformed_points)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud_result1)
    vis.add_geometry(point_cloud_result2)
    vis.run()
    vis.destroy_window()

    resulting_transformation_of_scan = np.genfromtxt("resultingTransformation.csv", delimiter=",")
