import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
import subprocess

# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

for j in range(1, 230):
    which_keyframe = j
    name_of_folder = '/home/tim-external/dataFolder/gazeboCorrectedPCLs/'
    first_scan = f'pclKeyFrame{which_keyframe}.pcd'
    second_scan = f'pclKeyFrame{which_keyframe + 1}.pcd'

    command = f'rosrun underwaterslam registrationOfTwoPCLs {name_of_folder}{first_scan} {name_of_folder}{second_scan}'
    subprocess.call(command, shell=True)

    magnitude_fftw_1 = np.genfromtxt("magnitudeFFTW1.csv", delimiter=",")
    n = int(np.sqrt(magnitude_fftw_1.shape[0]))

    phase_fftw_1 = np.genfromtxt("phaseFFTW1.csv", delimiter=",")
    voxel_data_used_1 = np.genfromtxt("voxelDataFFTW1.csv", delimiter=",")

    magnitude_fftw_2 = np.genfromtxt("magnitudeFFTW2.csv", delimiter=",")
    phase_fftw_2 = np.genfromtxt("phaseFFTW2.csv", delimiter=",")
    voxel_data_used_2 = np.genfromtxt("voxelDataFFTW2.csv", delimiter=",")

    magnitude_1 = np.zeros((n, n))
    phase_1 = np.zeros((n, n))
    voxel_data_1 = np.zeros((n, n))
    magnitude_2 = np.zeros((n, n))
    phase_2 = np.zeros((n, n))
    voxel_data_2 = np.zeros((n, n))

    for jj in range(n):
        for kk in range(n):
            magnitude_1[jj, kk] = magnitude_fftw_1[jj * n - n + kk]
            phase_1[jj, kk] = phase_fftw_1[jj * n - n + kk]
            voxel_data_1[jj, kk] = voxel_data_used_1[jj * n + kk]
            magnitude_2[jj, kk] = magnitude_fftw_2[jj * n - n + kk]
            phase_2[jj, kk] = phase_fftw_2[jj * n - n + kk]
            voxel_data_2[jj, kk] = voxel_data_used_2[jj * n + kk]

    magnitude_1 = np.fft.fftshift(magnitude_1)
    phase_1 = np.fft.fftshift(phase_1)
    magnitude_2 = np.fft.fftshift(magnitude_2)
    phase_2 = np.fft.fftshift(phase_2)

    plt.figure(1)
    plt.subplot(1, 3, 2)
    plt.imshow(magnitude_1)
    plt.title('Magnitude Voxel: 1')
    plt.axis('image')

    plt.subplot(1, 3, 3)
    plt.imshow(phase_1)
    plt.title('Phase Voxel: 1')
    plt.axis('image')

    plt.subplot(1, 3, 1)
    plt.imshow(voxel_data_1)
    plt.axis('image')

    plt.figure(2)
    plt.subplot(1, 3, 2)
    plt.imshow(magnitude_2)
    plt.title('Magnitude Voxel: 1')
    plt.axis('image')

    plt.subplot(1, 3, 3)
    plt.imshow(phase_2)
    plt.title('Phase Voxel: 1')
    plt.axis('image')

    plt.subplot(1, 3, 1)
    plt.imshow(voxel_data_2)
    plt.axis('image')

    resampled_data_for_sphere_1 = np.genfromtxt("resampledVoxel1.csv", delimiter=",")
    resampled_data_for_sphere_2 = np.genfromtxt("resampledVoxel2.csv", delimiter=",")

    resampled_data_for_sphere_result_1 = np.zeros((n, n))
    resampled_data_for_sphere_result_2 = np.zeros((n, n))

    for i in range(n):
        for jj in range(n):
            resampled_data_for_sphere_result_1[i, jj] = resampled_data_for_sphere_1[i * n + jj]
            resampled_data_for_sphere_result_2[i, jj] = resampled_data_for_sphere_2[i * n + jj]

    if True:
        pass
    else:
        plt.figure(3)
        plt.imshow(f_theta_phi_1)
        plt.axis('image')
        plt.figure(4)
        plt.imshow(resampled_data_for_sphere_result)
        plt.axis('image')

    n = 128
    results = np.genfromtxt("resultCorrelation3D.csv", delimiter=",")
    results = results / np.max(results)
    result_size = int(np.sqrt(np.sqrt(len(results))))
    a = results.reshape(result_size, result_size, result_size)

    correlation_number_matrix = a[:, :, 0]

    theta_list = np.linspace(0, 2 * np.pi - 2 * np.pi / result_size, result_size)
    psi_list = np.linspace(0, 2 * np.pi - 2 * np.pi / result_size, result_size)
    list_of_correlation_and_angle = np.zeros((result_size * result_size, 2))

    for jj in range(result_size):
        for kk in range(result_size):
            current_angle = (theta_list[jj] + psi_list[kk] + 4 * np.pi) % (2 * np.pi)
            list_of_correlation_and_angle[jj * result_size - result_size + kk, 0] = current_angle
            list_of_correlation_and_angle[jj * result_size - result_size + kk, 1] = correlation_number_matrix[kk, jj]

    list_of_correlation_and_angle = list_of_correlation_and_angle[list_of_correlation_and_angle[:, 0].argsort()]
    current_average_angle = list_of_correlation_and_angle[0, 0]
    number_of_angles = 1
    average_correlation = list_of_correlation_and_angle[0, 1]
    i = 1
    resulting_plotting = np.zeros((result_size * result_size, 2))

    for k in range(1, len(list_of_correlation_and_angle)):
        if abs(current_average_angle - list_of_correlation_and_angle[k, 0]) < 1 / n / 4:
            number_of_angles += 1
            average_correlation += list_of_correlation_and_angle[k, 1]
        else:
            resulting_plotting[i, :] = [current_average_angle, average_correlation / number_of_angles]
            i += 1
            number_of_angles = 1
            current_average_angle = list_of_correlation_and_angle[k, 0]
            average_correlation = list_of_correlation_and_angle[k, 1]

    resulting_plotting[i, :] = [current_average_angle, average_correlation / number_of_angles]

    plt.figure(6)
    correlation_of_angles = np.genfromtxt("resultingCorrelation1D.csv", delimiter=",")
    plt.plot(correlation_of_angles)

    correlation_matrix_shift_1d = np.genfromtxt("resultingCorrelationShift.csv", delimiter=",")
    result_size = int(np.sqrt(len(correlation_matrix_shift_1d)))
    correlation_matrix_shift_2d = correlation_matrix_shift_1d.reshape(result_size, result_size)

    plt.figure(8)
    x_plot, y_plot = np.meshgrid(np.arange(1, result_size + 1), np.arange(1, result_size + 1))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_plot, y_plot, correlation_matrix_shift_2d)
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")

    voxel_data_1_trans_padding = voxel_data_1
    voxel_data_2_trans_padding = voxel_data_2

    padding_size = 0
    voxel_data_1_trans_padding = np.pad(voxel_data_1_trans_padding, ((padding_size, padding_size), (padding_size, padding_size)), 'constant')
    voxel_data_2_trans_padding = np.pad(voxel_data_2_trans_padding, ((padding_size, padding_size), (padding_size, padding_size)), 'constant')

    spectrum_1_trans = np.fft.fftn(voxel_data_1_trans_padding)
    spectrum_2_trans = np.fft.fftn(voxel_data_2_trans_padding)

    inverse_fft_for_phase = np.fft.fftshift(np.fft.ifft2(128 * 128 * spectrum_1_trans * np.conj(spectrum_2_trans)))
    magnitude_translation = np.abs(inverse_fft_for_phase)

    m, i = np.unravel_index(np.argmax(magnitude_translation), magnitude_translation.shape)
    y_axis_index, x_axis_index = m, i

    how_many_remove = 0
    y = magnitude_translation[y_axis_index, :].flatten()
    shift_xy = (x_axis_index - 1) * 0.6
    x = np.linspace(0, 0.6 * (n - 1), n)

    plt.figure(5)
    plt.clf()
    plt.plot(x, y)

    y = y[how_many_remove + 1:-how_many_remove] if how_many_remove > 0 else y
    x = x[how_many_remove + 1:-how_many_remove] if how_many_remove > 0 else x

    plt.hold(True)

    from scipy.optimize import curve_fit

    def gauss_func(x, a, b, c):
        return a * np.exp(-((x - b) / c / 2) ** 2)

    popt, _ = curve_fit(gauss_func, x, y, p0=[np.max(y), shift_xy, 20])
    x_fit = np.linspace(x.min(), x.max(), 100)
    plt.plot(x_fit, gauss_func(x_fit, *popt))

    plt.figure(3)
    import open3d as o3d
    point_cloud_result_1 = o3d.io.read_point_cloud("resultingPCL1.pcd")
    point_cloud_result_2 = o3d.io.read_point_cloud("resultingPCL2.pcd")
    tform = np.eye(4)
    tform[0:3, 0:3] = rotation_matrix(0, 0, 0)
    tform[0, 3] = 0
    tform[1, 3] = 0

    from open3d.pipelines.registration import TransformationEstimationPointToPlane
    point_cloud_result_1.transform(tform)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_point_cloud(point_cloud_result_1)
    vis.add_point_cloud(point_cloud_result_2)
    vis.run()
    vis.destroy_window()


def rotation_matrix(roll, pitch, yaw):
    roll_rad = np.radians(roll)
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0, np.sin(roll_rad), np.cos(roll_rad)]
    ])

    R_y = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])

    R_z = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])

    return R_z @ R_y @ R_x
