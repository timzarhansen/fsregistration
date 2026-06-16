import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, rotate
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

point_cloud = o3d.io.read_point_cloud("after_voxel_second.pcd")

# create voxel grid of pcl
from_to = 30
number_of_points = 128

shift_first = np.array([0, 0, 0])
rotation = rotation_matrix(0.0, 0.0, 0.0)

voxel_data_1 = get_voxel_data(number_of_points, shift_first, rotation, point_cloud, from_to)

spectrum_1, magnitude_1, phase_1 = plot_ffts(voxel_data_1, 1)

# create voxel grid of pcl and shift by value
shift_first = np.array([0, 0, 0])
rotation = rotation_matrix(0.0, 0.0, 0.5)

voxel_data_2 = get_voxel_data(number_of_points, shift_first, rotation, point_cloud, from_to)

voxel_data_2 = voxel_data_2 + 0.0 * np.random.randn(*voxel_data_2.shape)
spectrum_2, magnitude_2, phase_2 = plot_ffts(voxel_data_2, 2)

# calculate sampled f(theta,phi)
r_numbers = np.arange(4, number_of_points // 2 - 1)
b = number_of_points // 2

theta_index = np.arange(1, 2 * b + 1)
phi_index = np.arange(1, 2 * b + 1)
phi = np.pi * phi_index / b
theta = np.pi * (2 * theta_index + 1) / (4 * b)

maximum_set_2 = np.max(magnitude_1)
magnitude_1 = magnitude_1 / maximum_set_2
magnitude_2 = magnitude_2 / maximum_set_2

f_theta_phi_1 = sampled_f_theta_phi(magnitude_1, theta, phi, b, r_numbers)
f_theta_phi_2 = sampled_f_theta_phi(magnitude_2, theta, phi, b, r_numbers)

if True:
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
    x = np.sin(u) * np.cos(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(u)
    ax.plot_surface(x, y, z, facecolors=plt.cm.jet(f_theta_phi_1 / f_theta_phi_1.max()), edgecolor='none')
    plt.axis('equal')

    fig = plt.figure(4)
    ax = fig.add_subplot(111, projection='3d')
    x = np.sin(u) * np.cos(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(u)
    ax.plot_surface(x, y, z, facecolors=plt.cm.jet(f_theta_phi_2 / f_theta_phi_2.max()), edgecolor='none')
    plt.axis('equal')
else:
    plt.figure(3)
    plt.imshow(f_theta_phi_1)
    plt.axis('image')

    plt.figure(4)
    plt.imshow(f_theta_phi_2)
    plt.axis('image')

f_theta_phi_1_interleaved = matrix_2_interleaved_format(f_theta_phi_1)
f_theta_phi_2_interleaved = matrix_2_interleaved_format(f_theta_phi_2)

np.savetxt("matrixone.csv", f_theta_phi_1_interleaved.T, delimiter=",")
np.savetxt("matrixtwo.csv", f_theta_phi_2_interleaved.T, delimiter=",")

n = 128
results = np.genfromtxt("ergWrap.txt", delimiter=",")
results = results / np.max(results)

a = results.reshape(n, n, n)

correlation_number_matrix = a[:, :, 0]

theta_list = np.linspace(0, 2 * np.pi, n)
psi_list = np.linspace(0, 2 * np.pi, n)
list_of_correlation_and_angle = np.zeros((n * n, 2))

for j in range(n):
    for k in range(n):
        current_angle = (theta_list[j] + psi_list[k] + 4 * np.pi) % (2 * np.pi)
        list_of_correlation_and_angle[j * n - n + k, 0] = current_angle
        list_of_correlation_and_angle[j * n - n + k, 1] = correlation_number_matrix[k, j]

plt.figure(3)
plt.plot(list_of_correlation_and_angle[:, 0], list_of_correlation_and_angle[:, 1], ".")

list_of_correlation_and_angle = list_of_correlation_and_angle[list_of_correlation_and_angle[:, 0].argsort()]
current_average_angle = list_of_correlation_and_angle[0, 0]
number_of_angles = 1
average_correlation = list_of_correlation_and_angle[0, 1]
i = 1
resulting_plotting = np.zeros((n * n, 2))

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

plt.figure(4)
plt.clf()
plt.plot(resulting_plotting[:, 0], resulting_plotting[:, 1], ".")

smooth_func = interp1d(resulting_plotting[:, 0], resulting_plotting[:, 1], kind='cubic')
x_smooth = np.linspace(resulting_plotting[:, 0].min(), resulting_plotting[:, 0].max(), 1000)
y_smooth = smooth_func(x_smooth)

pks, properties = find_peaks(y_smooth, x_smooth)
potential_rotations = x_smooth[pks]
print(potential_rotations)

plt.hold(True)
current_peak_x_value = potential_rotations[0]
y = resulting_plotting[(np.abs(resulting_plotting[:, 0] - current_peak_x_value) < 0.1), 1]
x = resulting_plotting[(np.abs(resulting_plotting[:, 0] - current_peak_x_value) < 0.1), 0]

from scipy.optimize import curve_fit

def gauss_func(x, a, b, c):
    return a * np.exp(-((x - b) / c / 2) ** 2)

popt, _ = curve_fit(gauss_func, x, y, p0=[np.max(y), current_peak_x_value, 0.5])
x_fit = np.linspace(x.min(), x.max(), 100)
plt.plot(x_fit, gauss_func(x_fit, *popt))

voxel_data_2_rotated = voxel_data_2.copy()
voxel_data_2_rotated[:, :, number_of_points // 2] = rotate(voxel_data_2[:, :, number_of_points // 2], 0.5 / np.pi * 180, reshape=False, mode='nearest')

spectrum_2_rotated, magnitude_2_rotated, phase_2_rotated = plot_ffts(voxel_data_2_rotated, 4)

resulting_phase_difference = phase_1 - phase_2_rotated

inverse_fft_for_phase = np.fft.fftshift(np.fft.ifftn(np.exp(1j * resulting_phase_difference)))

imaginary_part = np.imag(inverse_fft_for_phase)
real_part = np.real(inverse_fft_for_phase)
phase = np.arctan2(imaginary_part, real_part)

magnitude = np.abs(inverse_fft_for_phase)
magnitude = magnitude[:, :, number_of_points // 2]

plt.figure(5)
x_plot, y_plot = np.meshgrid(np.arange(1, number_of_points + 1), np.arange(1, number_of_points + 1))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_plot, y_plot, magnitude, edgecolor='none')
plt.title('magnitude of invFFT(arg(R)-arg(S)) ')

m, i = np.unravel_index(np.argmax(np.abs(inverse_fft_for_phase)), inverse_fft_for_phase.shape)
dim1, dim2, dim3 = m

tf1, p = find_peaks(magnitude.flatten())

preshaped = p.flatten()

peaks_of_shift, i_sorted = np.sort(preshaped)[::-1], np.argsort(preshaped)[::-1]

diff_peaks = np.abs(np.diff(peaks_of_shift))
tf1, pos = np.max(diff_peaks), np.argmax(diff_peaks)

index_p_col = np.zeros(pos + 1)
index_p_row = np.zeros(pos + 1)
position = np.zeros((pos + 1, 2))

for j in range(pos + 1):
    index_p_col[j] = np.ceil(i_sorted[j] / number_of_points)
    index_p_row[j] = i_sorted[j] - number_of_points * (index_p_col[j] - 1)
    position[j, 0:2] = [index_p_row[j] - number_of_points / 2, index_p_col[j] - number_of_points / 2]


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


def get_voxel_data(number_of_points, shift_first, rotation, point_cloud, from_to):
    voxel_data = np.zeros((number_of_points, number_of_points, number_of_points))
    points = np.array(point_cloud.points)
    scaling = number_of_points / (2 * from_to)
    
    for point in points:
        rotated_point = rotation @ point
        shifted_point = rotated_point + shift_first
        idx = ((shifted_point + from_to) * scaling).astype(int)
        idx = np.clip(idx, 0, number_of_points - 1)
        voxel_data[idx[0], idx[1], idx[2]] += 1
    
    return voxel_data


def plot_ffts(data, figure_num):
    spectrum = np.fft.fftn(data)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    return spectrum, magnitude, phase


def sampled_f_theta_phi(magnitude, theta, phi, b, r_numbers):
    f_theta_phi = np.zeros((len(theta), len(phi)))
    for i, t in enumerate(theta):
        for j, p in enumerate(phi):
            r = r_numbers
            f_theta_phi[i, j] = np.mean(magnitude[r, r, r])
    return f_theta_phi


def matrix_2_interleaved_format(matrix):
    return matrix.flatten()
