import json
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, factorial
from scipy.special import eval_jacobi
from scipy.ndimage import gaussian_filter
from skimage.morphology import regional_maxima


def read_json_graph_pcl(name_of_file):
    with open(name_of_file, 'r') as f:
        val = json.load(f)

    max_keyframes = len(val['keyFrames'])
    max_points = max(len(kf['pointCloud']) for kf in val['keyFrames']) + 1
    data = np.zeros((max_keyframes, max_points, 3))

    for i in range(len(val['keyFrames'])):
        data[i, 0, 0:3] = [
            val['keyFrames'][i]['position']['x'],
            val['keyFrames'][i]['position']['y'],
            val['keyFrames'][i]['position']['z']
        ]
        eulerAngles = np.array([
            val['keyFrames'][i]['position']['yaw'],
            val['keyFrames'][i]['position']['pitch'],
            val['keyFrames'][i]['position']['roll']
        ])
        yaw_rotation = np.array([
            [np.cos(eulerAngles[0]), -np.sin(eulerAngles[0]), 0],
            [np.sin(eulerAngles[0]), np.cos(eulerAngles[0]), 0],
            [0, 0, 1]
        ])
        pitch_rotation = np.array([
            [np.cos(eulerAngles[1]), 0, np.sin(eulerAngles[1])],
            [0, 1, 0],
            [-np.sin(eulerAngles[1]), 0, np.cos(eulerAngles[1])]
        ])
        roll_rotation = np.array([
            [1, 0, 0],
            [0, np.cos(eulerAngles[2]), -np.sin(eulerAngles[2])],
            [0, np.sin(eulerAngles[2]), np.cos(eulerAngles[2])]
        ])
        rotm = yaw_rotation @ pitch_rotation @ roll_rotation
        for j in range(len(val['keyFrames'][i]['pointCloud'])):
            point = np.array([
                val['keyFrames'][i]['pointCloud'][j]['point']['x'],
                val['keyFrames'][i]['pointCloud'][j]['point']['y'],
                val['keyFrames'][i]['pointCloud'][j]['point']['z']
            ])
            data[i, j + 1, 0:3] = rotm @ point

    return data


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


def sampled_f_theta_phi(magnitude, theta, phi, B, r_numbers):
    f_theta_phi = np.zeros((theta.shape[1], phi.shape[1]))
    for r in r_numbers:
        f_theta_phi_tmp = np.zeros((theta.shape[1], phi.shape[1]))
        for j in range(theta.shape[1]):
            for k in range(phi.shape[1]):
                u = int(r * np.sin(theta[j]) * np.cos(phi[k]) + B)
                v = int(r * np.sin(theta[j]) * np.sin(phi[k]) + B)
                w = int(r * np.cos(theta[j]) + B)
                f_theta_phi_tmp[j, k] = f_theta_phi_tmp[j, k] + magnitude[u, v, w]
        f_theta_phi = f_theta_phi + f_theta_phi_tmp
    return f_theta_phi


def legendre_p(n, x):
    from sympy import legendre
    return legendre(n, x)


def ylm_of_tp(l, theta, phi):
    ylp = np.zeros((l + 1, theta.shape[0], phi.shape[0]))
    ylm = np.zeros((l, theta.shape[0], phi.shape[0]))
    for m in range(l + 1):
        x = symbols('x')
        leg = legendre_p(l, x)
        leg_diff = diff(leg, x, abs(m))
        plm = ((-1)**abs(m)) * ((1 - x**2)**(abs(m)/2)) * leg_diff
        x0 = np.cos(theta)
        plm_solution_plus = np.array([[float(plm.subs(x, cos_val)) for cos_val in x0_flat] for _ in range(theta.shape[0])])
        if m > 0:
            plm_solution_minus = plm_solution_plus * (-1)**abs(m) * factorial(l - abs(m)) / factorial(l + abs(m))
            a = (2 * l + 1) * factorial(l - m)
            b = 4 * np.pi * factorial(l + m)
            c = np.sqrt(a / b)
            ylp[m, :, :] = (-1)**m * c * plm_solution_plus.T * np.exp(-1j * m * phi)
            a = (2 * l + 1) * factorial(l + m)
            b = 4 * np.pi * factorial(l - m)
            c = np.sqrt(a / b)
            ylm[m, :, :] = (-1)**(-m) * c * plm_solution_minus.T * np.exp(1j * m * phi)
        else:
            a = (2 * l + 1) * factorial(l - m)
            b = 4 * np.pi * factorial(l + m)
            c = np.sqrt(a / b)
            ylp[m, :, :] = (-1)**m * c * plm_solution_plus.T * np.exp(-1j * m * phi)
    return ylp, ylm


def fourie_coeff(f_theta_phi, l_max, B, theta, phi):
    b = B
    weights = np.ones(B * 2)
    flm_p = np.zeros((l_max, l_max + 1))
    flm_m = np.zeros((l_max, l_max))
    for l in range(1, l_max + 1):
        ylp, ylm = ylm_of_tp(l, theta, phi)
        for m in range(l + 1):
            for j in range(2 * B):
                for k in range(2 * B):
                    if m > 0:
                        flm_p[l - 1, m] += weights[j] * f_theta_phi[j, k] * ylp[m, j, k]
                        flm_m[l - 1, m - 1] += weights[j] * f_theta_phi[j, k] * ylm[m - 1, j, k]
                    else:
                        flm_p[l - 1, m] += weights[j] * f_theta_phi[j, k] * ylp[m, j, k]
    return flm_p


def wignerd_function(pitch, l, m1, m2):
    mu = abs(m1 - m2)
    vi = abs(m1 + m2)
    s = l - (mu + vi) / 2
    if m2 >= m1:
        zeta = 1
    else:
        zeta = (-1) ** (m2 - m1)
    d = zeta * np.sqrt(np.math.factorial(s) * np.math.factorial(s + mu + vi) / (np.math.factorial(s + mu) * np.math.factorial(s + vi))) * (np.sin(pitch / 2)) ** mu * (np.cos(pitch / 2)) ** vi * eval_jacobi(s, mu, vi, np.cos(pitch))
    return d


def calculate_score_of_rotation(yaw_to_test, flm1, flm2, B):
    c_output = 0
    roll = 0
    pitch = 0
    for l in range(1, B + 1):
        for m1 in range(1, l + 2):
            for m2 in range(1, l + 2):
                c_output += flm1[l, m1] * flm2[l, m2] * (-1) ** (m1 - m2) * np.exp(-1j * (m1 - 1) * roll) * wignerd_function(pitch, l, m1 - 1, m2 - 1) * np.exp(-1j * (m2 - 1) * yaw_to_test)
                if np.isnan(c_output):
                    print(c_output)
    print("done")
    print(yaw_to_test)
    print(c_output)
    return c_output


# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

data = read_json_graph_pcl("outputSlam.json")

# plot two pcl
position_cloud_one = 18
position_cloud_zwo = position_cloud_one + 1

pointcloud1 = np.squeeze(data[position_cloud_one, :, :])
pointcloud1 = pointcloud1[~(pointcloud1[:, 0] == 0 & pointcloud1[:, 1] == 0 & pointcloud1[:, 2] == 0), :]

pointcloud2 = np.squeeze(data[position_cloud_zwo, :, :])
pointcloud2 = pointcloud2[~(pointcloud2[:, 0] == 0 & pointcloud2[:, 1] == 0 & pointcloud2[:, 2] == 0), :]

# create voxel grid of pcl
from_to = 60
number_of_points = 64
voxel_data1 = np.zeros((number_of_points, number_of_points, number_of_points))

for j in range(pointcloud1.shape[0]):
    x_pos = pointcloud1[j, 0]
    y_pos = pointcloud1[j, 1]
    x_index = int((x_pos + from_to) / (from_to * 2) * number_of_points)
    y_index = int((y_pos + from_to) / (from_to * 2) * number_of_points)
    z_index = int((0 + from_to) / (from_to * 2) * number_of_points)
    voxel_data1[x_index, y_index, z_index] = 1

spectrum1, magnitude1, phase1 = plot_ffts(voxel_data1, 1)

voxel_data2 = np.zeros((number_of_points, number_of_points, number_of_points))

for j in range(pointcloud2.shape[0]):
    x_pos = pointcloud2[j, 0]
    y_pos = pointcloud2[j, 1]
    x_index = int((x_pos + from_to) / (from_to * 2) * number_of_points)
    y_index = int((y_pos + from_to) / (from_to * 2) * number_of_points)
    z_index = int((0 + from_to) / (from_to * 2) * number_of_points)
    voxel_data2[x_index, y_index, z_index] = 1

spectrum2, magnitude2, phase2 = plot_ffts(voxel_data2, 2)

# calculate rotation
r_numbers = np.arange(2, 32)
B = number_of_points // 2

theta_index = np.arange(1, 2 * B + 1)
phi_index = np.arange(1, 2 * B + 1)
phi = np.pi * phi_index / B
theta = np.pi * (2 * theta_index + 1) / (4 * B)

# 2D
f_theta_phi1 = sampled_f_theta_phi(magnitude1, theta, phi, B, r_numbers)
f_theta_phi2 = sampled_f_theta_phi(magnitude2, theta, phi, B, r_numbers)

plt.figure(3)
plt.imshow(f_theta_phi1, aspect='equal')

plt.figure(4)
plt.imshow(f_theta_phi2, aspect='equal')

# calculate fourie coeff of fThetaPhi1
flm1 = fourie_coeff(f_theta_phi1, number_of_points, B, theta, phi)

# calculate fourie coeff of fThetaPhi2
flm2 = fourie_coeff(f_theta_phi2, number_of_points, B, theta, phi)

n_test = 40
yaw_tmp = np.linspace(-np.pi, np.pi, n_test)
c_output = np.zeros((n_test, 1))
for j in range(n_test):
    c_output[j] = calculate_score_of_rotation(yaw_tmp[j], flm1, flm2, B)

# calculate translation
resulting_phase_difference = phase1 - phase2

inverse_fft_for_phase = np.fft.fftshift(np.fft.ifft2(np.exp(1j * resulting_phase_difference)))

immaginary_part = np.imag(inverse_fft_for_phase)
real_part = np.real(inverse_fft_for_phase)
phase = np.arctan2(immaginary_part, real_part)

magnitude = np.abs(inverse_fft_for_phase)
magnitude = magnitude[:, :, magnitude.shape[0] // 2]
magnitude = gaussian_filter(magnitude, sigma=2)

plt.figure(5)
plt.imshow(magnitude, aspect='equal')

P = regional_maxima(magnitude)
P = P.astype(int)

plt.figure(6)
x_plot, y_plot = np.meshgrid(np.arange(1, number_of_points + 1), np.arange(1, number_of_points + 1))
ax = plt.axes(projection='3d')
ax.plot_surface(x_plot, y_plot, magnitude)
plt.title('magnitude of invFFT(arg(R)-arg(S))')

p_reshaped = np.reshape(magnitude * P, (1, -1))

peaks_of_shift, I = np.sort(p_reshaped.flatten()[::-1]), np.argsort(-p_reshaped.flatten())

diff_peaks = np.abs(np.diff(peaks_of_shift))
tf1, pos = np.max(diff_peaks), np.argmax(diff_peaks)

index_p_col = np.zeros(pos, dtype=int)
index_p_row = np.zeros(pos, dtype=int)
position = np.zeros((pos, 2))
translation_shift = np.zeros((pos, 2))

for j in range(pos):
    index_p_col[j] = int(np.ceil(I[j] / number_of_points))
    index_p_row[j] = I[j] - number_of_points * (index_p_col[j] - 1)
    position[j, 0:2] = [index_p_row[j] - number_of_points / 2, index_p_col[j] - number_of_points / 2]

for j in range(pos):
    translation_tmp = -position[j, :] / number_of_points * from_to * 2
    translation_shift[j, 0:2] = translation_tmp

# test -1.55 first value 1.65 second value
plt.figure(7)
from_idx = 20
to_idx = 40
tmp = np.abs(c_output) / (10 ** 12)

plt.plot(yaw_tmp[from_idx:to_idx], tmp[from_idx:to_idx])
