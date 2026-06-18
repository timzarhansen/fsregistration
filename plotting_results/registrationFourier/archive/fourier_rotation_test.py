import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


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


def rotation_matrix(roll, pitch, yaw):
    rotation = np.array([
        [np.cos(yaw) * np.cos(pitch), np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll), np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)],
        [np.sin(yaw) * np.cos(pitch), np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll), np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)],
        [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)]
    ])
    return rotation


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


def matrix2_interleaved_format(A):
    output_vector = A.flatten(order='C')
    return output_vector


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

    flm_p = np.sqrt(2 * np.pi) / (2 * b) * flm_p
    flm_m = np.sqrt(2 * np.pi) / (2 * b) * flm_m

    return flm_p, flm_m


def legendre_p(n, x):
    from sympy import legendre
    return legendre(n, x)


def ylm_of_tp(l, theta, phi):
    from sympy import symbols, diff, sqrt, factorial

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


def wignerd_function(pitch, l, m1, m2):
    from scipy.special import eval_jacobi

    mu = abs(m1 - m2)
    vi = abs(m1 + m2)
    s = l - (mu + vi) / 2

    if m2 >= m1:
        zeta = 1
    else:
        zeta = (-1) ** (m2 - m1)

    d = zeta * np.sqrt(np.math.factorial(s) * np.math.factorial(s + mu + vi) / (np.math.factorial(s + mu) * np.math.factorial(s + vi))) * (np.sin(pitch / 2)) ** mu * (np.cos(pitch / 2)) ** vi * eval_jacobi(s, mu, vi, np.cos(pitch))

    return d


# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

point_cloud = o3d.io.read_point_cloud("after_voxel_second.pcd")

from_to = 30
number_of_points = 32
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
shift_first = np.array([0, 0, 0])
rotation = rotation_matrix(0.0, 0.0, 0.1)

for j in range(len(point_cloud.points)):
    position_point = np.array([point_cloud.points[j, 0] + shift_first[0], point_cloud.points[j, 1] + shift_first[1], 0])
    position_point = rotation @ position_point
    x_index = int((position_point[0] + from_to) / (from_to * 2) * number_of_points)
    y_index = int((position_point[1] + from_to) / (from_to * 2) * number_of_points)
    z_index = int((position_point[2] + from_to) / (from_to * 2) * number_of_points)
    voxel_data2[x_index, y_index, z_index] = 1

voxel_data2 = voxel_data2 + 0.0 * np.random.randn(*voxel_data2.shape)
spectrum2, magnitude2, phase2 = plot_ffts(voxel_data2, 2)

r_numbers = np.arange(10, number_of_points // 2)
B = number_of_points // 2

theta_index = np.arange(1, 2 * B + 1)
phi_index = np.arange(1, 2 * B + 1)
phi = np.pi * phi_index / B
theta = np.pi * (2 * theta_index + 1) / (4 * B)

maximum_set2 = np.max(magnitude1)
magnitude1 = magnitude1 / maximum_set2
magnitude2 = magnitude2 / maximum_set2

f_theta_phi1 = sampled_f_theta_phi(magnitude1, theta, phi, B, r_numbers)
f_theta_phi2 = sampled_f_theta_phi(magnitude2, theta, phi, B, r_numbers)

plt.figure(3)
plt.imshow(f_theta_phi1)
plt.axis('image')

plt.figure(4)
plt.imshow(f_theta_phi2)
plt.axis('image')

f_theta_phi1_interleaved = matrix2_interleaved_format(f_theta_phi1)
f_theta_phi2_interleaved = matrix2_interleaved_format(f_theta_phi2)

np.savetxt("matrixone.csv", f_theta_phi1_interleaved.T, delimiter=",")
np.savetxt("matrixtwo.csv", f_theta_phi2_interleaved.T, delimiter=",")

flm_p1, flm_m1 = fourie_coeff(f_theta_phi1, number_of_points, B, theta, phi)
flm_p2, flm_m2 = fourie_coeff(f_theta_phi2, number_of_points, B, theta, phi)

yaw_number = 51
yaw_array = np.linspace(-0.5, 0.5, yaw_number)
roll_number = 1
roll_array = np.linspace(-0.0, 0.0, roll_number)
pitch_number = 1
pitch_array = np.linspace(-0.0, 0.0, pitch_number)
c_output = np.zeros((yaw_number, roll_number, pitch_number))

for j in range(yaw_number):
    for i in range(roll_number):
        for k in range(pitch_number):
            roll = roll_array[i]
            pitch = pitch_array[k]
            yaw = yaw_array[j]

            for l in range(1, 2 * B + 1):
                for m1 in range(l + 1):
                    for m2 in range(l + 1):
                        if m1 > 0:
                            plus_add1 = flm_m1[l, m1]
                            minus_add1 = flm_p1[l, m1 + 1]
                        else:
                            plus_add1 = flm_p1[l, m1 + 1]
                            minus_add1 = 0

                        if m2 > 0:
                            plus_add2 = np.conj(flm_m2[l, m2])
                            minus_add2 = np.conj(flm_p2[l, m2 + 1])
                        else:
                            plus_add2 = np.conj(flm_p2[l, m2 + 1])
                            minus_add2 = 0

                        plus_wignerd_d_function = np.exp(-1j * m1 * roll) * wignerd_function(pitch, l, m1, m2) * np.exp(-1j * m2 * yaw)
                        c_output[j, i, k] = c_output[j, i, k] + plus_add1 * plus_add2 * (-1) ** (m1 - m2) * plus_wignerd_d_function

                        minus_wignerd_d_function = np.exp(1j * m1 * roll) * wignerd_function(pitch, l, -m1, -m2) * np.exp(1j * m2 * yaw)
                        c_output[j, i, k] = c_output[j, i, k] + minus_add1 * minus_add2 * (-1) ** (-m1 + m2) * minus_wignerd_d_function

    print("done")
    print(j)
    print(c_output[j, 0, 0])

plt.figure(5)
plt.plot(yaw_array, np.real(c_output[:, 0, 0]))

with open('test.txt', 'r') as f:
    lines = f.readlines()

c = np.array([float(line.strip()) for line in lines])
