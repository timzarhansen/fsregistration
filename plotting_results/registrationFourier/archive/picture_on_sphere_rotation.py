import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def rotation_matrix_zyz(r1, r2, r3):
    rotation = np.array([
        [np.cos(r1) * np.cos(r2) * np.cos(r3) - np.sin(r1) * np.sin(r3), -np.cos(r3) * np.sin(r1) - np.cos(r1) * np.cos(r2) * np.sin(r3), np.cos(r1) * np.sin(r2)],
        [np.cos(r1) * np.sin(r3) + np.cos(r2) * np.cos(r3) * np.sin(r1), np.cos(r1) * np.cos(r3) - np.cos(r2) * np.sin(r1) * np.sin(r3), np.sin(r1) * np.sin(r2)],
        [-np.cos(r3) * np.sin(r2), np.sin(r2) * np.sin(r3), np.cos(r2)]
    ])
    return rotation


def matrix2_interleaved_format(A):
    output_vector = A.flatten(order='C')
    return output_vector


# Clear workspace equivalent (not strictly needed in Python)
# clc, clear - skipped

I = np.array(Image.open('cameraman.tif'))
f_theta_phi1 = I
f_theta_phi2 = np.rot90(np.rot90(np.rot90(I.T), 3)).T
theta_rotation = np.linspace(0, np.pi, 256)
psi_rotation = np.linspace(0, 2 * np.pi, 256)
points_on_sphere = np.zeros((len(f_theta_phi2), len(f_theta_phi2), 3))
rotation_of_sphere = rotation_matrix_zyz(0.0, 0.1, 0.8)

for j in range(len(f_theta_phi2)):
    for k in range(len(f_theta_phi2)):
        points_on_sphere[j, k, :] = [
            np.sin(theta_rotation[j]) * np.cos(psi_rotation[k]),
            np.sin(theta_rotation[j]) * np.sin(psi_rotation[k]),
            np.cos(theta_rotation[j])
        ]

points_on_sphere_after_rotation = points_on_sphere.copy()
for j in range(len(f_theta_phi2)):
    for k in range(len(f_theta_phi2)):
        points_on_sphere_after_rotation[j, k, :] = rotation_of_sphere @ points_on_sphere_after_rotation[j, k, :]

new_image = np.zeros((len(f_theta_phi2), len(f_theta_phi2), 2))
for j in range(len(f_theta_phi2)):
    for k in range(len(f_theta_phi2)):
        theta_new = np.arccos(points_on_sphere_after_rotation[j, k, 2])
        psi_new = np.arctan2(points_on_sphere_after_rotation[j, k, 1], points_on_sphere_after_rotation[j, k, 0])
        psi_new = np.mod(psi_new + 2 * np.pi, 2 * np.pi)
        index_theta = int(np.floor(theta_new / np.pi * 256)) + 1
        index_psi = int(np.floor(psi_new / (2 * np.pi) * 256)) + 1
        if index_theta > 256:
            index_theta = 256
        if index_psi > 256:
            index_psi = 256

        new_image[index_theta, index_psi, 0] = new_image[index_theta, index_psi, 0] + f_theta_phi2[j, k]
        new_image[index_theta, index_psi, 1] = new_image[index_theta, index_psi, 1] + 1

for j in range(len(f_theta_phi2)):
    for k in range(len(f_theta_phi2)):
        if new_image[j, k, 1] > 0:
            new_image[j, k, 0] = new_image[j, k, 0] / new_image[j, k, 1]

f_theta_phi2 = new_image[:, :, 0]

if True:
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, facecolors=plt.cm.gray(f_theta_phi1 / 255), edgecolor='none')
    ax.axis('equal')

    fig = plt.figure(4)
    ax = fig.add_subplot(111, projection='3d')
    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, facecolors=plt.cm.gray(f_theta_phi2 / 255), edgecolor='none')
    ax.axis('equal')
else:
    plt.figure(3)
    plt.imshow(f_theta_phi1, aspect='equal')

    plt.figure(4)
    plt.imshow(f_theta_phi2, aspect='equal')

f_theta_phi1_interleaved = matrix2_interleaved_format(f_theta_phi1)
f_theta_phi2_interleaved = matrix2_interleaved_format(f_theta_phi2)

np.savetxt("matrixone.csv", f_theta_phi1_interleaved.T, delimiter=",")
np.savetxt("matrixtwo.csv", f_theta_phi2_interleaved.T, delimiter=",")

results = np.genfromtxt("ergWrap.txt", delimiter=",")

results = results / np.max(results)

A = results.reshape((256, 256, 256))

x = np.linspace(0, 2 * np.pi, 256)
y = np.linspace(0, 2 * np.pi, 256)
X, Y = np.meshgrid(x, y)
for i in range(256):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, A[:, :, i], edgecolor='none')
    ax.set_xlim([0, 7])
    ax.set_ylim([0, 7])
    ax.set_zlim([0.5, 1])
    plt.pause(0.01)
