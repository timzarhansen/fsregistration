import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def transformations_matrix(x, y, z, rx, ry, rz):
    # Create transformation matrix from translation (x,y,z) and rotation (rx,ry,rz) in radians
    R_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rx), -np.sin(rx), 0],
        [0, np.sin(rx), np.cos(rx), 0],
        [0, 0, 0, 1]
    ])
    R_y = np.array([
        [np.cos(ry), 0, np.sin(ry), 0],
        [0, 1, 0, 0],
        [-np.sin(ry), 0, np.cos(ry), 0],
        [0, 0, 0, 1]
    ])
    R_z = np.array([
        [np.cos(rz), -np.sin(rz), 0, 0],
        [np.sin(rz), np.cos(rz), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    T = np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])
    return T @ R_z @ R_y @ R_x


def volume_viewer(volume_3d):
    # Simple 3D volume visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Show a few slices
    mid_slice = volume_3d.shape[0] // 2
    x, y = np.meshgrid(np.arange(volume_3d.shape[1]), np.arange(volume_3d.shape[2]))
    ax.plot_surface(x, y, volume_3d[mid_slice, :, :], cmap='viridis')
    plt.show()


current_map_matrix = np.genfromtxt("csvFiles/current3DMapTest.csv", delimiter=",")

N = int(np.cbrt(current_map_matrix.shape[0]))
current_map_3d = np.zeros((N, N, N))

for j in range(N):
    for i in range(N):
        for k in range(N):
            current_map_3d[i, j, k] = current_map_matrix[(i - 1) * N + j + (k - 1) * N * N]

volume_viewer(current_map_3d)

# Test transformations
global_vector = np.array([10, 20, 2, 1])

A12 = transformations_matrix(0, 0, 0, 1, 2, 3)
A23 = transformations_matrix(0.2, 0.3, 0.5, 0, 0, 0)
A34 = transformations_matrix(0, 0, 0, 5, 3, 1)
A45 = transformations_matrix(0.1, 0.2, 0.5, 0, 0, 0)

B12 = transformations_matrix(0.2, 0.3, 0.5, 1, 2, 3)
B23 = transformations_matrix(0.1, 0.2, 0.5, 5, 3, 1)

local_vector = A45 @ A34 @ A23 @ A12 @ global_vector
local_vector_b = B23 @ B12 @ global_vector

global_vector_1 = np.linalg.inv(A12) @ np.linalg.inv(A23) @ np.linalg.inv(A34) @ np.linalg.inv(A45) @ local_vector
global_vector_2 = np.linalg.inv(B12) @ np.linalg.inv(B23) @ local_vector_b

transformation_test = transformations_matrix(0, 0, -np.pi / 2, -4, 6, 0)

print(np.linalg.inv(transformation_test))
print(np.linalg.inv(transformation_test) @ np.array([-5, 2, -12, 1]))
