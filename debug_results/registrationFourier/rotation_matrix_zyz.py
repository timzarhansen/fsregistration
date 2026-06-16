import numpy as np


def rotation_matrix_zyz(r1, r2, r3):
    rotation = np.array([
        [np.cos(r1) * np.cos(r2) * np.cos(r3) - np.sin(r1) * np.sin(r3), -np.cos(r3) * np.sin(r1) - np.cos(r1) * np.cos(r2) * np.sin(r3), np.cos(r1) * np.sin(r2)],
        [np.cos(r1) * np.sin(r3) + np.cos(r2) * np.cos(r3) * np.sin(r1), np.cos(r1) * np.cos(r3) - np.cos(r2) * np.sin(r1) * np.sin(r3), np.sin(r1) * np.sin(r2)],
        [-np.cos(r3) * np.sin(r2), np.sin(r2) * np.sin(r3), np.cos(r2)]
    ])
    return rotation
