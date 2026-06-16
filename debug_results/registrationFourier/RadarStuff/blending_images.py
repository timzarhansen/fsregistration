import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Multi-image blending with pose (grayscale-safe version)
# clc, clear, clf - skipped

folder_name = "512_25_1"
image_sizes = 512
how_many_images = 3
size_pixel = 0.25
skips = 1
gt_matching_pose_list = pd.read_csv(f"{folder_name}/gtPoseScanMatching.csv").values

# List of images
image_files = []
for i in range(how_many_images + 1):
    image_files.append(f"{folder_name}/original_image_{i*skips:04d}.png")

absolute_poses = np.zeros((how_many_images + 1, 3))

for i in range(2, how_many_images + 1):
    x = gt_matching_pose_list[i - 1, 0] / size_pixel
    y = gt_matching_pose_list[i - 1, 1] / size_pixel
    yaw = gt_matching_pose_list[i - 1, 2]
    x_old = absolute_poses[i - 1, 0]
    y_old = absolute_poses[i - 1, 1]
    yaw_old = absolute_poses[i - 1, 2]
    T_old = np.array([
        [np.cos(yaw_old), -np.sin(yaw_old), x_old],
        [np.sin(yaw_old), np.cos(yaw_old), y_old],
        [0, 0, 1]
    ])

    T_new = np.array([
        [np.cos(yaw), -np.sin(yaw), x],
        [np.sin(yaw), np.cos(yaw), y],
        [0, 0, 1]
    ])
    absolute_transformations = T_old @ T_new
    x_new = absolute_transformations[0, 2]
    y_new = absolute_transformations[1, 2]
    yaw_new = np.arctan2(absolute_transformations[1, 0], absolute_transformations[0, 0])
    absolute_poses[i, :] = [x_new, y_new, yaw_new]

# Green points to overlay for each image
green_points = np.ones((how_many_images + 1, 2)) * image_sizes / 2

# Load reference image
I_ref = np.array(Image.open(image_files[0]))
if len(I_ref.shape) == 2:
    I_ref = np.repeat(I_ref[:, :, np.newaxis], 3, axis=2)

# Identity transform for the first image
tforms = [None] * len(image_files)
tforms[0] = np.eye(3)

warped_images = [None] * len(image_files)
warped_points = [None] * len(image_files)

warped_images[0] = I_ref
warped_points[0] = green_points[0, :]

# Build transforms for the rest
for k in range(1, len(image_files)):
    I = np.array(Image.open(image_files[k]))
    if len(I.shape) == 2:
        I = np.repeat(I[:, :, np.newaxis], 3, axis=2)

    dx = absolute_poses[k, 0]
    dy = absolute_poses[k, 1]
    yaw = np.deg2rad(absolute_poses[k, 2])

    T = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [dx, dy, 1]
    ])
    tforms[k] = T

# Warp all images
for k in range(len(image_files)):
    I = np.array(Image.open(image_files[k]))
    if len(I.shape) == 2:
        I = np.repeat(I[:, :, np.newaxis], 3, axis=2)

    if k == 0:
        I_warp = I
    else:
        T = tforms[k]
        M = T[:2, :]
        I_warp = cv2.warpAffine(I, M, (I.shape[1], I.shape[0]))

    if len(I_warp.shape) == 2:
        I_warp = np.repeat(I_warp[:, :, np.newaxis], 3, axis=2)

    warped_images[k] = I_warp

    pts = green_points[k, :]
    pts_homogeneous = np.array([pts[0], pts[1], 1])
    pts_trans_homogeneous = tforms[k] @ pts_homogeneous
    pts_trans = pts_trans_homogeneous[:2]
    warped_points[k] = pts_trans

# Blend images
blend = np.zeros_like(warped_images[0], dtype=np.float32)
count = np.zeros(warped_images[0].shape[:2], dtype=np.float32)

for k in range(len(warped_images)):
    I = warped_images[k]

    mask = np.any(I > 0, axis=2)

    for c in range(3):
        channel = blend[:, :, c]
        temp = I[:, :, c]
        channel[mask] = channel[mask] + temp[mask]
        blend[:, :, c] = channel

    count[mask] = count[mask] + 1

count[count == 0] = 1
for c in range(3):
    blend[:, :, c] = blend[:, :, c] / count

# Display result
plt.figure(1)
plt.imshow(blend.astype(np.uint8))
plt.title('Blended Image with Green Points')
plt.hold(True)

for k in range(len(warped_points)):
    plt.plot(warped_points[k][0], warped_points[k][1], 'g.', markersize=15)

plt.hold(False)
