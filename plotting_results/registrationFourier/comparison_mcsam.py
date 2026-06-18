import numpy as np
import matplotlib.pyplot as plt

current_number = 21

motion_compensation_image = np.genfromtxt(f"csvFiles/IROSResults/imagesComparison/motionCompensation{current_number}.csv", delimiter=",")
slam_compensation_image = np.genfromtxt(f"csvFiles/IROSResults/imagesComparison/slamCompensation{current_number}.csv", delimiter=",")

n = int(np.sqrt(motion_compensation_image.shape[0]))

motion_compensation_image_plot = np.zeros((n, n))
slam_compensation_image_plot = np.zeros((n, n))

for j in range(n):
    for i in range(n):
        motion_compensation_image_plot[i, j] = motion_compensation_image[(i - 1) * n + j]
        slam_compensation_image_plot[i, j] = slam_compensation_image[(i - 1) * n + j]

plt.figure(1)
plt.imshow(motion_compensation_image_plot, aspect='equal')

plt.figure(2)
plt.imshow(slam_compensation_image_plot, aspect='equal')
