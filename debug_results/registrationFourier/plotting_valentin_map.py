import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import subprocess

plt.figure(3)
plt.clf()

name_of_dataset = "_valentinKeller_dynamic_slam_4_0_"

input_position = np.genfromtxt("csvFiles/IROSResults/positionEstimationOverTime" + name_of_dataset + ".csv", delimiter=",")
input_position = input_position[0:-4*70, 0:4]

how_many_skip = 800
input_position = input_position[4*how_many_skip:, :]

first_matrix_slam = input_position[0:4, 0:4]

number_of_vertices = input_position.shape[0] // 4
slam_matrix = np.zeros((number_of_vertices, 4, 4))

for i in range(number_of_vertices):
    slam_matrix[i, :, :] = np.linalg.solve(first_matrix_slam, input_position[(i-1)*4+1:(i-1)*4+4, 0:4])

    x_pos_slam = slam_matrix[i, 0, 3]
    y_pos_slam = slam_matrix[i, 1, 3]
    angle_slam = np.arctan2(slam_matrix[i, 1, 0], slam_matrix[i, 0, 0])

magnitude_map_input = np.genfromtxt("csvFiles/IROSResults/currentMap" + name_of_dataset + ".csv", delimiter=",")
N = int(np.sqrt(magnitude_map_input.shape[0]))
magnitude_map = np.zeros((N, N))
for j in range(N):
    for i in range(N):
        magnitude_map[i, j] = magnitude_map_input[(i-1)*N+j]

rotation_angle_image = 10
rotationpath = -0.5

size_pixel = 45/512
pos_x_zero = 255
pos_y_zero = 265
start_number = 1
end_number = 512
image_shift_x = -60
image_shift_y = -35

test_image = magnitude_map

x_des_mid_point = 0.0
y_des_mid_point = 0.0
diff_x = 0 - x_des_mid_point
diff_y = 0 - y_des_mid_point
diff_x = diff_x / size_pixel
diff_y = diff_y / size_pixel

test_image = rotate(test_image, rotation_angle_image, reshape=False)
test_image = np.roll(test_image, int(round(diff_x)) + image_shift_x, axis=1)
test_image = np.roll(test_image, int(round(diff_y)) + image_shift_y, axis=0)
test_image = test_image[start_number:end_number, int(start_number*1.1):int(end_number*0.9)]

quater_degree_matrix = np.eye(4)
quater_degree_matrix[0:3, 0:3] = rotation_matrix(0, 0, rotationpath) @ rotation_matrix(0, np.pi, 0) @ rotation_matrix(0, 0, np.pi/2)

picture_slam_plot_x = np.zeros(number_of_vertices)
picture_slam_plot_y = np.zeros(number_of_vertices)

for i in range(number_of_vertices):
    tmp_matrix_slam = quater_degree_matrix @ slam_matrix[i, :, :]

    picture_slam_plot_x[i] = tmp_matrix_slam[0, 3]
    picture_slam_plot_y[i] = tmp_matrix_slam[1, 3]

    picture_slam_plot_x[i] = picture_slam_plot_x[i] / size_pixel + pos_x_zero
    picture_slam_plot_y[i] = picture_slam_plot_y[i] / size_pixel + pos_y_zero

remove_last_rows = 300
picture_slam_plot_x = picture_slam_plot_x[0:-remove_last_rows]
picture_slam_plot_y = picture_slam_plot_y[0:-remove_last_rows]

plt.clf()
plt.hold(True)

plt.imshow(test_image, aspect='equal')
plt.plot(picture_slam_plot_x, picture_slam_plot_y, color='g')
plt.axis('equal')
plt.axis('image')

plt.xlim([1, test_image.shape[1]])
plt.ylim([1, test_image.shape[0]])

ax = plt.gca()
tmp = ax.get_xticks()
ax.set_xticklabels([str(round(t * size_pixel)) for t in tmp])
tmp = ax.get_yticks()
ax.set_yticklabels([str(round(t * size_pixel)) for t in tmp])

plt.ylabel("m")
plt.xlabel("m")
plt.rcParams['text.usetex'] = True

name_of_pdf_file = '/home/ws/matlab/registrationFourier/csvFiles/IROSResults/images/' + name_of_dataset
plt.savefig(name_of_pdf_file, format='pdf')

subprocess.run(['pdfcrop', name_of_pdf_file + '.pdf', name_of_pdf_file + '.pdf'])
