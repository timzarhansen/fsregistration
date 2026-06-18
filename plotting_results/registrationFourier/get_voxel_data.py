import numpy as np


def get_voxel_data(number_of_points, shift, rotation, point_cloud, from_to):
    voxel_data = np.zeros((number_of_points, number_of_points, number_of_points))

    for j in range(len(point_cloud.points)):
        position_point = np.array([
            point_cloud.points[j, 0] + shift[0],
            point_cloud.points[j, 1] + shift[1],
            point_cloud.points[j, 2] + shift[2]
        ])
        position_point = rotation @ position_point
        x_index = int((position_point[0] + from_to) / (from_to * 2) * number_of_points)
        y_index = int((position_point[1] + from_to) / (from_to * 2) * number_of_points)
        z_index = int((position_point[2] + from_to) / (from_to * 2) * number_of_points)
        voxel_data[y_index, x_index, z_index] = 1

    return voxel_data
