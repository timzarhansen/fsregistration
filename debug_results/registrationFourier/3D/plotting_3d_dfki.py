import numpy as np
import matplotlib.pyplot as plt
import json


def transformations_matrix(roll, pitch, yaw, x, y, z):
    R_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(roll), -np.sin(roll), 0],
        [0, np.sin(roll), np.cos(roll), 0],
        [0, 0, 0, 1]
    ])
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch), 0],
        [0, 1, 0, 0],
        [-np.sin(pitch), 0, np.cos(pitch), 0],
        [0, 0, 0, 1]
    ])
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0, 0],
        [np.sin(yaw), np.cos(yaw), 0, 0],
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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mid_slice = volume_3d.shape[0] // 2
    x, y = np.meshgrid(np.arange(volume_3d.shape[1]), np.arange(volume_3d.shape[2]))
    ax.plot_surface(x, y, volume_3d[mid_slice, :, :], cmap='viridis')
    plt.show()


with open('csvFiles/fullGraphDataset.csv', 'r') as f:
    data_set = json.load(f)

# plotting 2D map First
dimension_map = 256
map_2d = np.zeros((dimension_map, dimension_map))
map_index = np.zeros((dimension_map, dimension_map))
size_of_map = 35

for i in range(len(data_set['keyFrames']) - 30000):
    if data_set['keyFrames'][i]['intensityValues']['type'] != 5:
        x_pos = data_set['keyFrames'][i]['position']['x']
        y_pos = data_set['keyFrames'][i]['position']['y']
        z_pos = data_set['keyFrames'][i]['position']['z']
        roll = data_set['keyFrames'][i]['position']['roll']
        pitch = data_set['keyFrames'][i]['position']['pitch']
        yaw = data_set['keyFrames'][i]['position']['yaw']
        pose_sonar = transformations_matrix(roll, pitch, yaw, x_pos, y_pos, z_pos)
        local_rotation_intensity_ray = transformations_matrix(0, 0, data_set['keyFrames'][i]['intensityValues']['angle'], 0, 0, 0)

        ignore_distance_index = int(round((0.5 / data_set['keyFrames'][i]['intensityValues']['range']) * len(data_set['keyFrames'][i]['intensityValues']['intensity'])))

        for j in range(ignore_distance_index, len(data_set['keyFrames'][i]['intensityValues']['intensity'])):
            distance_of_intensity = j / len(data_set['keyFrames'][i]['intensityValues']['intensity']) * data_set['keyFrames'][i]['intensityValues']['range']
            increment_of_scan = 1

            for increment_tmp in range(-increment_of_scan - 5, increment_of_scan + 6):
                position_of_intensity = np.array([distance_of_intensity, 0, 0, 1])
                rotation_of_point = increment_tmp / 400.0
                rotation_for_better_view = transformations_matrix(0, 0, rotation_of_point, 0, 0, 0)
                position_of_intensity = rotation_for_better_view @ position_of_intensity
                position_of_intensity = pose_sonar @ local_rotation_intensity_ray @ position_of_intensity
                index_x = int(round(position_of_intensity[0] / (size_of_map / 2) * dimension_map / 2 + dimension_map / 2)) + 1
                index_y = int(round(position_of_intensity[1] / (size_of_map / 2) * dimension_map / 2 + dimension_map / 2)) + 1
                if 0 < index_x <= dimension_map and 0 < index_y <= dimension_map:
                    map_2d[index_x - 1, index_y - 1] = map_2d[index_x - 1, index_y - 1] + data_set['keyFrames'][i]['intensityValues']['intensity'][j]
                    map_index[index_x - 1, index_y - 1] = map_index[index_x - 1, index_y - 1] + 1

for i in range(dimension_map):
    for j in range(dimension_map):
        if map_index[i, j] > 0:
            map_2d[i, j] = map_2d[i, j] / map_index[i, j]

plt.imshow(map_2d)

# 3D map
dimension_3d_map = 128
map_3d = np.zeros((dimension_3d_map, dimension_3d_map, dimension_3d_map))
map_3d_index = np.zeros((dimension_3d_map, dimension_3d_map, dimension_3d_map))
size_of_map = 35

overall_mean = 0
number_of_mean_elements = 0

for i in range(len(data_set['keyFrames']) - 100):
    if data_set['keyFrames'][i]['intensityValues']['type'] == 5:
        current_angle = data_set['keyFrames'][i]['intensityValues']['angle']
        current_angle = np.mod(current_angle + 2 * np.pi, 2 * np.pi)
        if current_angle > np.pi:
            current_angle = current_angle - 2 * np.pi
        if abs(abs(current_angle) - np.pi) < 1.0:
            x_pos = data_set['keyFrames'][i]['position']['x']
            y_pos = data_set['keyFrames'][i]['position']['y']
            z_pos = data_set['keyFrames'][i]['position']['z']
            roll = data_set['keyFrames'][i]['position']['roll']
            pitch = data_set['keyFrames'][i]['position']['pitch']
            yaw = data_set['keyFrames'][i]['position']['yaw']
            pose_sonar = transformations_matrix(roll, pitch, yaw, x_pos, y_pos, z_pos)
            angle_of_sonar = data_set['keyFrames'][i]['intensityValues']['angle']
            angle_of_sonar_matrix = transformations_matrix(angle_of_sonar + np.pi, 0, 0, 0, 0, 0)
            pos_diff_sonar_matrix = transformations_matrix(0, 0, 0, 0.4, 0, 0)

            ignore_distance_index = int(round((1.0 / data_set['keyFrames'][i]['intensityValues']['range']) * len(data_set['keyFrames'][i]['intensityValues']['intensity'])))

            added_sum = 0
            for j in range(ignore_distance_index, len(data_set['keyFrames'][i]['intensityValues']['intensity'])):
                distance_of_intensity = j / len(data_set['keyFrames'][i]['intensityValues']['intensity']) * data_set['keyFrames'][i]['intensityValues']['range']
                added_sum = added_sum + data_set['keyFrames'][i]['intensityValues']['intensity'][j]
                if added_sum < 500 and distance_of_intensity < 5:
                    for increment_tmp in range(-5, 6):
                        for increment_other_tmp in range(-60, 61):
                            position_of_intensity = np.array([0, 0, distance_of_intensity, 1])
                            rotation_of_point = increment_tmp / 800.0
                            rotation_for_better_view = transformations_matrix(rotation_of_point, 0, 0, 0, 0, 0)
                            position_of_intensity = pos_diff_sonar_matrix @ transformations_matrix(0, increment_other_tmp / 720.0, 0, 0, 0, 0) @ angle_of_sonar_matrix @ rotation_for_better_view @ position_of_intensity
                            position_of_intensity = pose_sonar @ position_of_intensity
                            index_x = int(round(position_of_intensity[0] / (size_of_map / 2) * dimension_3d_map / 2 + dimension_3d_map / 2)) + 1
                            index_y = int(round(position_of_intensity[1] / (size_of_map / 2) * dimension_3d_map / 2 + dimension_3d_map / 2)) + 1
                            index_z = int(round(position_of_intensity[2] / (size_of_map / 2) * dimension_3d_map / 2 + dimension_3d_map / 2)) + 1
                            if 0 < index_x <= dimension_3d_map and 0 < index_y <= dimension_3d_map and 0 < index_z <= dimension_3d_map:
                                map_3d[index_x - 1, index_y - 1, index_z - 1] = map_3d[index_x - 1, index_y - 1, index_z - 1] + data_set['keyFrames'][i]['intensityValues']['intensity'][j]
                                map_3d_index[index_x - 1, index_y - 1, index_z - 1] = map_3d_index[index_x - 1, index_y - 1, index_z - 1] + 1
            number_of_mean_elements = number_of_mean_elements + 1
            overall_mean = overall_mean + added_sum

    if data_set['keyFrames'][i]['intensityValues']['type'] != 5:
        x_pos = data_set['keyFrames'][i]['position']['x']
        y_pos = data_set['keyFrames'][i]['position']['y']
        z_pos = data_set['keyFrames'][i]['position']['z']
        roll = data_set['keyFrames'][i]['position']['roll']
        pitch = data_set['keyFrames'][i]['position']['pitch']
        yaw = data_set['keyFrames'][i]['position']['yaw']
        pose_sonar = transformations_matrix(roll, pitch, yaw, x_pos, y_pos, z_pos)
        local_rotation_intensity_ray = transformations_matrix(0, 0, data_set['keyFrames'][i]['intensityValues']['angle'], 0, 0, 0)

        ignore_distance_index = int(round((0.5 / data_set['keyFrames'][i]['intensityValues']['range']) * len(data_set['keyFrames'][i]['intensityValues']['intensity'])))

        for j in range(ignore_distance_index, len(data_set['keyFrames'][i]['intensityValues']['intensity'])):
            distance_of_intensity = j / len(data_set['keyFrames'][i]['intensityValues']['intensity']) * data_set['keyFrames'][i]['intensityValues']['range']
            increment_of_scan = 1

            for increment_tmp in range(-6, 7):
                for increment_other_tmp in range(-30, 31):
                    position_of_intensity = np.array([distance_of_intensity, 0, 0, 1])
                    rotation_of_point = increment_tmp / 400.0
                    rotation_for_better_view = transformations_matrix(0, 0, rotation_of_point, 0, 0, 0)
                    opening_angle_matrix = transformations_matrix(0, increment_other_tmp / 400.0, 0, 0, 0, 0)
                    position_of_intensity = rotation_for_better_view @ opening_angle_matrix @ position_of_intensity
                    position_of_intensity = pose_sonar @ local_rotation_intensity_ray @ position_of_intensity
                    index_x = int(round(position_of_intensity[0] / (size_of_map / 2) * dimension_3d_map / 2 + dimension_3d_map / 2)) + 1
                    index_y = int(round(position_of_intensity[1] / (size_of_map / 2) * dimension_3d_map / 2 + dimension_3d_map / 2)) + 1
                    if 0 < index_x <= dimension_3d_map and 0 < index_y <= dimension_3d_map:
                        map_3d[index_x - 1, index_y - 1, dimension_3d_map // 2] = map_3d[index_x - 1, index_y - 1, dimension_3d_map // 2] + data_set['keyFrames'][i]['intensityValues']['intensity'][j] * 3.5
                        map_3d_index[index_x - 1, index_y - 1, dimension_3d_map // 2] = map_3d_index[index_x - 1, index_y - 1, dimension_3d_map // 2] + 1

print(overall_mean / number_of_mean_elements)

for i in range(dimension_3d_map):
    for j in range(dimension_3d_map):
        for k in range(dimension_3d_map):
            if map_3d_index[i, j, k] > 0:
                map_3d[i, j, k] = map_3d[i, j, k] / map_3d_index[i, j, k]

volume_viewer(map_3d)
