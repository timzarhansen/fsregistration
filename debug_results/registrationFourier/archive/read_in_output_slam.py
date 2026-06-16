import json
import numpy as np
import matplotlib.pyplot as plt


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

plt.figure(1)
plt.plot(pointcloud1[:, 0], pointcloud1[:, 1], ".")

plt.figure(2)
plt.plot(pointcloud2[:, 0], pointcloud2[:, 1], ".")
