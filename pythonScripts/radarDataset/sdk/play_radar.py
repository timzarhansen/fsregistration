################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Dan Barnes (dbarnes@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################

import argparse
import os
from radar import load_radar, radar_polar_to_cartesian
import numpy as np
import cv2
import csv
from scipy.spatial.transform import Rotation as R

np.set_printoptions(precision=5, suppress=True)
def matrix_to_transform(matrix):
  """Converts a 4x4 transformation matrix to x, y, z, roll, pitch, yaw."""
  translation = matrix[:3, 3]
  rotation = R.from_matrix(matrix[:3,:3])
  euler = rotation.as_euler('xyz', degrees=False)
  return np.asarray([translation[0], translation[1], translation[2], euler[0], euler[1], euler[2]])

def find_row_indices(data, target_value, column_index):
    """Finds the indices of rows containing a specific value in a given column."""
    indices = []
    for i, row in enumerate(data):
        try:
            if float(row[column_index]) == target_value:
                indices.append(i)
        except ValueError:
            pass  # Handle non-numeric values

    return indices


parser = argparse.ArgumentParser(description='Play back radar data from a given directory')

parser.add_argument('dir', type=str, help='Directory containing radar data.')

args = parser.parse_args()

timestamps_path = os.path.join(os.path.join(args.dir, os.pardir, 'radar.timestamps'))
gt_paths = os.path.join(os.path.join(args.dir, os.pardir, 'gt/radar_odometry.csv'))
if not os.path.isfile(timestamps_path):
    raise IOError("Could not find timestamps file")
if not os.path.isfile(gt_paths):
    raise IOError("Could not find GT file")


# Cartesian Visualsation Setup
# Resolution of the cartesian form of the radar scan in metres per pixel

# Cartesian visualisation size (used for both height and width)
cart_pixel_width = 128  # pixels
cart_resolution_Int = 100
path_data_saved = "../dataOutput/"+str(cart_pixel_width)+"_"+str(cart_resolution_Int)+"/"
try:
    os.mkdir(path_data_saved)
    print(f"Folder '{path_data_saved}' created successfully.")
except FileExistsError:
    print(f"Folder '{path_data_saved}' already exists.")

cart_resolution = float(cart_resolution_Int)/100.0
interpolate_crossover = True

title = "Radar Visualisation Example"

radar_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)

with open(path_data_saved+"GT_Data.csv", 'a', newline='') as resultingGtFile:
    writer = csv.writer(resultingGtFile)
    with open(gt_paths, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        header = next(reader)  # Skip header row
        gt_data = list(reader)
        timestampStart = radar_timestamps[0]
        startingRowIndex = find_row_indices(gt_data,timestampStart, 9)
        currentGtRow = gt_data[startingRowIndex[0]]

        x, y, z, roll, pitch, yaw = float(currentGtRow[2]), float(currentGtRow[3]), float(currentGtRow[4]), float(
            currentGtRow[5]), float(
            currentGtRow[6]), float(currentGtRow[7])
        transformation_matrix = np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])

        r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
        rotation_matrix = r.as_matrix()

        # Combine translation and rotation (order matters!)
        transformation_matrix[:3, :3] = rotation_matrix
        absoluteTransformation = np.linalg.inv(transformation_matrix)

        for index , radar_timestamp in enumerate(radar_timestamps):
            currentGtRow = gt_data[startingRowIndex[0]+index]

            x, y, z, roll, pitch, yaw = float(currentGtRow[2]), float(currentGtRow[3]), float(currentGtRow[4]), float(currentGtRow[5]), float(
                currentGtRow[6]), float(currentGtRow[7])

            # Create transformation matrix (simplified - assumes no scaling/shearing)
            transformation_matrix = np.array([
                [1, 0, 0, x],
                [0, 1, 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1]
            ])

            r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
            rotation_matrix = r.as_matrix()

            # Combine translation and rotation (order matters!)
            transformation_matrix[:3, :3] = rotation_matrix
            absoluteTransformation = np.matmul(transformation_matrix,absoluteTransformation)
            print(radar_timestamp)
            print(absoluteTransformation)
            print("\n")
            x_gt, y_gt, z_gt, roll_gt, pitch_gt, yaw_gt = matrix_to_transform(absoluteTransformation)
            writer.writerow([radar_timestamp,x_gt, y_gt,yaw_gt])


            filename = os.path.join(args.dir, str(radar_timestamp) + '.png')

            if not os.path.isfile(filename):
                raise FileNotFoundError("Could not find radar example: {}".format(filename))

            timestamps, azimuths, valid, fft_data, radar_resolution = load_radar(filename)
            cart_img = radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width,
                                                interpolate_crossover)

            # Combine polar and cartesian for visualisation
            # The raw polar data is resized to the height of the cartesian representation
            downsample_rate = 4
            fft_data_vis = fft_data[:, ::downsample_rate]
            resize_factor = float(cart_img.shape[0]) / float(fft_data_vis.shape[0])
            fft_data_vis = cv2.resize(fft_data_vis, (0, 0), None, resize_factor, resize_factor)
            vis = cv2.hconcat((fft_data_vis, fft_data_vis[:, :10] * 0 + 1, cart_img))
            cv2.imwrite(path_data_saved+str(radar_timestamp)+"_radar.png", cart_img*255.0)
            # cv2.imshow(title, cart_img)  # The data is doubled to improve visualisation
            # cv2.waitKey(1)
