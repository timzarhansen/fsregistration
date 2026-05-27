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


from sdk.radar import load_radar, radar_polar_to_cartesian
from pyboreas import BoreasDataset
import csv
from scipy.spatial.transform import Rotation as R
import os, argparse, sys

import cv2
import numpy as np
# import open3d as o3d
from pathlib import Path

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_script_dir))))
_install_lib = os.path.join(_root_dir, 'install', 'fsregistration', 'lib', 'fsregistration')
if os.path.isdir(_install_lib):
    sys.path.insert(0, _install_lib)

from pybind_registration_2d import SoftRegistrationWrapper2D



def fuse_images(imagesOverTime, estimatedTransformations):
    """
    Fuses multiple images using their absolute transformation matrices.

    Args:
        imagesOverTime (list): List of OpenCV images in BGR format.
        estimatedTransformations (list): List of transformation matrices (3x3 for Homography / 2x3 for Affine)

    Returns:
        numpy.ndarray: Fused image as a BGR array
    """
    # Sanity checks
    assert len(imagesOverTime) == len(estimatedTransformations), "Image and transform lists must be same length"

    if not imagesOverTime:
        return None

    # Calculate global canvas boundaries from all transformations applied to corners
    min_x = float('inf')
    max_x = -float('inf')
    min_y = float('inf')
    max_y = -float('inf')

    for i in range(len(imagesOverTime)):
        img = imagesOverTime[i]
        tmat = get_affine_matrix(estimatedTransformations[i])

        h, w = img.shape[:2]

        # Get image corners (reshaped correctly)
        corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float64).reshape(-1, 1, 2)  # Reshape to Nx1x2

        if tmat.shape == (3,3):  # Homography transform
            transformed_corners = cv2.perspectiveTransform(corners, tmat)
        else:  # Affine transform (2x3 matrix)
            transformed_corners = cv2.transform(corners, tmat)  # Already Nx1x2

        transformed_corners = transformed_corners.squeeze()  # Convert to Nx2

        # Update global min/max coordinates
        current_min_x = np.min(transformed_corners[:,0])
        current_max_x = np.max(transformed_corners[:,0])
        current_min_y = np.min(transformed_corners[:,1])
        current_max_y = np.max(transformed_corners[:,1])

        if current_min_x < min_x:
            min_x = int(current_min_x)
        if current_max_x > max_x:
            max_x = int(current_max_x)
        if current_min_y < min_y:
            min_y = int(current_min_y)
        if current_max_y > max_y:
            max_y = int(current_max_y)

    # Calculate canvas dimensions and shift offset to make everything positive
    canvas_width = (max_x - min_x) + 1
    canvas_height = (max_y - min_y) + 1
    tx, ty = -min_x, -min_y

    adjusted_transforms = []
    for tmat in estimatedTransformations:
        tmat_corrected = get_affine_matrix(tmat)
        if tmat_corrected.shape == (3,3):  # Homography case
            T_trans = np.eye(3, dtype=np.float64)
            T_trans[0,2] = tx
            T_trans[1,2] = ty
            adjusted_T = T_trans @ tmat_corrected
        else:  # Affine case (2x3 matrix)
            adjusted_T = tmat_corrected.copy()
            adjusted_T[:,2] += [tx, ty]

        adjusted_transforms.append(adjusted_T)

    # Warp all images into the common coordinate system
    warped_images = []
    for i in range(len(imagesOverTime)):
        img = imagesOverTime[i]
        adj_t = adjusted_transforms[i]

        if adj_t.shape == (3,3):
            warped = cv2.warpPerspective(img, adj_t, (canvas_width, canvas_height))
        else:
            # Reshape affine matrix to 2x3 before using
            warped = cv2.warpAffine(img, adj_t[:2], (canvas_width, canvas_height))  # Take top-left 2x3 part

        warped_images.append(warped)

    # Fuse images using average
    if warped_images:
        stacked = np.stack(warped_images).astype(np.float64)
        fused_image = np.mean(stacked, axis=0).astype(np.uint8)
    else:
        fused_image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    return fused_image

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

def get_image_from_path(filename,cart_resolution,cart_pixel_width,interpolate_crossover):

    if not os.path.isfile(filename):
        raise FileNotFoundError("Could not find radar example: {}".format(filename))

    timestamps, azimuths, valid, fft_data, radar_resolution = load_radar(filename)
    cart_img = radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_pixel_width, cart_resolution,
                                        interpolate_crossover)

    # Combine polar and cartesian for visualisation
    # The raw polar data is resized to the height of the cartesian representation
    downsample_rate = 4
    fft_data_vis = fft_data[:, ::downsample_rate]
    resize_factor = float(cart_img.shape[0]) / float(fft_data_vis.shape[0])
    fft_data_vis = cv2.resize(fft_data_vis, (0, 0), None, resize_factor, resize_factor)
    vis = cv2.hconcat((fft_data_vis, fft_data_vis[:, :10] * 0 + 1, cart_img))
    return cart_img

def normalize_image(image):
    normalized_array = image.astype(np.float64).reshape(-1)
    return normalized_array

def get_transformation_based_on_gt_row(gt_row):
    x, y, z, roll, pitch, yaw = float(gt_row[2]), float(gt_row[3]), float(gt_row[4]), float(gt_row[5]), float(
        gt_row[6]), float(gt_row[7])

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
    return transformation_matrix

def get_affine_matrix(inputMatrix):
    inputMatrix = np.linalg.inv(inputMatrix)
    perspectiveMatrixReturn = np.eye(3)
    perspectiveMatrixReturn[:2,:2]= inputMatrix[:2,:2]
    perspectiveMatrixReturn[0,2]= -inputMatrix[1,3]
    perspectiveMatrixReturn[1,2]= inputMatrix[0,3]


    # perspectiveMatrixReturn[:2,:2]= np.linalg.inv(inputMatrix[:2,:2])
    # perspectiveMatrixReturn[:2,:2]= inputMatrix[:2,:2]
    # perspectiveMatrixReturn[0,2]= -inputMatrix[1,3]
    # perspectiveMatrixReturn[1,2]= -inputMatrix[0,3]
    return perspectiveMatrixReturn

def transform_diff(matrix1, matrix2):
    """Computes translation and rotation difference (angle) between two 3x3 transformation matrices."""

    translation1 = np.array([matrix1[0, 2], matrix1[1, 2]])
    translation2 = np.array([matrix2[0, 2], matrix2[1, 2]])
    translation_diff = translation2 - translation1

    rotation1 = matrix1[:2,:2]
    rotation2 = matrix2[:2,:2]
    angle_diff = np.degrees(np.arctan2(rotation2[1,0], rotation2[0,0]) - np.arctan2(rotation1[1,0], rotation1[0,0]))

    return translation_diff, angle_diff

def getxyYaw(matrix1):
    x = matrix1[0, 2]
    y = matrix1[1, 2]
    yaw = np.degrees(np.arctan2(matrix1[1,0], matrix1[0,0]))
    return x,y,yaw

# Config:
# 32 1 4 28 0.001 0.001 1
if __name__ == '__main__':
    np.set_printoptions(precision=5, suppress=True)
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int, help='Path to the config file.')
    parser.add_argument('use_clahe', type=int)
    parser.add_argument('potential_for_necessary_peak', type=float, help='Path to the config file.')
    parser.add_argument('size_of_pixel', type=float, help='Path to the config file.')
    parser.add_argument('matching_every_nth_image', type=int, help='just Match Every Nth .')
    parser.add_argument('sequence', type=int, help='Which sequence of the data.')
    parser.add_argument('dir', type=str, help='Directory containing radar data.')


    args = parser.parse_args()
    N = args.N  # 32 64 128 256
    use_clahe = bool(args.use_clahe)  # True False
    size_of_pixel = float(args.size_of_pixel)
    potential_for_necessary_peak = float(args.potential_for_necessary_peak)  # 0.01 , 0.001
    radar_file_path = str(args.dir)
    sequence_number = args.sequence
    matching_every_nth_image = args.matching_every_nth_image
    print(args)

    reg = SoftRegistrationWrapper2D(N)

    bd = BoreasDataset(radar_file_path, split=None, verbose=True)
    seq = bd.sequences[sequence_number]# this is which sequence number is chosen
    # seq.calib.print_calibration()
    length_of_radar_scans = len(seq.radar_frames)
    print(length_of_radar_scans)

    pathToSave = f"saveRandomImagesBoreas/{sequence_number:02d}_{N:03d}_{int(size_of_pixel*100)}_{matching_every_nth_image}/"
    print(pathToSave)
    Path(pathToSave).mkdir(parents=True, exist_ok=True)
    # radar_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)


    absoluteTransformationList = []
    imagesOverTimeList = []
    estimatedTransformationList = []
    absoluteTransformationListForImagePrint = []
    rotationErrorList = []
    translationErrorList = []
    gtPoseList = []
    estPoseList = []
    #getting first image

    firstRadarFrame = seq.get_radar(0)
    filename = os.path.join(args.dir, str(firstRadarFrame.timestamp_micro) + '.png')
    firstRadarImage = firstRadarFrame.polar_to_cart(
    cart_resolution=size_of_pixel,
    cart_pixel_width=N,
    in_place=False,)

    imagesOverTimeList.append(firstRadarImage*255.0)
    estimatedTransformationList.append(np.eye(4))
    absoluteTransformationListForImagePrint.append(np.eye(4,dtype=np.float64))
    firstIndex = 0
    cv2.imwrite(pathToSave + f"original_image_{firstIndex:04d}.png", firstRadarImage * 255.0)
    for index in range(1,length_of_radar_scans):

        currentRadarFrame = seq.get_radar(index)
        radar_timestamp = currentRadarFrame.timestamp_micro
        # if index > 200:
        #     break
        #compute absolute Position
        # currentGtRow = gt_data[startingRowIndex[0]+index]
        # transformation_matrix_gt = get_transformation_based_on_gt_row(currentGtRow)

        # absoluteTransformationList.append(currentRadarFrame.pose)

        if index>matching_every_nth_image-1:
            if index % matching_every_nth_image==0:

                transformation_matrix_gt = np.matmul(np.linalg.inv(np.asarray(seq.get_radar(index-matching_every_nth_image).pose,dtype=np.float64)),np.asarray(currentRadarFrame.pose,dtype=np.float64))

                absoluteTransformationList.append(currentRadarFrame.pose)

                gtPoseList.append(transformation_matrix_gt)

                filename1 = os.path.join(args.dir, str(seq.get_radar(index-matching_every_nth_image).timestamp_micro) + '.png')
                cart_img1 = seq.get_radar(index-matching_every_nth_image).polar_to_cart(cart_resolution=size_of_pixel,cart_pixel_width=N,in_place=False)
                filename2 = os.path.join(args.dir, str(radar_timestamp) + '.png')
                cart_img2 = currentRadarFrame.polar_to_cart(cart_resolution=size_of_pixel,cart_pixel_width=N,in_place=False)
                cv2.imwrite(pathToSave + f"original_image_{index:04d}.png", cart_img2 * 255.0)
                imagesOverTimeList.append(cart_img2*255.0)
                image_1 = normalize_image(cart_img1)
                image_2 = normalize_image(cart_img2)

                listPeaks = reg.register_all_solutions(
                    image_1, image_2,
                    cellSize=size_of_pixel,
                    useGauss=False,
                    debug=False,
                    potentialNecessaryForPeak=potential_for_necessary_peak,
                    multipleRadii=True,
                    useClahe=use_clahe,
                    useHamming=True
                )

                highestPeak = 0.0
                indexHighestPeak = 0

                for index_solutions, peak in enumerate(listPeaks):
                    # find peak
                    if peak.potentialTranslations[0].peakHeight > highestPeak:
                        highestPeak = peak.potentialTranslations[0].peakHeight
                        indexHighestPeak = index_solutions

                peak = listPeaks[indexHighestPeak]

                np.set_printoptions(suppress=True)

                resultingTransformation = np.eye(4)
                yaw = peak.potentialRotation.angle
                resultingTransformation[:3, :3] = R.from_euler('z', yaw).as_matrix()
                tx, ty = peak.potentialTranslations[0].translationSI
                resultingTransformation[:3, 3] = [tx, ty, 0.0]
                estimatedTransformationList.append(np.matmul(estimatedTransformationList[-1],resultingTransformation))
                absoluteTransformationListForImagePrint.append(np.matmul(absoluteTransformationListForImagePrint[-1],transformation_matrix_gt))

                perspectiveMatrixGT = get_affine_matrix(transformation_matrix_gt)
                perspectiveMatrixEstimated = get_affine_matrix(resultingTransformation)


                print("resultingTransformation:")
                print(perspectiveMatrixEstimated)
                warped_img_est_2 = cv2.warpPerspective(cart_img2, perspectiveMatrixEstimated, (cart_img1.shape[1], cart_img1.shape[0]))
                blended_img_est = cv2.addWeighted(cart_img1, 0.5, warped_img_est_2, 0.5, 0)
                cv2.imwrite(pathToSave+f"blended_{index:04d}_est.png", blended_img_est*255.0)


                print("transformation_matrix_gt:")
                print(perspectiveMatrixGT)
                warped_img_gt2 = cv2.warpPerspective(cart_img2, perspectiveMatrixGT, (cart_img1.shape[1], cart_img1.shape[0]))
                blended_img_gt = cv2.addWeighted(cart_img1, 0.5, warped_img_gt2, 0.5, 0)
                cv2.imwrite(pathToSave+f"blended_{index:04d}_gt.png", blended_img_gt*255.0)

                trans_error, rot_error = transform_diff(perspectiveMatrixGT,perspectiveMatrixEstimated)
                # gtPoseList.append(getxyYaw(perspectiveMatrixGT))
                estPoseList.append(resultingTransformation)
                rotationErrorList.append(rot_error)
                translationErrorList.append(trans_error)
                print(index)
                print("next:")
                # now we have to also compute the GT
    #here are
    print(len(estimatedTransformationList))
    print(len(imagesOverTimeList))
    print(len(absoluteTransformationList))
    print(len(absoluteTransformationListForImagePrint))
    print("done")
    # fused = fuse_images(imagesOverTimeList,absoluteTransformationListForImagePrint)

    # cv2.imwrite(pathToSave+f"blended_fullMapGT.png", fused*5)
    # fused = fuse_images(imagesOverTimeList,estimatedTransformationList)

    # cv2.imwrite(pathToSave+f"blended_fullMapEstimated.png", fused*5)

    with open(pathToSave+f"rotationErrorList.csv", "w") as f:
        for item in rotationErrorList:
            f.write(str(item) + "\n")

    with open(pathToSave+f"translationErrorList.csv", "w") as f:
        for item in translationErrorList:
            f.write(str(item) + "\n")

    with open(pathToSave+f"gtPoseScanMatching.csv", "w", newline='') as f:
        writer = csv.writer(f)
        for tup in gtPoseList:
            x, y, z, roll, pitch, yaw = matrix_to_transform(tup)
            writer.writerow([x, y, yaw])

    with open(pathToSave+f"estPoseScanMatching.csv", "w", newline='') as f:
        writer = csv.writer(f)
        for tup in estPoseList:
            x, y, z, roll, pitch, yaw = matrix_to_transform(tup)
            writer.writerow([x, y, yaw])

    with open(pathToSave+f"absolutePoseListGT.csv", "w", newline='') as f:
        writer = csv.writer(f)
        for tup in absoluteTransformationList:
            x,y,z,roll,pitch,yaw = matrix_to_transform(tup)
            writer.writerow([x,y,yaw])
