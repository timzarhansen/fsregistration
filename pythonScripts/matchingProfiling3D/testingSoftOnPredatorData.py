#!/usr/bin/python3
"""
Testing script for soft registration on Predator dataset.
Uses the C++ softRegistrationClass3D directly via pybind11 (no ROS2 needed).

Usage:
    python testingSoftOnPredatorData.py configFiles/predatorNothing.yaml 128 0 16 48 0.001 0.001 2 high train --start-index 0 --end-index 99
"""

import os
import sys
import torch
import json
import argparse
import shutil
import csv
import gc
import time
import numpy as np
import open3d as o3d
import copy
import transforms3d.quaternions as quat
import transforms3d.euler as euler

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
fsregistration_src = os.path.join(root_dir, 'src')

sys.path.insert(0, fsregistration_src)

from pybind_registration_3d import SoftRegistrationWrapper

from dataloader_utils import PredatorDataLoader


def draw_registration_result(source, target, transformation, nameOfFile):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    source_temp += target_temp
    o3d.io.write_point_cloud(nameOfFile, source_temp, format='ply')


def add_gaussian_noise_to_pointcloud(pcd, mean=0.0, std=0.01, seed=None):
    points = np.asarray(pcd.points)
    if seed is not None:
        np.random.seed(seed)
    pcdNew = o3d.geometry.PointCloud()
    pcdNew.points = o3d.utility.Vector3dVector(points + np.random.normal(mean, std, points.shape))
    return pcdNew


def add_salt_pepper_noise(pcd, percentage, min_x, max_x, min_y, max_y, min_z, max_z):
    points = np.asarray(pcd.points)
    mask = np.random.rand(len(points)) < percentage
    points[mask] = np.random.uniform(
        [min_x, min_y, min_z],
        [max_x, max_y, max_z],
        size=(np.sum(mask), 3)
    )
    pcdNew = o3d.geometry.PointCloud()
    pcdNew.points = o3d.utility.Vector3dVector(points)
    return pcdNew


def compute_overlap_ratio(pcd0, pcd1, trans, voxel_size):
    pcd0_down = pcd0.voxel_down_sample(voxel_size)
    pcd1_down = pcd1.voxel_down_sample(voxel_size)
    matching01 = get_matching_indices(pcd0_down, pcd1_down, trans, voxel_size, 1)
    matching10 = get_matching_indices(pcd1_down, pcd0_down, np.linalg.inv(trans), voxel_size, 1)
    overlap0 = len(matching01) / len(pcd0_down.points)
    overlap1 = len(matching10) / len(pcd1_down.points)
    return max(overlap0, overlap1)


def pointToVoxel(pointcloud, N, voxelSize, shift):
    voxelGrid = np.zeros(N * N * N, dtype=np.float64)
    for point in pointcloud.points:
        pointShifted = point + shift
        voxelX = int(pointShifted[0] / voxelSize)
        voxelY = int(pointShifted[1] / voxelSize)
        voxelZ = int(pointShifted[2] / voxelSize)
        index = int(voxelZ + N / 2 + (voxelY + N / 2) * N + (voxelX + N / 2) * N * N)
        if index >= 0 and index < len(voxelGrid):
            voxelGrid[index] = 1
    return voxelGrid


def get_matching_indices(source, target, trans, search_voxel_size, K=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    source_copy.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)

    match_inds = []
    for i, point in enumerate(source_copy.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds


def compute_transformation_from_peak(peak, mean1Transform, mean2Transform):
    currentQuaternion = [
        peak.potentialRotation.w,
        peak.potentialRotation.x,
        peak.potentialRotation.y,
        peak.potentialRotation.z
    ]

    rotationMatrix = o3d.geometry.get_rotation_matrix_from_quaternion(currentQuaternion)

    translations = peak.potentialTranslations
    if not translations:
        translationVector = np.array([0.0, 0.0, 0.0])
    else:
        translationVector = np.array([
            translations[0].xTranslation,
            translations[0].yTranslation,
            translations[0].zTranslation
        ])

    resultingTransformation = np.eye(4)
    resultingTransformation[:3, :3] = rotationMatrix
    resultingTransformation[:3, 3] = np.matmul(rotationMatrix, translationVector)

    estimatedActualRotation1 = np.matmul(
        np.linalg.inv(mean2Transform),
        np.matmul(resultingTransformation, mean1Transform)
    )
    return estimatedActualRotation1


def difference_between_transformations(transformation1, transformation2):
    diffMatrix = np.linalg.inv(transformation1) @ transformation2
    np.set_printoptions(suppress=True)

    trans = diffMatrix[:3, 3]
    normTrans = np.linalg.norm(trans)
    quatRot = quat.mat2quat(diffMatrix[:3, :3])
    rotaionAngle = quat.quat2axangle(quatRot)

    return rotaionAngle[1] * normTrans


def transform_to_rpyxyz(matrix):
    translation = matrix[:3, 3]
    rotation_matrix = matrix[:3, :3]
    quaternion = quat.mat2quat(rotation_matrix)
    roll, pitch, yaw = euler.quat2euler(quaternion, 'sxyz')
    return [roll, pitch, yaw, translation[0], translation[1], translation[2]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to the config file.')
    parser.add_argument('N', type=int, help='Voxel grid dimension size.')
    parser.add_argument('use_clahe', type=int, help='Use CLAHE (0 or 1).')
    parser.add_argument('r_min', type=int, help='Minimum radius for peak search.')
    parser.add_argument('r_max', type=int, help='Maximum radius for peak search.')
    parser.add_argument('level_potential_rotation', type=float, help='Rotation potential level.')
    parser.add_argument('level_potential_translation', type=float, help='Translation potential level.')
    parser.add_argument('normalization_factor', type=int, help='Normalization factor (0 or 2).')
    parser.add_argument('type_of_noise', type=str, help='Noise level: None, low, high')
    parser.add_argument('type_of_data', type=str, help='Dataset type: train, val')
    parser.add_argument('--start-index', type=int, required=True, help='First sample index to process (inclusive)')
    parser.add_argument('--end-index', type=int, required=True, help='Last sample index to process (inclusive)')
    parser.add_argument('--output-file', type=str, default=None, help='Output CSV file path (default: auto-generated)')
    args = parser.parse_args()

    N = args.N
    use_clahe = bool(args.use_clahe)
    r_min = args.r_min
    r_max = args.r_max
    set_r_manual = True
    level_potential_rotation = args.level_potential_rotation
    level_potential_translation = args.level_potential_translation
    normalization_factor = args.normalization_factor
    noise_level = args.type_of_noise
    type_data = args.type_of_data
    start_index = args.start_index
    end_index = args.end_index

    if start_index < 0:
        raise ValueError(f"start_index must be >= 0, got {start_index}")
    if end_index < start_index:
        raise ValueError(f"end_index ({end_index}) must be >= start_index ({start_index})")

    loader = PredatorDataLoader(args.config, split=type_data)

    if start_index >= len(loader):
        raise ValueError(f"start_index {start_index} exceeds dataset size {len(loader)}")

    print(f"Initializing SoftRegistrationWrapper(N={N})...")
    reg = SoftRegistrationWrapper(N, N // 2, N // 2, N // 2 - 1)
    print(f"Registration object created.")

    output_dir = 'outputFiles'
    os.makedirs(output_dir, exist_ok=True)

    if args.output_file:
        csv_path = args.output_file
    else:
        csv_path = os.path.join(
            output_dir,
            f'results_{N}_{int(use_clahe)}_{r_min}_{r_max}_{level_potential_rotation}_{level_potential_translation}_{normalization_factor}_{noise_level}_{type_data}_{start_index:05d}_{end_index:05d}.csv'
        )

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'index', 'numSolutions', 'overlap%',
            'GT_roll', 'GT_pitch', 'GT_yaw', 'GT_x', 'GT_y', 'GT_z',
            'EstHighest_roll', 'EstHighest_pitch', 'EstHighest_yaw', 'EstHighest_x', 'EstHighest_y', 'EstHighest_z',
            'EstBest_roll', 'EstBest_pitch', 'EstBest_yaw', 'EstBest_x', 'EstBest_y', 'EstBest_z'
        ])

    print(f"Processing samples {start_index} to {end_index}...")

    for indexDataLoader in range(start_index, end_index + 1):
        gc.collect()
        inputs = loader.get_by_index(indexDataLoader)

        pcd1 = o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(inputs["src_pcd_raw"])
        pcd2.points = o3d.utility.Vector3dVector(inputs["tgt_pcd_raw"])

        mean1, _ = o3d.geometry.PointCloud.compute_mean_and_covariance(pcd1)
        mean2, _ = o3d.geometry.PointCloud.compute_mean_and_covariance(pcd2)

        if noise_level == "None":
            meanNoise = 0
            percentageNoise = 0
        elif noise_level == "low":
            meanNoise = 0.01
            percentageNoise = 0.01
        elif noise_level == "high":
            meanNoise = 0.05
            percentageNoise = 0.05
        else:
            raise ValueError(f"Unknown noise level: {noise_level}")

        pcd1_noisy = add_gaussian_noise_to_pointcloud(pcd1, mean=0.0, std=meanNoise, seed=None)
        pcd2_noisy = add_gaussian_noise_to_pointcloud(pcd2, mean=0.0, std=meanNoise, seed=None)

        pcd1_noisy = add_salt_pepper_noise(
            pcd1_noisy, percentageNoise,
            np.min(pcd1.points, axis=0)[0], np.max(pcd1.points, axis=0)[0],
            np.min(pcd1.points, axis=0)[1], np.max(pcd1.points, axis=0)[1],
            np.min(pcd1.points, axis=0)[2], np.max(pcd1.points, axis=0)[2]
        )
        pcd2_noisy = add_salt_pepper_noise(
            pcd2_noisy, percentageNoise,
            np.min(pcd2.points, axis=0)[0], np.max(pcd2.points, axis=0)[0],
            np.min(pcd2.points, axis=0)[1], np.max(pcd2.points, axis=0)[1],
            np.min(pcd2.points, axis=0)[2], np.max(pcd2.points, axis=0)[2]
        )

        mean1 = -mean1
        mean2 = -mean2

        maxDistance = max(np.max(pcd1.points + mean1), np.max(pcd2.points + mean2))
        voxelSize = (2 * maxDistance * 1.5) / N

        mean1Transform = np.eye(4)
        mean2Transform = np.eye(4)
        mean1Transform[:3, 3] = np.squeeze(np.asarray(mean1))
        mean2Transform[:3, 3] = np.squeeze(np.asarray(mean2))

        gtTransformation = np.eye(4)
        gtTransformation[:3, :3] = inputs["rot"]
        gtTransformation[:3, 3] = np.squeeze(np.asarray(inputs["trans"]))

        pcd1Vox = pcd1_noisy.voxel_down_sample(voxel_size=voxelSize / 10.0)
        pcd2Vox = pcd2_noisy.voxel_down_sample(voxel_size=voxelSize / 10.0)

        voxelArray1 = pointToVoxel(pcd1_noisy, N, voxelSize, mean1).astype(np.float64)
        voxelArray2 = pointToVoxel(pcd2_noisy, N, voxelSize, mean2).astype(np.float64)

        listPeaks = reg.registerVoxels(
            voxelArray1, voxelArray2,
            debug=False,
            useClahe=use_clahe,
            timeStuff=False,
            sizeVoxel=voxelSize,
            r_min=float(r_min),
            r_max=float(r_max),
            level_potential_rotation=level_potential_rotation,
            level_potential_translation=level_potential_translation,
            set_r_manual=set_r_manual,
            normalization=normalization_factor
        )

        overlapPercentage = compute_overlap_ratio(pcd1, pcd2, gtTransformation, voxelSize)
        numberOfSolutions = len(listPeaks)

        highestPeakCorrelation = 0.0
        indexHighestPeak = 0
        closestPeak = float('inf')
        indexClosestPeak = 0

        for index, peak in enumerate(listPeaks):
            if peak.potentialRotation.correlationHeight > highestPeakCorrelation:
                highestPeakCorrelation = peak.potentialRotation.correlationHeight
                indexHighestPeak = index
            currentPeakTransformation = compute_transformation_from_peak(peak, mean1Transform, mean2Transform)
            distanceSolutions = difference_between_transformations(gtTransformation, currentPeakTransformation)
            if closestPeak > distanceSolutions:
                closestPeak = distanceSolutions
                indexClosestPeak = index

        highestPeak = listPeaks[indexHighestPeak]
        bestPeak = listPeaks[indexClosestPeak]
        estimatedHighestTransformation = compute_transformation_from_peak(highestPeak, mean1Transform, mean2Transform)
        estimatedBestTransformation = compute_transformation_from_peak(bestPeak, mean1Transform, mean2Transform)

        np.set_printoptions(suppress=True)

        print(f"Sample {indexDataLoader}: solutions={numberOfSolutions}, overlap={overlapPercentage:.4f}")

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            inputWriter = [indexDataLoader, numberOfSolutions, overlapPercentage]
            inputWriter.extend(transform_to_rpyxyz(gtTransformation))
            inputWriter.extend(transform_to_rpyxyz(estimatedHighestTransformation))
            inputWriter.extend(transform_to_rpyxyz(estimatedBestTransformation))
            writer.writerow(inputWriter)

        if (indexDataLoader + 1) % 50 == 0:
            gc.collect()

    print(f"Completed! Results saved to {csv_path}")

    expected_rows = end_index - start_index + 1
    with open(csv_path, 'r') as f:
        actual_rows = sum(1 for _ in f) - 1
    if actual_rows == expected_rows:
        print(f"Validation: OK - {actual_rows} data rows (expected {expected_rows})")
    else:
        print(f"WARNING: File has {actual_rows} data rows, expected {expected_rows}")


if __name__ == '__main__':
    main()
