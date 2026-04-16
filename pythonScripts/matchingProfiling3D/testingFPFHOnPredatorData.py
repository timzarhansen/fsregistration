#!/usr/bin/python3
"""
Testing script for FPFH (Fast Point Feature Histograms) and RANSAC on Predator dataset.
This script evaluates the performance of global registration using Open3D.

Usage:
    python testingFPFHOnPredatorData.py configFiles/predatorNothing.yaml low train
    python testingFPFHOnPredatorData.py configFiles/predatorNothing.yaml high val
    python testingFPFHOnPredatorData.py configFiles/predatorNothing.yaml None train
"""

import os
import sys
import argparse
import csv
import gc
import numpy as np
import open3d as o3d
import transforms3d.quaternions as quat
import transforms3d.euler as euler
from easydict import EasyDict as edict

# Predator dataset imports
from predator.datasets.dataloader import get_dataloader, get_datasets
from predator.lib.utils import load_config


def compute_overlap_ratio(pcd0, pcd1, trans, voxel_size):
    """Compute overlap ratio between two point clouds after transformation."""
    pcd0_down = pcd0.voxel_down_sample(voxel_size)
    pcd1_down = pcd1.voxel_down_sample(voxel_size)
    
    if len(pcd0_down.points) == 0 or len(pcd1_down.points) == 0:
        return 0.0
    
    matching01 = get_matching_indices(pcd0_down, pcd1_down, trans, voxel_size, 1)
    matching10 = get_matching_indices(pcd1_down, pcd0_down, np.linalg.inv(trans), voxel_size, 1)
    overlap0 = len(matching01) / len(pcd0_down.points)
    overlap1 = len(matching10) / len(pcd1_down.points)
    return max(overlap0, overlap1)


def add_gaussian_noise_to_pointcloud(pcd, mean=0.0, std=0.01, seed=None):
    """Add Gaussian noise to point cloud."""
    points = np.asarray(pcd.points)
    if seed is not None:
        np.random.seed(seed)
    pcdNew = o3d.geometry.PointCloud()
    pcdNew.points = o3d.utility.Vector3dVector(points + np.random.normal(mean, std, points.shape))
    return pcdNew


def add_salt_pepper_noise(pcd, percentage, min_x, max_x, min_y, max_y, min_z, max_z):
    """Add salt-pepper noise to point cloud."""
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


def get_matching_indices(source, target, trans, search_voxel_size, K=None):
    """Get matching indices between source and target point clouds."""
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)
    points_homogeneous = np.concatenate([source_points, np.ones((source_points.shape[0], 1))], axis=1)
    transformed_points = (points_homogeneous @ trans.T)[:, :3]
    pcd_tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_points)))

    match_inds = []
    for i, point in enumerate(transformed_points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds


def transform_to_rpyxyz(matrix):
    """Convert 4x4 transformation matrix to Roll, Pitch, Yaw and translation XYZ."""
    translation = matrix[:3, 3]
    rotation_matrix = matrix[:3, :3]
    quaternion = quat.mat2quat(rotation_matrix)
    roll, pitch, yaw = euler.quat2euler(quaternion, 'sxyz')
    return [roll, pitch, yaw, translation[0], translation[1], translation[2]]


def preprocess_point_cloud(pcd, voxel_size):
    """Downsample, estimate normals, and compute FPFH features."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, pcd_fpfh


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to the config file.')
    parser.add_argument('type_of_noise', type=str, help='Noise level: None, low, high')
    parser.add_argument('type_of_data', type=str, help='Dataset type: train, val')
    args = parser.parse_args()

    noise_level = args.type_of_noise
    type_data = args.type_of_data

    # Load Predator config and dataset
    config = load_config(args.config)
    config = edict(config)

    # Define architectures (placeholder to maintain compatibility with dataloader)
    architectures = {
        'indoor': [
            'simple', 'resnetb', 'resnetb_strided', 'resnetb', 'resnetb',
            'resnetb_strided', 'resnetb', 'resnetb', 'resnetb_strided',
            'resnetb', 'resnetb', 'nearest_upsample', 'unary',
            'nearest_upsample', 'unary', 'nearest_upsample', 'last_unary'
        ]
    }
    config.architecture = architectures[config.dataset]
    train_set, val_set, benchmark_set = get_datasets(config)

    # Get dataloader
    if type_data == "train":
        config.train_loader, neighborhood_limits = get_dataloader(
            dataset=train_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
        dataSetSize = len(train_set)
    elif type_data == "val":
        config.train_loader, neighborhood_limits = get_dataloader(
            dataset=val_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
        dataSetSize = len(val_set)
    else:
        raise ValueError(f"Unknown data type: {type_data}")

    # Create output directory
    output_dir = 'outputFiles'
    os.makedirs(output_dir, exist_ok=True)

    # Open CSV file for writing
    csv_path = os.path.join(output_dir, f'outfile_fpfh_{noise_level}_{type_data}.csv')

    # Write header if file doesn't exist
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'overlap%', 'GT_roll', 'GT_pitch', 'GT_yaw', 'GT_x', 'GT_y', 'GT_z',
                           'Est_roll', 'Est_pitch', 'Est_yaw', 'Est_x', 'Est_y', 'Est_z'])

    # Process dataset
    # Note: Both train and val branches assign to config.train_loader
    dataIter = iter(config.train_loader)

    # FPFH Parameters
    voxel_size = 0.05

    for indexDataLoader in range(dataSetSize):
        inputs = next(dataIter)

        # Create Open3D point clouds
        pcd1 = o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(inputs["src_pcd_raw"])
        pcd2.points = o3d.utility.Vector3dVector(inputs["tgt_pcd_raw"])

        # Set noise parameters
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

        # Add standard noise
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

        # Get ground truth transformation
        gtTransformation = np.eye(4)
        gtTransformation[:3, :3] = inputs["rot"]
        gtTransformation[:3, 3] = np.squeeze(np.asarray(inputs["trans"]))

        # Compute voxel size for overlap calculation
        mean1, _ = o3d.geometry.PointCloud.compute_mean_and_covariance(pcd1)
        mean2, _ = o3d.geometry.PointCloud.compute_mean_and_covariance(pcd2)
        mean1 = -mean1
        mean2 = -mean2
        N = 64
        maxDistance = max(np.max(pcd1.points + mean1), np.max(pcd2.points + mean2))
        overlapVoxelSize = (2 * maxDistance * 1.5) / N

        # FPFH Global Registration
        try:
            # Preprocess both clouds
            pcd1_down, fpfh1 = preprocess_point_cloud(pcd1_noisy, voxel_size)
            pcd2_down, fpfh2 = preprocess_point_cloud(pcd2_noisy, voxel_size)
            
            # RANSAC registration
            distance_threshold = voxel_size * 4.0
            ransac_criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(
                max_iteration=100000,
                confidence=0.999
            )
            result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                pcd1_down,
                pcd2_down,
                fpfh1,
                fpfh2,
                mutual_filter=True,
                max_correspondence_distance=distance_threshold,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=3,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.5),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
                ],
                criteria=ransac_criteria
            )
            estimated_transform = result_ransac.transformation
            print(f"Sample {indexDataLoader}: fitness={result_ransac.fitness:.4f}, inlier_rmse={result_ransac.inlier_rmse:.6f}")
            if np.allclose(estimated_transform, np.eye(4)):
                print(f"  WARNING: RANSAC returned identity transform!")
        except Exception as e:
            print(f"Error processing sample {indexDataLoader}: {e}")
            estimated_transform = np.eye(4)

        # Compute metrics
        overlapPercentage = compute_overlap_ratio(pcd1_noisy, pcd2_noisy, gtTransformation, overlapVoxelSize)

        # Save to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            inputWriter = [indexDataLoader, overlapPercentage]
            inputWriter.extend(transform_to_rpyxyz(gtTransformation))
            inputWriter.extend(transform_to_rpyxyz(estimated_transform))
            writer.writerow(inputWriter)

        print(f"Processed: {indexDataLoader}")
        
        # Force garbage collection to prevent memory corruption on macOS
        gc.collect()

    print("Completed!")
    print(f"FPFH+RANSAC registration test finished. Results saved to {csv_path}")


if __name__ == '__main__':
    main()
