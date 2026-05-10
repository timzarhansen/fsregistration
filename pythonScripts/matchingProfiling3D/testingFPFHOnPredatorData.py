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
import time
import numpy as np
import open3d as o3d
import transforms3d.quaternions as quat
import transforms3d.euler as euler
from easydict import EasyDict as edict

# Predator dataset imports
from predator.datasets.dataloader import get_dataloader, get_datasets, collate_fn_descriptor
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


def init_retry_log(noise_level, type_data, output_dir):
    """Initialize retry log file."""
    log_path = os.path.join(output_dir, f'retry_log_{noise_level}_{type_data}.txt')
    with open(log_path, 'w') as f:
        f.write(f"Retry Log - Noise: {noise_level}, Data: {type_data}\n")
        f.write("=" * 60 + "\n\n")
    return log_path


def log_retry(sample_idx, attempt, max_retries, error_msg, log_path):
    """Log retry attempt."""
    with open(log_path, 'a') as f:
        f.write(f"Sample {sample_idx} | Retry {attempt + 1}/{max_retries} | Error: {error_msg}\n")
    print(f"  [RETRY {attempt + 1}/{max_retries}] Sample {sample_idx}: {error_msg[:100]}")


def log_max_retries_exceeded(sample_idx, error_msg, log_path):
    """Log when max retries exceeded."""
    with open(log_path, 'a') as f:
        f.write(f"Sample {sample_idx} | MAX RETRIES EXCEEDED | Error: {error_msg}\n\n")
    print(f"  [FAILED] Sample {sample_idx}: Max retries exceeded, using identity transform")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to the config file.')
    parser.add_argument('type_of_noise', type=str, help='Noise level: None, low, high')
    parser.add_argument('type_of_data', type=str, help='Dataset type: train, val')
    parser.add_argument('--start-index', type=int, required=True, help='First sample index to process (inclusive)')
    parser.add_argument('--end-index', type=int, required=True, help='Last sample index to process (inclusive)')
    parser.add_argument('--output-file', type=str, default=None, help='Output CSV file path (default: auto-generated)')
    args = parser.parse_args()

    noise_level = args.type_of_noise
    type_data = args.type_of_data
    start_index = args.start_index
    end_index = args.end_index

    # Validate index range
    if start_index < 0:
        raise ValueError(f"start_index must be >= 0, got {start_index}")
    if end_index < start_index:
        raise ValueError(f"end_index ({end_index}) must be >= start_index ({start_index})")

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

    # Initialize retry log
    retry_log_path = init_retry_log(noise_level, type_data, output_dir)

    # Set output CSV path
    if args.output_file:
        csv_path = args.output_file
    else:
        csv_path = os.path.join(output_dir, f'batch_{noise_level}_{type_data}_{start_index:05d}_{end_index:05d}.csv')

    # Write header (always overwrite for batch files)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'overlap%', 'GT_roll', 'GT_pitch', 'GT_yaw', 'GT_x', 'GT_y', 'GT_z',
                       'Est_roll', 'Est_pitch', 'Est_yaw', 'Est_x', 'Est_y', 'Est_z'])

    if start_index >= dataSetSize:
        raise ValueError(f"start_index {start_index} exceeds dataset size {dataSetSize}")

    print(f"Processing samples {start_index} to {end_index}...")

    dataset = config.train_loader.dataset
    for indexDataLoader in range(start_index, end_index + 1):
        raw_sample = dataset[indexDataLoader]
        inputs = collate_fn_descriptor([raw_sample], config, neighborhood_limits)

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

        # FPFH Global Registration with retry logic
        max_retries = 3
        retry_delay = 1.0
        estimated_transform = np.eye(4)
        result_ransac = None

        for attempt in range(max_retries + 1):
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
                
                break  # Success, exit retry loop
                
            except (MemoryError, RuntimeError) as e:
                error_str = str(e).lower()
                if "malloc" in error_str or "heap" in error_str or "corruption" in error_str:
                    if attempt < max_retries:
                        log_retry(indexDataLoader, attempt, max_retries, str(e), retry_log_path)
                        time.sleep(retry_delay)
                        continue
                    else:
                        log_max_retries_exceeded(indexDataLoader, str(e), retry_log_path)
                        estimated_transform = np.eye(4)
                        print(f"  Exiting after max retries exceeded for sample {indexDataLoader}")
                        sys.exit(1)
                else:
                    raise
            except Exception as e:
                print(f"Error processing sample {indexDataLoader}: {e}")
                estimated_transform = np.eye(4)
                break

        # Compute metrics
        overlapPercentage = compute_overlap_ratio(pcd1_noisy, pcd2_noisy, gtTransformation, overlapVoxelSize)

        # Save to CSV using atomic write (write to temp, then append to main)
        temp_csv_path = csv_path + '.tmp'
        with open(temp_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            inputWriter = [indexDataLoader, overlapPercentage]
            inputWriter.extend(transform_to_rpyxyz(gtTransformation))
            inputWriter.extend(transform_to_rpyxyz(estimated_transform))
            writer.writerow(inputWriter)
        # Append temp file to main CSV, then remove temp
        with open(csv_path, 'a', newline='') as main_f:
            with open(temp_csv_path, 'r') as temp_f:
                main_f.write(temp_f.read())
        os.remove(temp_csv_path)

        print(f"Processed: {indexDataLoader}")
        
        # Force garbage collection every 50 samples to prevent memory buildup
        # Avoid frequent gc.collect() to prevent malloc corruption on macOS
        if (indexDataLoader + 1) % 50 == 0:
            gc.collect()

    print("Completed!")
    print(f"Batch {start_index}-{end_index} finished. Results saved to {csv_path}")
    
    # Quick validation: check file has expected number of rows
    expected_rows = end_index - start_index + 1
    with open(csv_path, 'r') as f:
        actual_rows = sum(1 for _ in f) - 1  # Subtract header
    if actual_rows == expected_rows:
        print(f"Validation: OK - {actual_rows} data rows (expected {expected_rows})")
    else:
        print(f"WARNING: File has {actual_rows} data rows, expected {expected_rows}")


if __name__ == '__main__':
    main()
