#!/usr/bin/python3
"""
Testing script for Predator (KPFCNN) model on Predator dataset.
Direct inference without ROS2 — matches the pattern of other testing scripts.

Usage:
    python testingPredatorModelOnPredatorData.py configFiles/predatorNothing.yaml low train \
        --start-index 0 --end-index 99
    python testingPredatorModelOnPredatorData.py configFiles/predatorNothing.yaml high val \
        --start-index 0 --end-index 99
    python testingPredatorModelOnPredatorData.py configFiles/predatorNothing.yaml None train \
        --start-index 0 --end-index 99
"""

import os
import sys
import torch
import argparse
import csv
import gc
import time
import numpy as np
import open3d as o3d
import copy
import transforms3d.quaternions as quat
import transforms3d.euler as euler
from easydict import EasyDict as edict
from pathlib import Path

# Predator library imports
from predator.models.architectures import KPFCNN
from predator.lib.utils import load_config as predator_load_config, setup_seed
from predator.lib.benchmark_utils import ransac_pose_estimation

# Local imports
from dataloader_utils import PredatorDataLoader


def compute_overlap_ratio(pcd0, pcd1, trans, voxel_size):
    """Compute overlap ratio between two point clouds after transformation."""
    pcd0_down = pcd0.voxel_down_sample(voxel_size)
    pcd1_down = pcd1.voxel_down_sample(voxel_size)
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


def transform_to_rpyxyz(matrix):
    """Convert 4x4 transformation matrix to Roll, Pitch, Yaw and translation XYZ."""
    translation = matrix[:3, 3]
    rotation_matrix = matrix[:3, :3]
    quaternion = quat.mat2quat(rotation_matrix)
    roll, pitch, yaw = euler.quat2euler(quaternion, 'sxyz')
    return [roll, pitch, yaw, translation[0], translation[1], translation[2]]


def init_retry_log(noise_level, type_data, output_dir):
    """Initialize retry log file."""
    log_path = os.path.join(output_dir, f'retry_log_{noise_level}_{type_data}.txt')
    with open(log_path, 'w') as f:
        f.write(f"Retry Log - Noise: {noise_level}, Data: {type_data}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
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


def build_predator_architecture(config):
    """Build the KPFCNN architecture list from config (same logic as demo.py)."""
    architecture = [
        'simple',
        'resnetb',
    ]
    for i in range(config.num_layers - 1):
        architecture.append('resnetb_strided')
        architecture.append('resnetb')
        architecture.append('resnetb')
    for i in range(config.num_layers - 2):
        architecture.append('nearest_upsample')
        architecture.append('unary')
    architecture.append('nearest_upsample')
    architecture.append('last_unary')
    return architecture


def run_predator_inference(model, inputs, device, n_points):
    """
    Run Predator KPFCNN forward pass + RANSAC to get estimated transformation.

    Returns:
        estimated_transform: 4x4 numpy array
    """
    model.eval()
    with torch.no_grad():
        # Move inputs to device
        for k, v in inputs.items():
            if isinstance(v, list):
                inputs[k] = [item.to(device) for item in v]
            else:
                inputs[k] = v.to(device)

        # Forward pass
        feats, scores_overlap, scores_saliency = model(inputs)

        pcd = inputs['points'][0]
        len_src = inputs['stack_lengths'][0][0]

        src_pcd, tgt_pcd = pcd[:len_src], pcd[len_src:]
        src_feats, tgt_feats = feats[:len_src].detach().cpu(), feats[len_src:].detach().cpu()
        src_overlap, src_saliency = scores_overlap[:len_src].detach().cpu(), scores_saliency[:len_src].detach().cpu()
        tgt_overlap, tgt_saliency = scores_overlap[len_src:].detach().cpu(), scores_saliency[len_src:].detach().cpu()

        # Probabilistic sampling guided by scores
        src_scores = src_overlap * src_saliency
        tgt_scores = tgt_overlap * tgt_saliency

        if src_pcd.size(0) > n_points:
            idx = np.arange(src_pcd.size(0))
            probs = (src_scores / src_scores.sum()).numpy().flatten()
            idx = np.random.choice(idx, size=n_points, replace=False, p=probs)
            src_pcd, src_feats = src_pcd[idx], src_feats[idx]

        if tgt_pcd.size(0) > n_points:
            idx = np.arange(tgt_pcd.size(0))
            probs = (tgt_scores / tgt_scores.sum()).numpy().flatten()
            idx = np.random.choice(idx, size=n_points, replace=False, p=probs)
            tgt_pcd, tgt_feats = tgt_pcd[idx], tgt_feats[idx]

        # RANSAC pose estimation
        tsfm = ransac_pose_estimation(src_pcd, tgt_pcd, src_feats, tgt_feats, mutual=False)

    return tsfm


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

    # Use CPU (MPS has known gaps for ops used by these 3D registration models)
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load Predator config and dataset
    loader = PredatorDataLoader(args.config, split=type_data)
    config = loader.config
    neighborhood_limits = loader.neighborhood_limits

    # Build Predator architecture
    config.architecture = build_predator_architecture(config)

    # Build model
    model = KPFCNN(config).to(device)

    # Load pretrained weights
    weights_path = '/Users/timhansen/Documents/dataFolder/3dmatch/models/predator/data/weights/indoor.pth'
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Pretrained weights not found at {weights_path}")
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        model.load_state_dict(state)
    print(f"Loaded pretrained weights from {weights_path}")

    model.eval()

    # n_points for RANSAC sampling (from config)
    n_points = config.get('n_points', 1000)

    # Create output directory
    output_dir = os.path.join('outputFiles', 'predator')
    os.makedirs(output_dir, exist_ok=True)

    # Initialize retry log
    retry_log_path = init_retry_log(noise_level, type_data, output_dir)

    # Set output CSV path
    if args.output_file:
        csv_path = args.output_file
    else:
        csv_path = os.path.join(output_dir, f'batch_{noise_level}_{type_data}_{start_index:05d}_{end_index:05d}.csv')

    # Write header
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'overlap%', 'GT_roll', 'GT_pitch', 'GT_yaw', 'GT_x', 'GT_y', 'GT_z',
                         'Est_roll', 'Est_pitch', 'Est_yaw', 'Est_x', 'Est_y', 'Est_z'])

    if start_index >= len(loader):
        raise ValueError(f"start_index {start_index} exceeds dataset size {len(loader)}")

    print(f"Processing samples {start_index} to {end_index}...")

    for indexDataLoader in range(start_index, end_index + 1):
        inputs = loader.get_by_index(indexDataLoader)

        # Create Open3D point clouds from raw data
        pcd1 = o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(inputs["src_pcd_raw"])
        pcd2.points = o3d.utility.Vector3dVector(inputs["tgt_pcd_raw"])

        # Compute means for overlap calculation
        mean1, _ = o3d.geometry.PointCloud.compute_mean_and_covariance(pcd1)
        mean2, _ = o3d.geometry.PointCloud.compute_mean_and_covariance(pcd2)

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

        # Add noise
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

        # Build ground truth transformation
        gtTransformation = np.eye(4)
        gtTransformation[:3, :3] = inputs["rot"]
        gtTransformation[:3, 3] = np.squeeze(np.asarray(inputs["trans"]))

        # Compute voxel size for overlap
        mean1 = -mean1
        mean2 = -mean2
        N = 64
        maxDistance = max(np.max(pcd1.points + mean1), np.max(pcd2.points + mean2))
        voxelSize = (2 * maxDistance * 1.5) / N

        # Replace point coordinates in inputs with noisy versions (Option A)
        # inputs['points'][0] is the stacked [src; tgt] tensor
        src_noisy = torch.from_numpy(np.asarray(pcd1_noisy.points)).float()
        tgt_noisy = torch.from_numpy(np.asarray(pcd2_noisy.points)).float()
        inputs['points'][0] = torch.cat([src_noisy, tgt_noisy], dim=0)

        # Run inference with retry logic
        max_retries = 3
        retry_delay = 1.0
        estimated_transform = np.eye(4)

        for attempt in range(max_retries + 1):
            try:
                estimated_transform = run_predator_inference(model, inputs, device, n_points)
                break

            except (MemoryError, RuntimeError) as e:
                error_str = str(e).lower()
                if "malloc" in error_str or "heap" in error_str or "corruption" in error_str:
                    if attempt < max_retries:
                        log_retry(indexDataLoader, attempt, max_retries, str(e), retry_log_path)
                        time.sleep(retry_delay)
                        gc.collect()
                        continue
                    else:
                        log_max_retries_exceeded(indexDataLoader, str(e), retry_log_path)
                        estimated_transform = np.eye(4)
                        break
                else:
                    raise
            except Exception as e:
                print(f"Error processing sample {indexDataLoader}: {e}")
                estimated_transform = np.eye(4)
                break

        # Compute metrics
        overlapPercentage = compute_overlap_ratio(pcd1_noisy, pcd2_noisy, gtTransformation, voxelSize)

        # Save to CSV using atomic write
        temp_csv_path = csv_path + '.tmp'
        with open(temp_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            inputWriter = [indexDataLoader, overlapPercentage]
            inputWriter.extend(transform_to_rpyxyz(gtTransformation))
            inputWriter.extend(transform_to_rpyxyz(estimated_transform))
            writer.writerow(inputWriter)
        with open(csv_path, 'a', newline='') as main_f:
            with open(temp_csv_path, 'r') as temp_f:
                main_f.write(temp_f.read())
        os.remove(temp_csv_path)

        print(f"Processed: {indexDataLoader}")

        # Force garbage collection every 50 samples
        if (indexDataLoader + 1) % 50 == 0:
            gc.collect()

    print("Completed!")
    print(f"Batch {start_index}-{end_index} finished. Results saved to {csv_path}")

    # Validate row count
    expected_rows = end_index - start_index + 1
    with open(csv_path, 'r') as f:
        actual_rows = sum(1 for _ in f) - 1
    if actual_rows == expected_rows:
        print(f"Validation: OK - {actual_rows} data rows (expected {expected_rows})")
    else:
        print(f"WARNING: File has {actual_rows} data rows, expected {expected_rows}")


if __name__ == '__main__':
    main()
