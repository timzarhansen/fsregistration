#!/usr/bin/python3
"""
Testing script for RegTR model on Predator dataset.
Mirrors the structure of testingPredatorModelOnPredatorData.py but uses direct Python inference.

Usage:
    python testingRegTROnPredatorData.py configFiles/predatorNothing.yaml low train
    python testingRegTROnPredatorData.py configFiles/predatorNothing.yaml high val
    python testingRegTROnPredatorData.py configFiles/predatorNothing.yaml None train
"""

import os
import sys
import torch
import argparse
import csv
import numpy as np
import open3d as o3d
import copy
import transforms3d.quaternions as quat
import transforms3d.euler as euler
from easydict import EasyDict as edict
from pathlib import Path

# Add RegTR to path
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent.parent / 'ml_registration' / 'regtr' / 'src'))

from models.regtr import RegTR
from utils.misc import load_config as regtr_load_config

# Predator dataset imports
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


def se3_to_matrix(pose):
    """Convert 3x4 SE3 pose matrix to 4x4 transformation matrix.
    
    Args:
        pose: numpy array of shape (3, 4) containing rotation (3x3) and translation (3x1)
              pose[:3, :3] is the rotation matrix
              pose[:3, 3] is the translation vector
    
    Returns:
        4x4 transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = pose[:3, :3]
    T[:3, 3] = pose[:3, 3]
    return T


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to the config file.')
    parser.add_argument('type_of_noise', type=str, help='Noise level: None, low, high')
    parser.add_argument('type_of_data', type=str, help='Dataset type: train, val')
    args = parser.parse_args()

    noise_level = args.type_of_noise
    type_data = args.type_of_data

    # Get script directory for path resolution
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    # Load Predator config and dataset
    loader = PredatorDataLoader(args.config, split=type_data)
    config = loader.config
    neighborhood_limits = loader.neighborhood_limits

    # Setup RegTR model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load RegTR config
    regtr_cfg_path = project_root / 'ml_registration' / 'regtr' / 'src' / 'conf' / '3dmatch.yaml'
    regtr_cfg = edict(regtr_load_config(Path(regtr_cfg_path)))

    model = RegTR(regtr_cfg).to(device)

    # Load pretrained weights
    weights_path = project_root / 'ml_registration' / 'regtr' / 'trained_models' / '3dmatch' / 'ckpt' / 'model-best.pth'
    if not os.path.exists(weights_path):
        print(f"Warning: Pretrained weights not found at {weights_path}")
        print("Please download from: https://github.com/yewzijian/RegTR/releases")
        print("Continuing without loading weights...")
    else:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state['state_dict'])
        print(f"Loaded pretrained weights from {weights_path}")

    model.eval()

    # Check for crop_radius in config
    crop_radius = regtr_cfg.get('crop_radius', None)

    # Create output directory
    output_dir = 'outputFiles'
    os.makedirs(output_dir, exist_ok=True)

    # Open CSV file for writing
    csv_path = os.path.join(output_dir, f'outfile_regtr_{noise_level}_{type_data}.csv')

    # Write header if file doesn't exist
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'overlap%', 'GT_roll', 'GT_pitch', 'GT_yaw', 'GT_x', 'GT_y', 'GT_z',
                           'Est_roll', 'Est_pitch', 'Est_yaw', 'Est_x', 'Est_y', 'Est_z'])

    # Process dataset
    for indexDataLoader in range(len(loader)):
        inputs = loader.get_next()

        # Create Open3D point clouds
        pcd1 = o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(inputs["src_pcd_raw"])
        pcd2.points = o3d.utility.Vector3dVector(inputs["tgt_pcd_raw"])

        # Compute means
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

        # Get ground truth transformation
        gtTransformation = np.eye(4)
        gtTransformation[:3, :3] = inputs["rot"]
        gtTransformation[:3, 3] = np.squeeze(np.asarray(inputs["trans"]))

        # Compute voxel size for overlap calculation
        mean1 = -mean1
        mean2 = -mean2
        N = 64
        maxDistance = max(np.max(pcd1.points + mean1), np.max(pcd2.points + mean2))
        voxelSize = (2 * maxDistance * 1.5) / N

        # Convert to RegTR input format
        src_xyz = np.asarray(pcd1_noisy.points).astype(np.float32)
        tgt_xyz = np.asarray(pcd2_noisy.points).astype(np.float32)

        # Apply crop_radius if specified
        if crop_radius is not None:
            src_xyz = src_xyz[np.linalg.norm(src_xyz, axis=1) < crop_radius, :]
            tgt_xyz = tgt_xyz[np.linalg.norm(tgt_xyz, axis=1) < crop_radius, :]

        data_batch = {
            'src_xyz': [torch.from_numpy(src_xyz).float().to(device)],
            'tgt_xyz': [torch.from_numpy(tgt_xyz).float().to(device)]
        }

        # Run inference
        try:
            with torch.no_grad():
                outputs = model(data_batch)
                pose = outputs['pose'][-1, 0].cpu().numpy()

            # Convert pose to 4x4 transformation matrix
            # RegTR outputs SE3 pose as 6-element array [tx, ty, tz, rx, ry, rz]
            estimated_transform = se3_to_matrix(pose)

        except Exception as e:
            print(f"Error processing sample {indexDataLoader}: {e}")
            estimated_transform = np.eye(4)

        # Compute metrics
        overlapPercentage = compute_overlap_ratio(pcd1_noisy, pcd2_noisy, gtTransformation, voxelSize)

        # Save to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            inputWriter = [indexDataLoader, overlapPercentage]
            inputWriter.extend(transform_to_rpyxyz(gtTransformation))
            inputWriter.extend(transform_to_rpyxyz(estimated_transform))
            writer.writerow(inputWriter)

        print(f"Processed: {indexDataLoader}")

    print("Completed!")


if __name__ == '__main__':
    main()
