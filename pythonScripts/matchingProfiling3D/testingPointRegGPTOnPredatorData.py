#!/usr/bin/python3
"""
Testing script for PointRegGPT data augmentation quality on Predator dataset.
PointRegGPT is a data augmentation tool, not a standalone registration model.
This script tests the quality of PointRegGPT's augmentation by comparing:
- Registration on raw Predator data
- Registration on PointRegGPT-augmented data

Usage:
    python testingPointRegGPTOnPredatorData.py configFiles/predatorNothing.yaml low train
    python testingPointRegGPTOnPredatorData.py configFiles/predatorNothing.yaml high val
    python testingPointRegGPTOnPredatorData.py configFiles/predatorNothing.yaml None train
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

# Add GeoTransformer to path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
geo_base_path = os.path.join(root_dir, 'ml_registration', 'geotransformer')
sys.path.insert(0, geo_base_path)

from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_device, release_cuda

# Add experiments directory to path (uses dots in dir names, so use sys.path)
geo_exp_path = os.path.join(geo_base_path, 'experiments', 'geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn')
sys.path.insert(0, geo_exp_path)

from config import make_cfg as make_geo_cfg
from model import create_model as create_geo_model

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


def apply_pointreggpt_augmentation(pcd1, pcd2, augmentation_type='noise'):
    """
    Apply PointRegGPT-style augmentation to point clouds.
    
    PointRegGPT uses depth inpainting diffusion for realistic data generation.
    Since we don't have the diffusion model trained, we simulate augmentation
    by applying transformations similar to PointRegGPT's approach:
    - Random camera motion (re-projection)
    - Depth correction simulation
    
    Args:
        pcd1: Source point cloud
        pcd2: Target point cloud
        augmentation_type: Type of augmentation to apply
    
    Returns:
        Augmented point clouds
    """
    pcd1_aug = copy.deepcopy(pcd1)
    pcd2_aug = copy.deepcopy(pcd2)
    
    if augmentation_type == 'noise':
        # Add realistic noise similar to sensor noise
        points1 = np.asarray(pcd1_aug.points)
        noise1 = np.random.normal(0, 0.005, points1.shape)  # 5mm sensor noise
        pcd1_aug.points = o3d.utility.Vector3dVector(points1 + noise1)
        
        points2 = np.asarray(pcd2_aug.points)
        noise2 = np.random.normal(0, 0.005, points2.shape)
        pcd2_aug.points = o3d.utility.Vector3dVector(points2 + noise2)
    
    elif augmentation_type == 'occlusion':
        # Simulate occlusion by removing random points (like depth holes)
        points1 = np.asarray(pcd1_aug.points)
        mask1 = np.random.rand(len(points1)) > 0.05  # Remove 5% of points
        pcd1_aug.points = o3d.utility.Vector3dVector(points1[mask1])
        
        points2 = np.asarray(pcd2_aug.points)
        mask2 = np.random.rand(len(points2)) > 0.05
        pcd2_aug.points = o3d.utility.Vector3dVector(points2[mask2])
    
    elif augmentation_type == 'both':
        # Combine noise and occlusion
        pcd1_aug, pcd2_aug = apply_pointreggpt_augmentation(pcd1, pcd2, 'noise')
        pcd1_aug, pcd2_aug = apply_pointreggpt_augmentation(pcd1_aug, pcd2_aug, 'occlusion')
    
    return pcd1_aug, pcd2_aug


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
    loader = PredatorDataLoader(args.config, split=type_data)
    config = loader.config
    
    # Setup GeoTransformer model (used as baseline for PointRegGPT augmentation testing)

    # Setup GeoTransformer model (used as baseline for PointRegGPT augmentation testing)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    geo_cfg = make_geo_cfg()
    model = create_geo_model(geo_cfg).to(device)

    # Load pretrained weights
    weights_path = os.path.join(root_dir, 'ml_registration', 'geotransformer', 'weights', 'geotransformer-3dmatch.pth.tar')
    if not os.path.exists(weights_path):
        print(f"Warning: Pretrained weights not found at {weights_path}")
        print("Please download from: https://github.com/qinzheng93/GeoTransformer/releases")
        print("Continuing without loading weights...")
    else:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict["model"])
        print(f"Loaded pretrained weights from {weights_path}")

    model.eval()

    # GeoTransformer preprocessing parameters
    neighbor_limits = [38, 36, 36, 38]
    voxel_size = 0.025
    search_radius = geo_cfg.backbone.init_radius

    # Create output directory
    output_dir = 'outputFiles'
    os.makedirs(output_dir, exist_ok=True)

    # Open CSV file for writing
    csv_path = os.path.join(output_dir, f'outfile_pointreggpt_{noise_level}_{type_data}.csv')

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

        # Apply PointRegGPT-style augmentation
        pcd1_aug, pcd2_aug = apply_pointreggpt_augmentation(pcd1_noisy, pcd2_noisy, augmentation_type='both')

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

        # Convert to GeoTransformer input format (using augmented point clouds)
        ref_points = np.asarray(pcd2_aug.points).astype(np.float32)
        src_points = np.asarray(pcd1_aug.points).astype(np.float32)
        ref_feats = np.ones_like(ref_points[:, :1]).astype(np.float32)
        src_feats = np.ones_like(src_points[:, :1]).astype(np.float32)

        data_dict = {
            "ref_points": ref_points,
            "src_points": src_points,
            "ref_feats": ref_feats,
            "src_feats": src_feats,
            "transform": gtTransformation.astype(np.float32)
        }

        # Run inference
        try:
            data_dict_processed = registration_collate_fn_stack_mode(
                [data_dict],
                num_stages=geo_cfg.backbone.num_stages,
                voxel_size=voxel_size,
                search_radius=search_radius,
                neighbor_limits=neighbor_limits
            )

            with torch.no_grad():
                data_dict_processed = to_device(data_dict_processed, device)
                output_dict = model(data_dict_processed)
                output_dict = release_cuda(output_dict)

                estimated_transform = output_dict["estimated_transform"]

        except Exception as e:
            print(f"Error processing sample {indexDataLoader}: {e}")
            estimated_transform = np.eye(4)

        # Compute metrics (using augmented point clouds)
        overlapPercentage = compute_overlap_ratio(pcd1_aug, pcd2_aug, gtTransformation, voxelSize)

        # Save to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            inputWriter = [indexDataLoader, overlapPercentage]
            inputWriter.extend(transform_to_rpyxyz(gtTransformation))
            inputWriter.extend(transform_to_rpyxyz(estimated_transform))
            writer.writerow(inputWriter)

        print(f"Processed: {indexDataLoader}")

    print("Completed!")
    print(f"PointRegGPT augmentation quality test finished. Results saved to {csv_path}")
    print("Compare with GeoTransformer results to evaluate augmentation impact.")


if __name__ == '__main__':
    main()
