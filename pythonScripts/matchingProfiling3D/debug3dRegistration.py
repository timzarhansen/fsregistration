#!/usr/bin/env python3
"""
Debug runner for 3D SOFT registration.

Runs a single sample through the full 3D registration pipeline with debug output,
then saves a metadata JSON file with all peak information for visualization in the
Jupyter notebook.

CONFIGURATION — edit the variables below, then run:
    python debug3dRegistration.py
"""

import os
import sys
import json
import time
import numpy as np
import open3d as o3d
import transforms3d.quaternions as quat
import transforms3d.euler as euler

# ============================================================================
# CONFIGURATION — edit these to change behavior
# ============================================================================

# --- Input mode (choose ONE by setting the appropriate variable): ---
#   Option A: Predator dataset
INPUT_CONFIG = 'configFiles/predatorNothing.yaml'          # path to predator config YAML, e.g. 'configFiles/predatorNothing.yaml'
INPUT_INDEX = 2              # sample index for Predator dataset
INPUT_DATA_SPLIT = 'val'   # 'train' or 'val'

#   Option B: Raw voxel files (set both paths)
INPUT_VOXEL1 = None          # path to .npy voxel file 1
INPUT_VOXEL2 = None          # path to .npy voxel file 2

#   Option C: Raw point cloud files (set both paths)
INPUT_PCD1 = None            # path to .ply/.obj point cloud 1
INPUT_PCD2 = None            # path to .ply/.obj point cloud 2

# --- Registration parameters ---
N = 64                       # voxel grid dimension
USE_CLAHE = False             # enable CLAHE contrast enhancement
R_MIN = 8                 # min radius for rotation filter (None = N/8)
R_MAX = 24                 # max radius for rotation filter (None = N/2 - N/8)
LEVEL_POTENTIAL_ROTATION = 0.0001
LEVEL_POTENTIAL_TRANSLATION = 0.0001
NORMALIZATION = 2            # 0, 1, or 2

# --- Noise ---
NOISE_LEVEL = 'None'         # None, low, high, low_gauss, high_gauss, low_salt_pepper, high_salt_pepper

# --- Output ---
OUTPUT_JSON = None           # None = default (plotting_results/3d/debug_metadata.json)

# ============================================================================
# END OF CONFIGURATION
# ============================================================================

# Add fsregistration directories to path for pybind11 module
script_dir = os.path.dirname(os.path.abspath(__file__))
fsregistration_root = os.path.dirname(os.path.dirname(script_dir))  # -> fsregistration
install_dir = os.path.join(fsregistration_root, '..', '..', 'install', 'fsregistration', 'lib', 'fsregistration')

for p in [install_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Also add matchingProfiling3D path for dataloader_utils
profiling_path = os.path.join(script_dir, 'matchingProfiling3D')
if profiling_path not in sys.path:
    sys.path.insert(0, profiling_path)

try:
    from pybind_registration_3d import SoftRegistrationWrapper
except ImportError:
    print("ERROR: Could not import pybind_registration_3d module.")
    print(f" Searched in: {install_dir}")
    print("Make sure you have built the package with: colcon build --packages-select fsregistration")
    sys.exit(1)

try:
    from dataloader_utils import PredatorDataLoader
except ImportError:
    PredatorDataLoader = None


# Default debug output directory (matches the hardcoded C++ path)
DEBUG_DATA_DIR = os.path.join(fsregistration_root, 'plotting_results', '3d', 'data')
DEFAULT_METADATA_FILE = os.path.join(fsregistration_root, 'plotting_results', '3d', 'debug_metadata.json')


def point_to_voxel(pointcloud, N, voxel_size, shift):
    """Convert a point cloud to a voxel grid (binary occupancy)."""
    voxel_grid = np.zeros(N * N * N, dtype=np.float64)
    for point in pointcloud.points:
        point_shifted = point + shift
        voxel_x = int(point_shifted[0] / voxel_size)
        voxel_y = int(point_shifted[1] / voxel_size)
        voxel_z = int(point_shifted[2] / voxel_size)
        index = int(voxel_z + N // 2 + (voxel_y + N // 2) * N + (voxel_x + N // 2) * N * N)
        if 0 <= index < len(voxel_grid):
            voxel_grid[index] = 1
    return voxel_grid


def add_noise_to_pcd(pcd, noise_level):
    """Add Gaussian and/or salt-and-pepper noise to a point cloud."""
    if noise_level == "None" or noise_level == "none":
        return pcd

    points = np.asarray(pcd.points)
    mean_noise = 0.0
    percentage_noise = 0.0

    if noise_level == "low":
        mean_noise = 0.01
        percentage_noise = 0.01
    elif noise_level == "high":
        mean_noise = 0.05
        percentage_noise = 0.05
    elif noise_level == "low_gauss":
        mean_noise = 0.01
    elif noise_level == "high_gauss":
        mean_noise = 0.05
    elif noise_level == "low_salt_pepper":
        percentage_noise = 0.01
    elif noise_level == "high_salt_pepper":
        percentage_noise = 0.05
    else:
        raise ValueError(f"Unknown noise level: {noise_level}")

    # Add Gaussian noise
    if mean_noise > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points + np.random.normal(0, mean_noise, points.shape))
        points = np.asarray(pcd.points)

    # Add salt-and-pepper noise
    if percentage_noise > 0:
        pcd = o3d.geometry.PointCloud()
        mask = np.random.rand(len(points)) < percentage_noise
        points[mask] = np.random.uniform(
            [np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2])],
            [np.max(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 2])],
            size=(np.sum(mask), 3)
        )
        pcd.points = o3d.utility.Vector3dVector(points)

    return pcd


def compute_transformation_from_peak(peak, mean1_transform, mean2_transform, translation_index=0):
    """Compute the full SE(3) transformation from a registration peak."""
    current_quaternion = [
        peak.potentialRotation.w,
        peak.potentialRotation.x,
        peak.potentialRotation.y,
        peak.potentialRotation.z
    ]
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(current_quaternion)

    translations = peak.potentialTranslations
    if not translations or translation_index >= len(translations):
        translation_vector = np.array([0.0, 0.0, 0.0])
    else:
        translation_vector = np.array([
            translations[translation_index].xTranslation,
            translations[translation_index].yTranslation,
            translations[translation_index].zTranslation
        ])

    resulting_transformation = np.eye(4)
    resulting_transformation[:3, :3] = rotation_matrix
    resulting_transformation[:3, 3] = np.matmul(rotation_matrix, translation_vector)

    estimated_actual = np.matmul(
        np.linalg.inv(mean2_transform),
        np.matmul(resulting_transformation, mean1_transform)
    )
    return estimated_actual


def matrix_to_json(matrix):
    """Convert a numpy matrix to a JSON-serializable list."""
    return matrix.tolist()


def get_rpy_xyz(matrix):
    """Extract roll, pitch, yaw, x, y, z from a 4x4 transformation matrix."""
    translation = matrix[:3, 3]
    rotation_matrix = matrix[:3, :3]
    quaternion = quat.mat2quat(rotation_matrix)
    roll, pitch, yaw = euler.quat2euler(quaternion, 'sxyz')
    return {
        'roll_deg': np.degrees(roll),
        'pitch_deg': np.degrees(pitch),
        'yaw_deg': np.degrees(yaw),
        'x': float(translation[0]),
        'y': float(translation[1]),
        'z': float(translation[2])
    }


def compute_error_metrics(gt_transform, est_transform):
    """Compute rotation error (angle in degrees) and translation error (distance)."""
    diff_matrix = np.linalg.inv(gt_transform) @ est_transform
    trans_error = np.linalg.norm(diff_matrix[:3, 3])

    rot_quat = quat.mat2quat(diff_matrix[:3, :3])
    rot_angle = quat.quat2axangle(rot_quat)
    rot_error_deg = np.degrees(rot_angle[1])

    return {
        'rotation_error_deg': float(rot_error_deg),
        'translation_error_m': float(trans_error)
    }


def main():
    # Resolve output path
    output_json = OUTPUT_JSON if OUTPUT_JSON else DEFAULT_METADATA_FILE

    # Resolve r_min / r_max
    r_min = R_MIN if R_MIN is not None else N // 8
    r_max = R_MAX if R_MAX is not None else N // 2 - N // 8
    set_r_manual = R_MIN is not None or R_MAX is not None

    # Print config
    print("=" * 70)
    print("3D SOFT Registration Debug Runner")
    print("=" * 70)
    print(f"  N:                      {N}")
    print(f"  use_clahe:              {USE_CLAHE}")
    print(f"  r_min / r_max:          {r_min} / {r_max}")
    print(f"  level_potential_rot:    {LEVEL_POTENTIAL_ROTATION}")
    print(f"  level_potential_trans:  {LEVEL_POTENTIAL_TRANSLATION}")
    print(f"  normalization:          {NORMALIZATION}")
    print(f"  noise:                  {NOISE_LEVEL}")
    print(f"  debug data dir:         {DEBUG_DATA_DIR}")
    print(f"  metadata output:        {output_json}")
    print("=" * 70)

    # --- Load data ---
    if INPUT_CONFIG and PredatorDataLoader is not None:
        # Predator dataset mode
        print(f"\nLoading Predator dataset (index={INPUT_INDEX}, split={INPUT_DATA_SPLIT})...")
        loader = PredatorDataLoader(INPUT_CONFIG, split=INPUT_DATA_SPLIT)
        if INPUT_INDEX >= len(loader):
            raise ValueError(f"Index {INPUT_INDEX} out of range for dataset of size {len(loader)}")

        inputs = loader.get_by_index(INPUT_INDEX)
        pcd1 = o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(inputs["src_pcd_raw"])
        pcd2.points = o3d.utility.Vector3dVector(inputs["tgt_pcd_raw"])
        gt_rot = inputs["rot"]
        gt_trans = np.asarray(inputs["trans"]).squeeze()

        print(f"  Point cloud 1: {len(pcd1.points)} points")
        print(f"  Point cloud 2: {len(pcd2.points)} points")

    elif INPUT_VOXEL1 and INPUT_VOXEL2:
        # Raw voxel files mode
        print(f"\nLoading voxel files...")
        voxel1 = np.load(INPUT_VOXEL1)
        voxel2 = np.load(INPUT_VOXEL2)
        print(f"  Voxel 1 shape: {voxel1.shape}, dtype: {voxel1.dtype}, range: [{voxel1.min():.4f}, {voxel1.max():.4f}]")
        print(f"  Voxel 2 shape: {voxel2.shape}, dtype: {voxel2.dtype}, range: [{voxel2.min():.4f}, {voxel2.max():.4f}]")

        # Create dummy point clouds for shift computation (use full extent)
        pcd1 = o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()
        indices1 = np.argwhere(voxel1 > 0)
        indices2 = np.argwhere(voxel2 > 0)
        if len(indices1) > 0:
            pcd1.points = o3d.utility.Vector3dVector(indices1.astype(float))
        if len(indices2) > 0:
            pcd2.points = o3d.utility.Vector3dVector(indices2.astype(float))

        gt_rot = np.eye(3)
        gt_trans = np.zeros(3)

    elif INPUT_PCD1 and INPUT_PCD2:
        # Raw point cloud files mode
        print(f"\nLoading point cloud files...")
        pcd1 = o3d.io.read_point_cloud(INPUT_PCD1)
        pcd2 = o3d.io.read_point_cloud(INPUT_PCD2)
        print(f"  Point cloud 1: {len(pcd1.points)} points")
        print(f"  Point cloud 2: {len(pcd2.points)} points")

        gt_rot = np.eye(3)
        gt_trans = np.zeros(3)

    else:
        print("ERROR: No input mode selected.")
        print("Set one of: INPUT_CONFIG, INPUT_VOXEL1+INPUT_VOXEL2, INPUT_PCD1+INPUT_PCD2")
        print("at the top of this file.")
        sys.exit(1)

    # --- Compute means from ORIGINAL point clouds (before noise) ---
    mean1, _ = o3d.geometry.PointCloud.compute_mean_and_covariance(pcd1)
    mean2, _ = o3d.geometry.PointCloud.compute_mean_and_covariance(pcd2)
    mean1 = np.squeeze(np.asarray(mean1))
    mean2 = np.squeeze(np.asarray(mean2))

    # --- Add noise ---
    if NOISE_LEVEL != "None" and NOISE_LEVEL != "none":
        print(f"\nAdding noise: {NOISE_LEVEL}")
        pcd1 = add_noise_to_pcd(pcd1, NOISE_LEVEL)
        pcd2 = add_noise_to_pcd(pcd2, NOISE_LEVEL)

    # Center point clouds by subtracting mean
    pcd1_centered = o3d.geometry.PointCloud()
    pcd1_centered.points = o3d.utility.Vector3dVector(np.asarray(pcd1.points) - mean1)
    pcd2_centered = o3d.geometry.PointCloud()
    pcd2_centered.points = o3d.utility.Vector3dVector(np.asarray(pcd2.points) - mean2)

    # Compute voxel size based on extent
    max_distance_1 = np.max(np.abs(np.asarray(pcd1_centered.points)))
    max_distance_2 = np.max(np.abs(np.asarray(pcd2_centered.points)))
    # max_distance_1 = np.max(np.asarray(pcd1_centered.points))
    # max_distance_2 = np.max(np.asarray(pcd2_centered.points))
    max_distance = max(max_distance_1, max_distance_2)
    voxel_size = (2 * max_distance * 1.001) / (N-2)
    print(f"\nVoxelization parameters:")
    print(f"  mean1: [{mean1[0]:.4f}, {mean1[1]:.4f}, {mean1[2]:.4f}]")
    print(f"  mean2: [{mean2[0]:.4f}, {mean2[1]:.4f}, {mean2[2]:.4f}]")
    print(f"  voxel_size: {voxel_size:.6f}")
    print(f"  max_distance: {max_distance:.4f}")

    # --- Build mean transforms ---
    mean1_transform = np.eye(4)
    mean2_transform = np.eye(4)
    mean1_transform[:3, 3] = -mean1
    mean2_transform[:3, 3] = -mean2

    # --- GT transformation ---
    gt_transform = np.eye(4)
    gt_transform[:3, :3] = gt_rot
    gt_transform[:3, 3] = gt_trans

    # --- Voxelize ---
    print(f"\nVoxelizing point clouds...")
    voxel_array1 = point_to_voxel(pcd1_centered, N, voxel_size, np.zeros(3)).astype(np.float64)
    voxel_array2 = point_to_voxel(pcd2_centered, N, voxel_size, np.zeros(3)).astype(np.float64)
    print(f"  Voxel 1: {np.sum(voxel_array1 > 0)} occupied cells")
    print(f"  Voxel 2: {np.sum(voxel_array2 > 0)} occupied cells")

    # --- Initialize registration ---
    print(f"\nInitializing SoftRegistrationWrapper(N={N}, bwOut={N//2}, bwIn={N//2}, degLim={N//2-1})...")
    reg = SoftRegistrationWrapper(N, N // 2, N // 2, N // 2 - 1)
    print("  Ready.")

    # --- Run registration ---
    print(f"\nRunning 3D SOFT registration (all solutions, debug=True)...")
    start_time = time.time()

    list_peaks = reg.registerVoxels(
        voxel_array1, voxel_array2,
        debug=True,
        useClahe=USE_CLAHE,
        timeStuff=False,
        sizeVoxel=voxel_size,
        r_min=float(r_min),
        r_max=float(r_max),
        level_potential_rotation=LEVEL_POTENTIAL_ROTATION,
        level_potential_translation=LEVEL_POTENTIAL_TRANSLATION,
        set_r_manual=set_r_manual,
        normalization=NORMALIZATION
    )

    elapsed = time.time() - start_time
    print(f"\nRegistration completed in {elapsed:.2f}s")
    total_solutions = sum(len(peak.potentialTranslations) for peak in list_peaks)
    print(f"  Rotation peaks found: {len(list_peaks)}")
    print(f"  Total solutions (rotation x translation): {total_solutions}")
    for i, peak in enumerate(list_peaks):
        total_trans = len(peak.potentialTranslations)
        print(f"    Solution {i}: {total_trans} translation peak(s), rotation corr={peak.potentialRotation.correlationHeight:.4f}")

    # --- Compute errors and build metadata ---
    print(f"\nComputing error metrics...")
    solutions_data = []
    for i, peak in enumerate(list_peaks):
        est_transform = compute_transformation_from_peak(peak, mean1_transform, mean2_transform)
        errors = compute_error_metrics(gt_transform, est_transform)

        solution = {
            'rotation_index': i,
            'rotation': {
                'w': float(peak.potentialRotation.w),
                'x': float(peak.potentialRotation.x),
                'y': float(peak.potentialRotation.y),
                'z': float(peak.potentialRotation.z),
                'correlation_height': float(peak.potentialRotation.correlationHeight),
                'persistence': float(peak.potentialRotation.persistence),
                'level_potential': float(peak.potentialRotation.levelPotential)
            },
            'estimated_transform': matrix_to_json(est_transform),
            'estimated_rpy_xyz': get_rpy_xyz(est_transform),
            'error': errors,
            'translations': []
        }

        for j, trans_peak in enumerate(peak.potentialTranslations):
            trans_info = {
                'index': j,
                'x_translation': float(trans_peak.xTranslation),
                'y_translation': float(trans_peak.yTranslation),
                'z_translation': float(trans_peak.zTranslation),
                'correlation_height': float(trans_peak.correlationHeight),
                'global_correlation_height': float(trans_peak.globalCorrelationHeight),
                'persistence': float(trans_peak.persistence),
                'level_potential': float(trans_peak.levelPotential)
            }
            solution['translations'].append(trans_info)

        solutions_data.append(solution)

    # --- Build metadata ---
    metadata = {
        'sample_index': INPUT_INDEX,
        'N': N,
        'voxel_size': float(voxel_size),
        'params': {
            'use_clahe': USE_CLAHE,
            'r_min': r_min,
            'r_max': r_max,
            'set_r_manual': set_r_manual,
            'level_potential_rotation': LEVEL_POTENTIAL_ROTATION,
            'level_potential_translation': LEVEL_POTENTIAL_TRANSLATION,
            'normalization': NORMALIZATION,
            'noise': NOISE_LEVEL
        },
        'gt_transform': matrix_to_json(gt_transform),
        'gt_rpy_xyz': get_rpy_xyz(gt_transform),
        'mean1': mean1.tolist(),
        'mean2': mean2.tolist(),
        'total_time_seconds': elapsed,
        'num_rotation_peaks': len(list_peaks),
        'solutions': solutions_data
    }

    # --- Save metadata ---
    output_dir = os.path.dirname(output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_json, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to: {output_json}")
    print(f"Debug CSV files written to: {DEBUG_DATA_DIR}")

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("REGISTRATION SUMMARY")
    print("=" * 70)
    print(f"GT Transform:")
    print(f"  Rotation (RPY): [{np.degrees(euler.mat2euler(gt_rot))[0]:.2f}, "
          f"{np.degrees(euler.mat2euler(gt_rot))[1]:.2f}, "
          f"{np.degrees(euler.mat2euler(gt_rot))[2]:.2f}] deg")
    print(f"  Translation:    [{gt_trans[0]:.4f}, {gt_trans[1]:.4f}, {gt_trans[2]:.4f}]")
    print(f"\n{'Sol#':<6} {'RotErr°':<10} {'TransErr(m)':<14} {'Rot Corr':<10} {'Trans Corr':<12} {'Persist':<10}")
    print("-" * 70)

    for sol in solutions_data:
        rot_err = sol['error']['rotation_error_deg']
        trans_err = sol['error']['translation_error_m']
        rot_corr = sol['rotation']['correlation_height']
        trans_corr = sol['translations'][0]['correlation_height'] if sol['translations'] else 0
        persist = sol['rotation']['persistence']
        print(f"{sol['rotation_index']:<6} {rot_err:<10.4f} {trans_err:<14.4f} {rot_corr:<10.4f} {trans_corr:<12.4f} {persist:<10.4f}")

    print("=" * 70)
    print("\nNext step: Open the Jupyter notebook to visualize the results.")
    print(f"  cd {os.path.dirname(os.path.abspath(output_json))}")
    print(f"  jupyter notebook plot_full_3d_transformation.ipynb")
    print()


if __name__ == '__main__':
    main()
