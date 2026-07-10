################################################################################
#
# Boreas Pair Viewer - Automated pair-by-pair registration with image saving
#
# Usage:
#     python viewBoreasPairs.py
#
# Saves display images to viewBoreasOutput/ folder:
#     image1.png - previous frame
#     image2.png - current frame
#     blended.png - warped overlay
#
# Edit config at top of file to change settings.
################################################################################

import os
import sys
import time
import inspect
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Any

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from boreasDatasetLoader import (
    BoreasSequence,
    load_sequence,
    load_single_sequence,
    get_affine_matrix,
    transform_diff,
)
from boreasRegistrationMethods import RegistrationFactory



# ============================================================================
# CONFIGURATION - Edit these to test different settings
# ============================================================================
DATA_DIR = "/home/tim-external/dataFolder/radar_boreas"
SEQUENCE_NUMBER = 0
SEQUENCE_NAME = "boreas-2020-11-26-13-58" # Sequence name string, e.g. 'boreas-2020-11-26-13-58'
REGISTRATION_METHOD = "akaze"  # Options: fs2d, icp, ndt_p2d, fourier_mellin, sift, surf, kaze, akaze


# FS2D-specific config
N = 256         #256 128               # Image grid size (N x N)
RADIUS = 150                   # Scene radius in meters (pixel_size = 2*radius/N computed automatically)
SIZE_OF_PIXEL = (2.0 * RADIUS) / N  # Computed from RADIUS and N
DEBUG_MODE = True
MATCHING_STEP = 1                # Match every Nth frame
START_FRAME = 50                  # First frame index; first pair = (START_FRAME, START_FRAME + MATCHING_STEP) good example: 3685
MAX_FRAMES = None                # None = full sequence, or cap it
OUTPUT_DIR = "viewBoreasOutput"  # Blended images saved here
USE_DIRECT = True               # Use direct registration (1-angle) vs SO3 (multiple angles)
LEVEL_POTENTIAL_ROTATION = 0.001  # Persistence threshold for rotation peak filtering
POTENTIAL_NECCESSARY_FOR_PEAK = 0.1  # 2D peak detection threshold
NORMALIZATION = 1  # 0=1, 1=1/sqrt(norm), 2=1/norm
USE_PHASE_CORRELATION = False  # If True, use phase correlation instead of standard cross-correlation
ROUND = False  # If True, apply circular mask (corners → 0)

# Raw point cloud config (used by ICP, NDT etc.)
USE_RAW_POINTCLOUD = True        # True = raw polar data (best), False = cartesian image
RAW_INTENSITY_THRESHOLD = 0.3    # Noise floor for raw polar (0.0 = all points)

# ICP-specific config
ICP_MAX_DISTANCE = 30.0
ICP_MAX_ITERATION = 200
ICP_SCALE = 1.0
ICP_THRESHOLD_PCT = 20.0
ICP_VOXEL_SIZE = 0.5             # Downsampling in meters (0 = skip)

# NDT_P2D-specific config
NDT_VOXEL_SIZE = 5.0
NDT_MAX_ITERATION = 50
NDT_TRANSFORMATION_EPSILON = 0.01
NDT_STEP_SIZE = 0.1
NDT_SCALE = 1.0
NDT_THRESHOLD_PCT = 5.0

# Fourier-Mellin config # N = 64/128/256 has impact on the result
FM_HIGHPASS = True

# SIFT-specific config
SIFT_NFEATURES = 0
SIFT_N_OCTAVE_LAYERS = 3
SIFT_CONTRAST_THRESHOLD = 0.08
SIFT_EDGE_THRESHOLD = 10
SIFT_SIGMA = 1.6
SIFT_RATIO_THRESHOLD = 0.75
SIFT_RANSAC_THRESHOLD = 5.0
SIFT_RANSAC_CONFIDENCE = 0.99

# SURF-specific config (requires opencv-contrib-python)
SURF_HESSIAN_THRESHOLD = 400
SURF_N_OCTAVES = 4
SURF_N_OCTAVE_LAYERS = 3
SURF_EXTENDED = True
SURF_UPRIGHT = False
SURF_RATIO_THRESHOLD = 0.75
SURF_RANSAC_THRESHOLD = 3.0
SURF_RANSAC_CONFIDENCE = 0.99

# KAZE-specific config
KAZE_EXTENDED = False
KAZE_UPRIGHT = False
KAZE_THRESHOLD = 0.001
KAZE_N_OCTAVES = 4
KAZE_N_OCTAVE_LAYERS = 4
KAZE_DIFFUSIVITY = 1
KAZE_RATIO_THRESHOLD = 0.75
KAZE_RANSAC_THRESHOLD = 3.0
KAZE_RANSAC_CONFIDENCE = 0.99

# AKAZE-specific config
AKAZE_DESCRIPTOR_TYPE = "MLDB"
AKAZE_DESCRIPTOR_SIZE = 0
AKAZE_DESCRIPTOR_CHANNELS = 3
AKAZE_THRESHOLD = 0.001
AKAZE_N_OCTAVES = 4
AKAZE_N_OCTAVE_LAYERS = 4
AKAZE_DIFFUSIVITY = 2
AKAZE_RATIO_THRESHOLD = 0.75
AKAZE_RANSAC_THRESHOLD = 3.0
AKAZE_RANSAC_CONFIDENCE = 0.99
# ============================================================================


def get_config_from_file():
    """Reload config from this file in case it was edited."""
    global DATA_DIR, SEQUENCE_NUMBER, SEQUENCE_NAME, N, RADIUS, SIZE_OF_PIXEL
    global MATCHING_STEP, START_FRAME, MAX_FRAMES, OUTPUT_DIR, USE_DIRECT, LEVEL_POTENTIAL_ROTATION, POTENTIAL_NECCESSARY_FOR_PEAK, ROUND
    global REGISTRATION_METHOD, USE_RAW_POINTCLOUD, RAW_INTENSITY_THRESHOLD
    global ICP_MAX_DISTANCE, ICP_MAX_ITERATION, ICP_SCALE, ICP_THRESHOLD_PCT, ICP_VOXEL_SIZE
    global NDT_VOXEL_SIZE, NDT_MAX_ITERATION, NDT_TRANSFORMATION_EPSILON, NDT_STEP_SIZE, NDT_SCALE, NDT_THRESHOLD_PCT
    global FM_HIGHPASS
    global SIFT_NFEATURES, SIFT_N_OCTAVE_LAYERS, SIFT_CONTRAST_THRESHOLD, SIFT_EDGE_THRESHOLD, SIFT_SIGMA
    global SIFT_RATIO_THRESHOLD, SIFT_RANSAC_THRESHOLD, SIFT_RANSAC_CONFIDENCE
    global SURF_HESSIAN_THRESHOLD, SURF_N_OCTAVES, SURF_N_OCTAVE_LAYERS, SURF_EXTENDED, SURF_UPRIGHT
    global SURF_RATIO_THRESHOLD, SURF_RANSAC_THRESHOLD, SURF_RANSAC_CONFIDENCE
    global KAZE_EXTENDED, KAZE_UPRIGHT, KAZE_THRESHOLD, KAZE_N_OCTAVES, KAZE_N_OCTAVE_LAYERS, KAZE_DIFFUSIVITY
    global KAZE_RATIO_THRESHOLD, KAZE_RANSAC_THRESHOLD, KAZE_RANSAC_CONFIDENCE
    global AKAZE_DESCRIPTOR_TYPE, AKAZE_DESCRIPTOR_SIZE, AKAZE_DESCRIPTOR_CHANNELS
    global AKAZE_THRESHOLD, AKAZE_N_OCTAVES, AKAZE_N_OCTAVE_LAYERS, AKAZE_DIFFUSIVITY
    global AKAZE_RATIO_THRESHOLD, AKAZE_RANSAC_THRESHOLD, AKAZE_RANSAC_CONFIDENCE

    source_file = inspect.getfile(inspect.currentframe())
    source_dir = os.path.dirname(os.path.abspath(source_file))
    config_path = os.path.join(source_dir, "viewBoreasPairs.py")

    with open(config_path, "r") as f:
        content = f.read()

    # Extract config values
    import re

    def extract_var(name, default):
        pattern = rf'^{name}\s*=\s*(.+)$'
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            value_str = match.group(1).split('#')[0].strip()
            # Try to parse
            if value_str == "None":
                return None
            try:
                return int(value_str)
            except ValueError:
                pass
            try:
                return float(value_str)
            except ValueError:
                pass
            if value_str.startswith('"') or value_str.startswith("'"):
                return value_str.strip('"').strip("'")
            if value_str in ("True", "False"):
                return value_str == "True"
        return default

    DATA_DIR = extract_var("DATA_DIR", DATA_DIR)
    SEQUENCE_NUMBER = extract_var("SEQUENCE_NUMBER", SEQUENCE_NUMBER)
    SEQUENCE_NAME = extract_var("SEQUENCE_NAME", SEQUENCE_NAME)
    N = extract_var("N", N)
    RADIUS = extract_var("RADIUS", RADIUS)
    SIZE_OF_PIXEL = (2.0 * RADIUS) / N
    MATCHING_STEP = extract_var("MATCHING_STEP", MATCHING_STEP)
    START_FRAME = extract_var("START_FRAME", START_FRAME)
    MAX_FRAMES = extract_var("MAX_FRAMES", MAX_FRAMES)
    OUTPUT_DIR = extract_var("OUTPUT_DIR", OUTPUT_DIR)
    USE_DIRECT = extract_var("USE_DIRECT", USE_DIRECT)
    LEVEL_POTENTIAL_ROTATION = extract_var("LEVEL_POTENTIAL_ROTATION", LEVEL_POTENTIAL_ROTATION)
    POTENTIAL_NECCESSARY_FOR_PEAK = extract_var("POTENTIAL_NECCESSARY_FOR_PEAK", POTENTIAL_NECCESSARY_FOR_PEAK)
    ROUND = extract_var("ROUND", ROUND)
    REGISTRATION_METHOD = extract_var("REGISTRATION_METHOD", REGISTRATION_METHOD)
    USE_RAW_POINTCLOUD = extract_var("USE_RAW_POINTCLOUD", USE_RAW_POINTCLOUD)
    RAW_INTENSITY_THRESHOLD = extract_var("RAW_INTENSITY_THRESHOLD", RAW_INTENSITY_THRESHOLD)
    ICP_MAX_DISTANCE = extract_var("ICP_MAX_DISTANCE", ICP_MAX_DISTANCE)
    ICP_MAX_ITERATION = extract_var("ICP_MAX_ITERATION", ICP_MAX_ITERATION)
    ICP_SCALE = extract_var("ICP_SCALE", ICP_SCALE)
    ICP_THRESHOLD_PCT = extract_var("ICP_THRESHOLD_PCT", ICP_THRESHOLD_PCT)
    ICP_VOXEL_SIZE = extract_var("ICP_VOXEL_SIZE", ICP_VOXEL_SIZE)
    NDT_VOXEL_SIZE = extract_var("NDT_VOXEL_SIZE", NDT_VOXEL_SIZE)
    NDT_MAX_ITERATION = extract_var("NDT_MAX_ITERATION", NDT_MAX_ITERATION)
    NDT_TRANSFORMATION_EPSILON = extract_var("NDT_TRANSFORMATION_EPSILON", NDT_TRANSFORMATION_EPSILON)
    NDT_STEP_SIZE = extract_var("NDT_STEP_SIZE", NDT_STEP_SIZE)
    NDT_SCALE = extract_var("NDT_SCALE", NDT_SCALE)
    NDT_THRESHOLD_PCT = extract_var("NDT_THRESHOLD_PCT", NDT_THRESHOLD_PCT)
    FM_HIGHPASS = extract_var("FM_HIGHPASS", FM_HIGHPASS)
    SIFT_NFEATURES = extract_var("SIFT_NFEATURES", SIFT_NFEATURES)
    SIFT_N_OCTAVE_LAYERS = extract_var("SIFT_N_OCTAVE_LAYERS", SIFT_N_OCTAVE_LAYERS)
    SIFT_CONTRAST_THRESHOLD = extract_var("SIFT_CONTRAST_THRESHOLD", SIFT_CONTRAST_THRESHOLD)
    SIFT_EDGE_THRESHOLD = extract_var("SIFT_EDGE_THRESHOLD", SIFT_EDGE_THRESHOLD)
    SIFT_SIGMA = extract_var("SIFT_SIGMA", SIFT_SIGMA)
    SIFT_RATIO_THRESHOLD = extract_var("SIFT_RATIO_THRESHOLD", SIFT_RATIO_THRESHOLD)
    SIFT_RANSAC_THRESHOLD = extract_var("SIFT_RANSAC_THRESHOLD", SIFT_RANSAC_THRESHOLD)
    SIFT_RANSAC_CONFIDENCE = extract_var("SIFT_RANSAC_CONFIDENCE", SIFT_RANSAC_CONFIDENCE)
    SURF_HESSIAN_THRESHOLD = extract_var("SURF_HESSIAN_THRESHOLD", SURF_HESSIAN_THRESHOLD)
    SURF_N_OCTAVES = extract_var("SURF_N_OCTAVES", SURF_N_OCTAVES)
    SURF_N_OCTAVE_LAYERS = extract_var("SURF_N_OCTAVE_LAYERS", SURF_N_OCTAVE_LAYERS)
    SURF_EXTENDED = extract_var("SURF_EXTENDED", SURF_EXTENDED)
    SURF_UPRIGHT = extract_var("SURF_UPRIGHT", SURF_UPRIGHT)
    SURF_RATIO_THRESHOLD = extract_var("SURF_RATIO_THRESHOLD", SURF_RATIO_THRESHOLD)
    SURF_RANSAC_THRESHOLD = extract_var("SURF_RANSAC_THRESHOLD", SURF_RANSAC_THRESHOLD)
    SURF_RANSAC_CONFIDENCE = extract_var("SURF_RANSAC_CONFIDENCE", SURF_RANSAC_CONFIDENCE)
    KAZE_EXTENDED = extract_var("KAZE_EXTENDED", KAZE_EXTENDED)
    KAZE_UPRIGHT = extract_var("KAZE_UPRIGHT", KAZE_UPRIGHT)
    KAZE_THRESHOLD = extract_var("KAZE_THRESHOLD", KAZE_THRESHOLD)
    KAZE_N_OCTAVES = extract_var("KAZE_N_OCTAVES", KAZE_N_OCTAVES)
    KAZE_N_OCTAVE_LAYERS = extract_var("KAZE_N_OCTAVE_LAYERS", KAZE_N_OCTAVE_LAYERS)
    KAZE_DIFFUSIVITY = extract_var("KAZE_DIFFUSIVITY", KAZE_DIFFUSIVITY)
    KAZE_RATIO_THRESHOLD = extract_var("KAZE_RATIO_THRESHOLD", KAZE_RATIO_THRESHOLD)
    KAZE_RANSAC_THRESHOLD = extract_var("KAZE_RANSAC_THRESHOLD", KAZE_RANSAC_THRESHOLD)
    KAZE_RANSAC_CONFIDENCE = extract_var("KAZE_RANSAC_CONFIDENCE", KAZE_RANSAC_CONFIDENCE)
    AKAZE_DESCRIPTOR_TYPE = extract_var("AKAZE_DESCRIPTOR_TYPE", AKAZE_DESCRIPTOR_TYPE)
    AKAZE_DESCRIPTOR_SIZE = extract_var("AKAZE_DESCRIPTOR_SIZE", AKAZE_DESCRIPTOR_SIZE)
    AKAZE_DESCRIPTOR_CHANNELS = extract_var("AKAZE_DESCRIPTOR_CHANNELS", AKAZE_DESCRIPTOR_CHANNELS)
    AKAZE_THRESHOLD = extract_var("AKAZE_THRESHOLD", AKAZE_THRESHOLD)
    AKAZE_N_OCTAVES = extract_var("AKAZE_N_OCTAVES", AKAZE_N_OCTAVES)
    AKAZE_N_OCTAVE_LAYERS = extract_var("AKAZE_N_OCTAVE_LAYERS", AKAZE_N_OCTAVE_LAYERS)
    AKAZE_DIFFUSIVITY = extract_var("AKAZE_DIFFUSIVITY", AKAZE_DIFFUSIVITY)
    AKAZE_RATIO_THRESHOLD = extract_var("AKAZE_RATIO_THRESHOLD", AKAZE_RATIO_THRESHOLD)
    AKAZE_RANSAC_THRESHOLD = extract_var("AKAZE_RANSAC_THRESHOLD", AKAZE_RANSAC_THRESHOLD)
    AKAZE_RANSAC_CONFIDENCE = extract_var("AKAZE_RANSAC_CONFIDENCE", AKAZE_RANSAC_CONFIDENCE)


PLOT_DATA_DIR = "/home/tim-external/ros_ws/src/fsregistration/plotting_results/2d/data"


def apply_circular_mask(image: np.ndarray) -> np.ndarray:
    """Zero out pixels outside the inscribed circle of a square image."""
    N = image.shape[0]
    cy = cx = N // 2
    radius = N // 2
    Y, X = np.ogrid[:N, :N]
    mask = (X - cx)**2 + (Y - cy)**2 <= radius**2
    return image * mask


def run_pair(
    seq: BoreasSequence,
    idx1: int,
    idx2: int,
    method: Any,
) -> Tuple:
    """Register one pair via method.register() and return (img1, img2, blended, result, gt_error)."""
    img1 = seq.get_cartesian_image(idx1, N, SIZE_OF_PIXEL)
    img2 = seq.get_cartesian_image(idx2, N, SIZE_OF_PIXEL)

    if ROUND:
        img1 = apply_circular_mask(img1)
        img2 = apply_circular_mask(img2)

    gt_transform = seq.get_gt_transform(idx1, idx2)
    gt_affine = get_affine_matrix(gt_transform)

    # Registration — use raw polar data for point-cloud methods if configured
    sig = inspect.signature(method.register)
    if USE_RAW_POINTCLOUD and "pcd1" in sig.parameters:
        raw1 = seq.get_raw_point_cloud(idx1, RAW_INTENSITY_THRESHOLD)
        raw2 = seq.get_raw_point_cloud(idx2, RAW_INTENSITY_THRESHOLD)
        result = method.register(img1, img2, pcd1=raw1, pcd2=raw2)
    else:
        result = method.register(img1, img2)
    transform = result.transform

    # Compute GT error
    est_affine = get_affine_matrix(transform)
    gt_error = transform_diff(gt_affine, est_affine)

    # Create blended image
    fs2d_affine = get_affine_matrix(transform, pixel_size=SIZE_OF_PIXEL, img_size=N)
    warped = cv2.warpPerspective(img2, fs2d_affine, (img1.shape[1], img1.shape[0]))
    blended = cv2.addWeighted((img1 * 255).astype(np.uint8), 0.5, (warped * 255).astype(np.uint8), 0.5, 0)

    # Extract GT yaw and translation from the 4x4 GT matrix
    gt_yaw = np.arctan2(gt_transform[1, 0], gt_transform[0, 0])
    if gt_yaw < 0:
        gt_yaw += 2 * np.pi
    gt_tx = gt_transform[0, 3]
    gt_ty = gt_transform[1, 3]

    return img1, img2, blended, result, gt_error, gt_yaw, gt_tx, gt_ty


def main():
    # Reload config from file (in case it was edited)
    get_config_from_file()

    print(f"Config:")
    print(f"  DATA_DIR: {DATA_DIR}")
    print(f"  SEQUENCE_NUMBER: {SEQUENCE_NUMBER}")
    print(f"  SEQUENCE_NAME: {SEQUENCE_NAME}")
    print(f"  DEBUG_MODE: {DEBUG_MODE}")
    print(f"  N: {N}, RADIUS: {RADIUS} m, pixel_size: {SIZE_OF_PIXEL:.3f} m")
    print(f"  MATCHING_STEP: {MATCHING_STEP}, START_FRAME: {START_FRAME}, MAX_FRAMES: {MAX_FRAMES}")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"  ROUND: {ROUND}")
    print()

    # Load sequence
    if SEQUENCE_NAME is not None:
        print(f"Loading sequence '{SEQUENCE_NAME}' from {DATA_DIR} (single sequence mode)...")
        seq = load_single_sequence(DATA_DIR, SEQUENCE_NAME)
    else:
        print(f"Loading sequence {SEQUENCE_NUMBER} from {DATA_DIR}...")
        seq = load_sequence(DATA_DIR, SEQUENCE_NUMBER)
    print(f"Sequence has {seq.length} radar scans")
    print()

    # Setup method — all keys are unique (method-prefixed to avoid collisions)
    method_config = {
        "N": N,
        "size_of_pixel": SIZE_OF_PIXEL,
        # ---- FS2D params ----
        "use_clahe": True,
        "use_hamming": True,
        "potential_for_necessary_peak": POTENTIAL_NECCESSARY_FOR_PEAK,
        "multiple_radii": True,
        "use_gauss": False,
        "use_direct": USE_DIRECT,
        "level_potential_rotation": LEVEL_POTENTIAL_ROTATION,
        "normalization": NORMALIZATION,
        "use_phase_correlation": USE_PHASE_CORRELATION,
        # ---- ICP params ----
        "icp_max_distance": ICP_MAX_DISTANCE,
        "icp_max_iteration": ICP_MAX_ITERATION,
        "icp_scale": ICP_SCALE,
        "icp_threshold_pct": ICP_THRESHOLD_PCT,
        "icp_voxel_size": ICP_VOXEL_SIZE,
        "initial_guess": np.eye(4),
        # ---- NDT params ----
        "ndt_voxel_size": NDT_VOXEL_SIZE,
        "ndt_max_iteration": NDT_MAX_ITERATION,
        "ndt_transformation_epsilon": NDT_TRANSFORMATION_EPSILON,
        "ndt_step_size": NDT_STEP_SIZE,
        "ndt_scale": NDT_SCALE,
        "ndt_threshold_pct": NDT_THRESHOLD_PCT,
        # ---- Fourier-Mellin params ----
        "fm_highpass": FM_HIGHPASS,
        # ---- SIFT params ----
        "sift_nfeatures": SIFT_NFEATURES,
        "sift_n_octave_layers": SIFT_N_OCTAVE_LAYERS,
        "sift_contrast_threshold": SIFT_CONTRAST_THRESHOLD,
        "sift_edge_threshold": SIFT_EDGE_THRESHOLD,
        "sift_sigma": SIFT_SIGMA,
        "sift_ratio_threshold": SIFT_RATIO_THRESHOLD,
        "sift_ransac_threshold": SIFT_RANSAC_THRESHOLD,
        "sift_ransac_confidence": SIFT_RANSAC_CONFIDENCE,
        # ---- SURF params ----
        "surf_hessian_threshold": SURF_HESSIAN_THRESHOLD,
        "surf_n_octaves": SURF_N_OCTAVES,
        "surf_n_octave_layers": SURF_N_OCTAVE_LAYERS,
        "surf_extended": SURF_EXTENDED,
        "surf_upright": SURF_UPRIGHT,
        "surf_ratio_threshold": SURF_RATIO_THRESHOLD,
        "surf_ransac_threshold": SURF_RANSAC_THRESHOLD,
        "surf_ransac_confidence": SURF_RANSAC_CONFIDENCE,
        # ---- KAZE params ----
        "kaze_extended": KAZE_EXTENDED,
        "kaze_upright": KAZE_UPRIGHT,
        "kaze_threshold": KAZE_THRESHOLD,
        "kaze_n_octaves": KAZE_N_OCTAVES,
        "kaze_n_octave_layers": KAZE_N_OCTAVE_LAYERS,
        "kaze_diffusivity": KAZE_DIFFUSIVITY,
        "kaze_ratio_threshold": KAZE_RATIO_THRESHOLD,
        "kaze_ransac_threshold": KAZE_RANSAC_THRESHOLD,
        "kaze_ransac_confidence": KAZE_RANSAC_CONFIDENCE,
        # ---- AKAZE params ----
        "akaze_descriptor_type": AKAZE_DESCRIPTOR_TYPE,
        "akaze_descriptor_size": AKAZE_DESCRIPTOR_SIZE,
        "akaze_descriptor_channels": AKAZE_DESCRIPTOR_CHANNELS,
        "akaze_threshold": AKAZE_THRESHOLD,
        "akaze_n_octaves": AKAZE_N_OCTAVES,
        "akaze_n_octave_layers": AKAZE_N_OCTAVE_LAYERS,
        "akaze_diffusivity": AKAZE_DIFFUSIVITY,
        "akaze_ratio_threshold": AKAZE_RATIO_THRESHOLD,
        "akaze_ransac_threshold": AKAZE_RANSAC_THRESHOLD,
        "akaze_ransac_confidence": AKAZE_RANSAC_CONFIDENCE,
    }
    method = RegistrationFactory.create(REGISTRATION_METHOD, method_config)
    print(f"Method: {REGISTRATION_METHOD} (N={N}, pixel_size={SIZE_OF_PIXEL:.3f} m)")
    print()

    # Setup output directory
    save_dir = Path(OUTPUT_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {save_dir}")
    print()

    # Determine number of frames
    length_of_radar_scans = seq.length
    if MAX_FRAMES is not None:
        length_of_radar_scans = min(length_of_radar_scans, MAX_FRAMES)
    print(f"Matching every {MATCHING_STEP}th image (from frame {START_FRAME}, up to {length_of_radar_scans})")
    print(f"Display images saved to: {save_dir}/")
    print("Image files: image1.png, image2.png, blended.png (overwritten each pair)")
    print("=" * 80)
    print()

    idx = START_FRAME + MATCHING_STEP

    while idx < length_of_radar_scans:
        prev_idx = idx - MATCHING_STEP

        print(f"\n--- Pair: {prev_idx} -> {idx} ---")

        # Run registration
        img1, img2, blended, result, gt_error, gt_yaw, gt_tx, gt_ty = run_pair(seq, prev_idx, idx, method)

        # Save display images (overwrite each time)
        cv2.imwrite(str(save_dir / "image1.png"), cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        cv2.imwrite(str(save_dir / "image2.png"), cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        cv2.imwrite(str(save_dir / "blended.png"), blended)

        # Extract estimated yaw/translation from result.transform (method-agnostic)
        est_transform = result.transform
        est_yaw = np.arctan2(est_transform[1, 0], est_transform[0, 0])
        if est_yaw < 0:
            est_yaw += 2 * np.pi
        est_tx = est_transform[0, 3]
        est_ty = est_transform[1, 3]

        # Save current pair images and meta to plot data directory
        try:
            os.makedirs(PLOT_DATA_DIR, exist_ok=True)
            gt_trans, gt_rot = gt_error
            gt_trans_norm = np.linalg.norm(gt_trans)
            np.savetxt(os.path.join(PLOT_DATA_DIR, "input1.csv"), img1, fmt='%.10f', delimiter=' ')
            np.savetxt(os.path.join(PLOT_DATA_DIR, "input2.csv"), img2, fmt='%.10f', delimiter=' ')
            header = "frame1,frame2,rot_angle_deg,tx_m,ty_m,confidence,time_ms,gt_rot_err_deg,gt_trans_err_m,N,n_solutions,radius_m,pixel_size_m,method"
            row = [
                prev_idx, idx,
                est_yaw * 180 / np.pi,
                est_tx,
                est_ty,
                result.confidence,
                result.computation_time * 1000,
                abs(gt_rot),
                gt_trans_norm,
                N,
                result.metadata.get('num_solutions', 0),
                RADIUS,
                SIZE_OF_PIXEL,
                result.method_name,
            ]
            with open(os.path.join(PLOT_DATA_DIR, "registration_meta.csv"), 'w') as f:
                f.write(header + '\n')
                f.write(','.join(f'{v:.10g}' if isinstance(v, (int, float)) else str(v) for v in row) + '\n')
        except Exception as e:
            print(f"  [WARN] Could not save plot data: {e}")

        # Print results
        gt_trans, gt_rot = gt_error
        gt_trans_norm = np.linalg.norm(gt_trans)
        print(f"  GT Rot: {gt_yaw:.4f}, Est Rot: {est_yaw:.4f}")
        print(f"  GT Tx: {gt_tx:.4f} m, Est Tx: {est_tx:.4f} m   GT Ty: {gt_ty:.4f} m, Est Ty: {est_ty:.4f} m")
        print(f"  Confidence: {result.confidence:.4f}")
        print(f"  Time: {result.computation_time * 1000:.1f} ms")
        print(f"  GT RotErr: {abs(gt_rot):.4f} deg, GT TransErr: {gt_trans_norm:.4f} m")
        print(f"  -> Saved to {save_dir}/ (image1.png, image2.png, blended.png)")
        
        
        idx += MATCHING_STEP

    print("\nDone.")


if __name__ == "__main__":
    main()
