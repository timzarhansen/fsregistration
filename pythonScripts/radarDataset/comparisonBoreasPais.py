################################################################################
#
# Comparison Boreas Pairs - run ALL registration methods on each frame pair,
# save the two input point clouds (as .ply and rendered .png) with every
# method's result and the ground truth overlaid.
#
# Usage:
#     python comparisonBoreasPais.py
#
# Output per pair (idx = first frame of the pair):
#     2D_registration_results/pairComparison/{idx}/{SEQUENCE_NAME}/
#         input1.png, input2.png           - cartesian radar images
#         pointcloud1.ply, pointcloud2.ply - raw point clouds (x, y, intensity)
#         pointcloud1.png, pointcloud2.png - point clouds rendered as images
#         gt.ply, gt.png                   - cloud1 (red) + cloud2 aligned by GT (green)
#         {method}.ply, {method}.png       - cloud1 + cloud2 aligned per method
#         registration_meta.csv            - all results in one table
#         summary.png                      - montage of all overlays
#
# Edit config at top of file to change settings.
################################################################################

import os
import sys
import time
import inspect
import re
import ast
import cv2
import numpy as np
from pathlib import Path

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
METHODS_TO_RUN = ["fs2d", "icp", "ndt_p2d", "fourier_mellin", "sift", "kaze", "akaze", "loftr", "eloftr", "lightglue"]  # All except SURF


# FS2D-specific config
N = 256         #256 128               # Image grid size (N x N)
RADIUS = 140                   # Scene radius in meters (pixel_size = 2*radius/N computed automatically) 140  # matches boreas2d step-3 benchmark
SIZE_OF_PIXEL = (2.0 * RADIUS) / N  # Computed from RADIUS and N
DEBUG_MODE = True
MATCHING_STEP = 5                # Match every Nth frame
START_FRAME = 0                  # First frame index; first pair = (START_FRAME, START_FRAME + MATCHING_STEP) good example: 3685
MAX_FRAMES = None                # None = full sequence, or cap it
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2D_registration_results", "pairComparison")  # Base output dir; per-pair subfolder is added automatically
USE_DIRECT = True               # Use direct registration (1-angle) vs SO3 (multiple angles)
NUM_ANGLES = -1              # Number of angles sampled for the direct 1D correlation curve; -1 = auto (N)
LEVEL_POTENTIAL_ROTATION = 0.001  # Persistence threshold for rotation peak filtering
POTENTIAL_NECCESSARY_FOR_PEAK = 0.01  # 2D peak detection threshold  # matches boreas2d step-3 benchmark
R_MIN = 0.0                  # Min radial frequency radius for FS2D (FFT grid units / px); 0.0 = auto (N-dependent default)
R_MAX = 0.0                  # Max radial frequency radius for FS2D (FFT grid units / px); 0.0 = auto (N-dependent default)
NORMALIZATION = 1  # 0=1, 1=1/sqrt(norm), 2=1/norm
USE_PHASE_CORRELATION = False  # If True, use phase correlation instead of standard cross-correlation
ROUND = False  # If True, apply circular mask (corners → 0)

# Raw point cloud config (used by ICP, NDT and for saving point clouds)
USE_RAW_POINTCLOUD = True        # True = raw polar data (matches boreas2d benchmark; ICP/NDT use raw clouds)
RAW_INTENSITY_THRESHOLD = 0.3    # Noise floor for raw polar (0.0 = all points)
PLY_DOWNSAMPLE = 1               # Save every Nth point in .ply files (1 = all points)

# ICP-specific config
ICP_MAX_DISTANCE = 10.0
ICP_MAX_ITERATION = 200
ICP_SCALE = 1.0
ICP_THRESHOLD_PCT = 10.0  # matches boreas2d step-3 benchmark
ICP_VOXEL_SIZE = 1.0             # Downsampling in meters (0 = skip)

# NDT_P2D-specific config
# NOTE: NDT_VOXEL_SIZE is the NDT occupancy-grid resolution (coarse = fast),
# NOT a downsampling of the input cloud. Use NDT_DOWNSAMPLE_VOXEL for that.
NDT_VOXEL_SIZE = 15.0  # matches boreas2d step-3 benchmark
NDT_DOWNSAMPLE_VOXEL = 1.0  # input downsampling in meters, mirrors ICP_VOXEL_SIZE (0 = skip)
NDT_MAX_ITERATION = 50
NDT_TRANSFORMATION_EPSILON = 0.01
NDT_STEP_SIZE = 1.0  # matches boreas2d step-3 benchmark
NDT_SCALE = 1.0
NDT_THRESHOLD_PCT = 5.0
NDT_Z_SCALE = 0.1  # matches boreas2d step-3 benchmark

# Fourier-Mellin config # N = 64/128/256 has impact on the result
FM_HIGHPASS = False  # matches boreas2d step-3 benchmark

# SIFT-specific config
SIFT_NFEATURES = 0
SIFT_N_OCTAVE_LAYERS = 3
SIFT_CONTRAST_THRESHOLD = 0.01  # matches boreas2d step-3 benchmark
SIFT_EDGE_THRESHOLD = 10
SIFT_SIGMA = 1.2  # matches boreas2d step-3 benchmark
SIFT_RATIO_THRESHOLD = 0.6  # matches boreas2d step-3 benchmark
SIFT_RANSAC_THRESHOLD = 1.0  # matches boreas2d step-3 benchmark
SIFT_RANSAC_CONFIDENCE = 0.99

# KAZE-specific config
KAZE_EXTENDED = False
KAZE_UPRIGHT = True
KAZE_THRESHOLD = 0.0001
KAZE_N_OCTAVES = 4
KAZE_N_OCTAVE_LAYERS = 4
KAZE_DIFFUSIVITY = 3
KAZE_RATIO_THRESHOLD = 0.6
KAZE_RANSAC_THRESHOLD = 1.0
KAZE_RANSAC_CONFIDENCE = 0.99

# AKAZE-specific config
AKAZE_DESCRIPTOR_TYPE = "MLDB"
AKAZE_DESCRIPTOR_SIZE = 0
AKAZE_DESCRIPTOR_CHANNELS = 3
AKAZE_THRESHOLD = 0.0001
AKAZE_N_OCTAVES = 4
AKAZE_N_OCTAVE_LAYERS = 4
AKAZE_DIFFUSIVITY = 1
AKAZE_RATIO_THRESHOLD = 0.6
AKAZE_RANSAC_THRESHOLD = 1.0
AKAZE_RANSAC_CONFIDENCE = 0.99

# LoFTR-specific config
LOFTR_RANSAC_THRESHOLD = 5.0  # matches boreas2d step-3 benchmark
LOFTR_RANSAC_CONFIDENCE = 0.99
LOFTR_CONFIDENCE_THRESHOLD = 0.5

# EfficientLoFTR-specific config
ELOFTR_MODEL_TYPE = "full"
ELOFTR_RANSAC_THRESHOLD = 5.0  # matches boreas2d step-3 benchmark
ELOFTR_RANSAC_CONFIDENCE = 0.99
ELOFTR_CONFIDENCE_THRESHOLD = 0.5

# LightGlue-specific config
LIGHTGLUE_FEATURES = "superpoint"
LIGHTGLUE_MAX_NUM_KEYPOINTS = 2048
LIGHTGLUE_DEPTH_CONFIDENCE = 0.95
LIGHTGLUE_WIDTH_CONFIDENCE = -1  # matches boreas2d step-3 benchmark
LIGHTGLUE_FILTER_THRESHOLD = 0.1
LIGHTGLUE_RANSAC_THRESHOLD = 3.0
LIGHTGLUE_RANSAC_CONFIDENCE = 0.99
# ============================================================================

# Names that get_config_from_file() reloads from this file (in case it was edited mid-run)
_RELOAD_NAMES = [
    "DATA_DIR", "SEQUENCE_NUMBER", "SEQUENCE_NAME", "METHODS_TO_RUN",
    "N", "RADIUS", "MATCHING_STEP", "START_FRAME", "MAX_FRAMES", "OUTPUT_DIR",
    "USE_DIRECT", "LEVEL_POTENTIAL_ROTATION", "POTENTIAL_NECCESSARY_FOR_PEAK",
    "NUM_ANGLES", "R_MIN", "R_MAX",
    "NORMALIZATION", "USE_PHASE_CORRELATION", "ROUND",
    "USE_RAW_POINTCLOUD", "RAW_INTENSITY_THRESHOLD", "PLY_DOWNSAMPLE",
    "ICP_MAX_DISTANCE", "ICP_MAX_ITERATION", "ICP_SCALE", "ICP_THRESHOLD_PCT", "ICP_VOXEL_SIZE",
    "NDT_VOXEL_SIZE", "NDT_MAX_ITERATION", "NDT_TRANSFORMATION_EPSILON", "NDT_STEP_SIZE", "NDT_SCALE", "NDT_THRESHOLD_PCT",
    "NDT_Z_SCALE",
    "FM_HIGHPASS",
    "SIFT_NFEATURES", "SIFT_N_OCTAVE_LAYERS", "SIFT_CONTRAST_THRESHOLD", "SIFT_EDGE_THRESHOLD", "SIFT_SIGMA",
    "SIFT_RATIO_THRESHOLD", "SIFT_RANSAC_THRESHOLD", "SIFT_RANSAC_CONFIDENCE",
    "KAZE_EXTENDED", "KAZE_UPRIGHT", "KAZE_THRESHOLD", "KAZE_N_OCTAVES", "KAZE_N_OCTAVE_LAYERS", "KAZE_DIFFUSIVITY",
    "KAZE_RATIO_THRESHOLD", "KAZE_RANSAC_THRESHOLD", "KAZE_RANSAC_CONFIDENCE",
    "AKAZE_DESCRIPTOR_TYPE", "AKAZE_DESCRIPTOR_SIZE", "AKAZE_DESCRIPTOR_CHANNELS",
    "AKAZE_THRESHOLD", "AKAZE_N_OCTAVES", "AKAZE_N_OCTAVE_LAYERS", "AKAZE_DIFFUSIVITY",
    "AKAZE_RATIO_THRESHOLD", "AKAZE_RANSAC_THRESHOLD", "AKAZE_RANSAC_CONFIDENCE",
    "LOFTR_RANSAC_THRESHOLD", "LOFTR_RANSAC_CONFIDENCE", "LOFTR_CONFIDENCE_THRESHOLD",
    "ELOFTR_MODEL_TYPE", "ELOFTR_RANSAC_THRESHOLD", "ELOFTR_RANSAC_CONFIDENCE", "ELOFTR_CONFIDENCE_THRESHOLD",
    "LIGHTGLUE_FEATURES", "LIGHTGLUE_MAX_NUM_KEYPOINTS",
    "LIGHTGLUE_DEPTH_CONFIDENCE", "LIGHTGLUE_WIDTH_CONFIDENCE", "LIGHTGLUE_FILTER_THRESHOLD",
    "LIGHTGLUE_RANSAC_THRESHOLD", "LIGHTGLUE_RANSAC_CONFIDENCE",
]


def get_config_from_file():
    """Reload config from this file in case it was edited mid-run."""
    global SIZE_OF_PIXEL
    source_file = inspect.getfile(inspect.currentframe())
    with open(source_file) as f:
        content = f.read()
    for name in _RELOAD_NAMES:
        m = re.search(rf'^{name}\s*=\s*(.+)$', content, re.MULTILINE)
        if not m:
            continue
        raw = m.group(1).split('#')[0].strip()
        try:
            globals()[name] = ast.literal_eval(raw)
        except Exception:
            pass
    SIZE_OF_PIXEL = (2.0 * RADIUS) / N


# ============================================================================
# Point cloud helpers
# ============================================================================

def apply_circular_mask(image: np.ndarray) -> np.ndarray:
    """Zero out pixels outside the inscribed circle of a square image."""
    N = image.shape[0]
    cy = cx = N // 2
    radius = N // 2
    Y, X = np.ogrid[:N, :N]
    mask = (X - cx)**2 + (Y - cy)**2 <= radius**2
    return image * mask


def pc_to_veh(pts: np.ndarray) -> np.ndarray:
    """Convert points from point-cloud frame to vehicle frame.

    Boreas pc convention: pc_x = +right, pc_y = -forward.
    Vehicle convention: veh_x = forward, veh_y = left.
    -> veh_x = -pc_y, veh_y = -pc_x  (z = 0).
    """
    out = np.zeros((len(pts), 3), np.float64)
    out[:, 0] = -pts[:, 1]
    out[:, 1] = -pts[:, 0]
    return out


def veh_to_pc(pts: np.ndarray) -> np.ndarray:
    """Inverse of pc_to_veh: pc_x = -veh_y, pc_y = -veh_x."""
    out = np.zeros((len(pts), 3), np.float64)
    out[:, 0] = -pts[:, 1]
    out[:, 1] = -pts[:, 0]
    return out


def align_points(pts_pc: np.ndarray, T_veh: np.ndarray) -> np.ndarray:
    """Transform points from frame2's pc frame into frame1's pc frame.

    T_veh maps frame2 -> frame1 in the vehicle frame (matching the GT and
    all registration method transforms).
    """
    v = pc_to_veh(pts_pc)
    hom = np.column_stack([v, np.ones(len(v))])
    v_al = (np.asarray(T_veh, np.float64) @ hom.T).T[:, :3]
    return veh_to_pc(v_al)


def render_points(pts: np.ndarray, intensity: np.ndarray, N: int, pixel_size: float) -> np.ndarray:
    """Scatter (x, y) pc-frame points onto an N x N image; returns float [0, 1].

    Image convention (matches pyboreas cartesian): row = -x_veh/cs (forward up),
    col = y_veh/cs (left right). With veh_x = -pc_y and veh_y = -pc_x:
        row = N/2 + pc_y/cs, col = N/2 - pc_x/cs.
    """
    row = N / 2.0 + pts[:, 1] / pixel_size
    col = N / 2.0 - pts[:, 0] / pixel_size
    r = np.clip(np.round(row).astype(np.int64), 0, N - 1)
    c = np.clip(np.round(col).astype(np.int64), 0, N - 1)
    img = np.zeros((N, N), np.float64)
    np.add.at(img, (r, c), intensity)
    # Intensity is already normalized to [0, 1] by the loader; accumulated
    # overlaps can exceed 1, so clip instead of re-normalizing (dividing by
    # the max would crush low-intensity cells to invisibility).
    return np.clip(img, 0.0, 1.0)


def overlay_image(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Red = cloud1, green = cloud2 (aligned), yellow = overlap."""
    out = np.zeros((img1.shape[0], img1.shape[1], 3), np.uint8)
    out[..., 2] = (np.clip(img1, 0, 1) * 255).astype(np.uint8)  # R = cloud1
    out[..., 1] = (np.clip(img2, 0, 1) * 255).astype(np.uint8)  # G = cloud2
    return out


def write_ply(path, pts, intensity, colors):
    """Write a binary little-endian PLY file.

    Args:
        path: Output file path.
        pts: (M, 2) or (M, 3) x, y[, z] coordinates.
        intensity: (M,) float intensity values.
        colors: (M, 3) uint8 RGB colors.
    """
    pts = np.asarray(pts, np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 1)
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts), np.float64)])
    intensity = np.asarray(intensity, np.float64).reshape(-1)
    colors = np.asarray(colors, np.uint8).reshape(-1, 3)
    n = len(pts)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float intensity\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                      ('intensity', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    arr = np.empty(n, dtype)
    arr['x'], arr['y'], arr['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
    arr['intensity'] = intensity
    arr['red'], arr['green'], arr['blue'] = colors[:, 0], colors[:, 1], colors[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


def write_aligned_ply(path, raw1, raw2, T_veh, downsample=1):
    """Save merged point cloud: cloud1 (red) + cloud2 aligned by T_veh (green)."""
    p1 = raw1[::downsample]
    p2 = raw2[::downsample]
    aligned2 = align_points(p2[:, :2], T_veh)
    pts = np.vstack([p1[:, :2], aligned2[:, :2]])
    inten = np.concatenate([p1[:, 2], p2[:, 2]])
    colors = np.vstack([
        np.full((len(p1), 3), (255, 0, 0), np.uint8),
        np.full((len(p2), 3), (0, 255, 0), np.uint8),
    ])
    write_ply(path, pts, inten, colors)


def make_montage(items, cell=256, per_row=6):
    """Tile (name, image) pairs into a labeled grid."""
    rows = []
    for i in range(0, len(items), per_row):
        chunk = items[i:i + per_row]
        imgs = []
        for name, img in chunk:
            im = img.copy()
            if im.shape[0] != cell:
                im = cv2.resize(im, (cell, cell))
            cv2.putText(im, name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            imgs.append(im)
        while len(imgs) < per_row:  # pad incomplete rows so vstack works
            imgs.append(np.zeros((cell, cell, 3), np.uint8))
        rows.append(np.hstack(imgs))
    return np.vstack(rows)


# ============================================================================
# Registration
# ============================================================================

def build_methods():
    """Create all requested registration methods (skip ones that fail to construct)."""
    method_config = {
        "N": N,
        "size_of_pixel": SIZE_OF_PIXEL,
        # ---- FS2D params ----
        "use_clahe": False,  # matches boreas2d step-3 benchmark (use_clahe=False)
        "use_hamming": True,
        "potential_for_necessary_peak": POTENTIAL_NECCESSARY_FOR_PEAK,
        "multiple_radii": True,
        "use_gauss": False,
        "use_direct": USE_DIRECT,
        "num_angles": NUM_ANGLES,
        "r_min": R_MIN,
        "r_max": R_MAX,
        "level_potential_rotation": LEVEL_POTENTIAL_ROTATION,
        "normalization": NORMALIZATION,
        "use_phase_correlation": USE_PHASE_CORRELATION,
        "use_weighted_peak_score": True,  # matches boreas2d step-3 benchmark
        "debug": DEBUG_MODE,
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
        "ndt_z_scale": NDT_Z_SCALE,
        "ndt_downsample_voxel": NDT_DOWNSAMPLE_VOXEL,
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
        # ---- LoFTR params ----
        "loftr_ransac_threshold": LOFTR_RANSAC_THRESHOLD,
        "loftr_ransac_confidence": LOFTR_RANSAC_CONFIDENCE,
        "loftr_confidence_threshold": LOFTR_CONFIDENCE_THRESHOLD,
        # ---- EfficientLoFTR params ----
        "eloftr_model_type": ELOFTR_MODEL_TYPE,
        "eloftr_ransac_threshold": ELOFTR_RANSAC_THRESHOLD,
        "eloftr_ransac_confidence": ELOFTR_RANSAC_CONFIDENCE,
        "eloftr_confidence_threshold": ELOFTR_CONFIDENCE_THRESHOLD,
        # ---- LightGlue params ----
        "lightglue_features": LIGHTGLUE_FEATURES,
        "lightglue_max_num_keypoints": LIGHTGLUE_MAX_NUM_KEYPOINTS,
        "lightglue_depth_confidence": LIGHTGLUE_DEPTH_CONFIDENCE,
        "lightglue_width_confidence": LIGHTGLUE_WIDTH_CONFIDENCE,
        "lightglue_filter_threshold": LIGHTGLUE_FILTER_THRESHOLD,
        "lightglue_ransac_threshold": LIGHTGLUE_RANSAC_THRESHOLD,
        "lightglue_ransac_confidence": LIGHTGLUE_RANSAC_CONFIDENCE,
    }
    methods = {}
    for name in METHODS_TO_RUN:
        try:
            methods[name] = RegistrationFactory.create(name, method_config)
            print(f"  [OK] Method '{name}' created")
        except Exception as e:
            print(f"  [WARN] Method '{name}' could not be created: {e}")
    return methods


def run_pair(seq: BoreasSequence, idx1: int, idx2: int, methods: dict):
    """Register the pair with every method.

    Returns (img1, img2, raw1, raw2, gt_transform, results) where results is
    {method_name: (result, rot_err_deg, trans_err_m) or None on failure}.
    """
    img1 = seq.get_cartesian_image(idx1, N, SIZE_OF_PIXEL)
    img2 = seq.get_cartesian_image(idx2, N, SIZE_OF_PIXEL)

    if ROUND:
        img1 = apply_circular_mask(img1)
        img2 = apply_circular_mask(img2)

    gt_transform = seq.get_gt_transform(idx1, idx2)
    gt_affine = get_affine_matrix(gt_transform)

    raw1 = seq.get_raw_point_cloud(idx1, RAW_INTENSITY_THRESHOLD)
    raw2 = seq.get_raw_point_cloud(idx2, RAW_INTENSITY_THRESHOLD)

    results = {}
    for name, method in methods.items():
        try:
            sig = inspect.signature(method.register)
            if USE_RAW_POINTCLOUD and "pcd1" in sig.parameters:
                result = method.register(img1, img2, pcd1=raw1, pcd2=raw2)
            else:
                result = method.register(img1, img2)
            est_affine = get_affine_matrix(result.transform)
            gt_trans, gt_rot = transform_diff(gt_affine, est_affine)
            results[name] = (result, abs(gt_rot), np.linalg.norm(gt_trans))
        except Exception as e:
            print(f"    [WARN] Method '{name}' failed on pair {idx1}->{idx2}: {e}")
            results[name] = None

    return img1, img2, raw1, raw2, gt_transform, results


def save_pair_output(pair_dir: Path, idx1: int, idx2: int,
                     img1, img2, raw1, raw2, gt_transform, results: dict):
    """Save all images, point clouds and meta for one pair."""
    pair_dir.mkdir(parents=True, exist_ok=True)

    # Input cartesian images
    cv2.imwrite(str(pair_dir / "input1.png"), cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    cv2.imwrite(str(pair_dir / "input2.png"), cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))

    # Raw point clouds (.ply) + rendered as images
    write_ply(pair_dir / "pointcloud1.ply", raw1[::PLY_DOWNSAMPLE][:, :2], raw1[::PLY_DOWNSAMPLE][:, 2],
              np.full((len(raw1[::PLY_DOWNSAMPLE]), 3), (255, 255, 255), np.uint8))
    write_ply(pair_dir / "pointcloud2.ply", raw2[::PLY_DOWNSAMPLE][:, :2], raw2[::PLY_DOWNSAMPLE][:, 2],
              np.full((len(raw2[::PLY_DOWNSAMPLE]), 3), (255, 255, 255), np.uint8))
    r1 = render_points(raw1[:, :2], raw1[:, 2], N, SIZE_OF_PIXEL)
    r2 = render_points(raw2[:, :2], raw2[:, 2], N, SIZE_OF_PIXEL)
    cv2.imwrite(str(pair_dir / "pointcloud1.png"), cv2.cvtColor((r1 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    cv2.imwrite(str(pair_dir / "pointcloud2.png"), cv2.cvtColor((r2 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))

    # Ground truth alignment
    r2_gt = render_points(align_points(raw2[:, :2], gt_transform), raw2[:, 2], N, SIZE_OF_PIXEL)
    gt_overlay = overlay_image(r1, r2_gt)
    cv2.imwrite(str(pair_dir / "gt.png"), gt_overlay)
    write_aligned_ply(pair_dir / "gt.ply", raw1, raw2, gt_transform, PLY_DOWNSAMPLE)

    # Per-method alignment
    meta_rows = []
    overlays = [("pointcloud1", cv2.cvtColor((r1 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)),
                ("pointcloud2", cv2.cvtColor((r2 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)),
                ("gt", gt_overlay)]
    for name, entry in results.items():
        if entry is None:
            meta_rows.append([idx1, idx2, name, "", "", "", "", "failed"])
            print(f"    {name:16s} FAILED")
            continue
        result, rot_err, trans_err = entry
        r2_al = render_points(align_points(raw2[:, :2], result.transform), raw2[:, 2], N, SIZE_OF_PIXEL)
        ov = overlay_image(r1, r2_al)
        cv2.imwrite(str(pair_dir / f"{name}.png"), ov)
        write_aligned_ply(pair_dir / f"{name}.ply", raw1, raw2, result.transform, PLY_DOWNSAMPLE)
        overlays.append((name, ov))
        meta_rows.append([
            idx1, idx2, name,
            f"{rot_err:.4f}", f"{trans_err:.4f}",
            f"{result.confidence:.6f}", f"{result.computation_time * 1000:.1f}", "",
        ])
        print(f"    {name:16s} rot_err={rot_err:7.3f} deg  trans_err={trans_err:7.3f} m  "
              f"conf={result.confidence:.3f}  time={result.computation_time * 1000:7.1f} ms")

    # Meta CSV
    header = "frame1,frame2,method,rot_err_deg,trans_err_m,confidence,time_ms,error"
    with open(pair_dir / "registration_meta.csv", "w") as f:
        f.write(header + "\n")
        for row in meta_rows:
            f.write(",".join(str(v) for v in row) + "\n")

    # Summary montage
    montage = make_montage(overlays)
    cv2.imwrite(str(pair_dir / "summary.png"), montage)


def main():
    # Reload config from file (in case it was edited)
    get_config_from_file()

    print("Config:")
    print(f"  DATA_DIR: {DATA_DIR}")
    print(f"  SEQUENCE_NUMBER: {SEQUENCE_NUMBER}")
    print(f"  SEQUENCE_NAME: {SEQUENCE_NAME}")
    print(f"  N: {N}, RADIUS: {RADIUS} m, pixel_size: {SIZE_OF_PIXEL:.3f} m")
    print(f"  MATCHING_STEP: {MATCHING_STEP}, START_FRAME: {START_FRAME}, MAX_FRAMES: {MAX_FRAMES}")
    print(f"  METHODS_TO_RUN: {METHODS_TO_RUN}")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print()

    # Load sequence
    if SEQUENCE_NAME is not None:
        print(f"Loading sequence '{SEQUENCE_NAME}' from {DATA_DIR} (single sequence mode)...")
        seq = load_single_sequence(DATA_DIR, SEQUENCE_NAME)
        seq_name = SEQUENCE_NAME
    else:
        print(f"Loading sequence {SEQUENCE_NUMBER} from {DATA_DIR}...")
        seq = load_sequence(DATA_DIR, SEQUENCE_NUMBER)
        seq_name = f"sequence_{SEQUENCE_NUMBER}"
    print(f"Sequence has {seq.length} radar scans")
    print()

    # Build all methods
    print("Creating registration methods...")
    methods = build_methods()
    if not methods:
        print("No methods could be created - aborting.")
        return
    print(f"Running {len(methods)} methods: {list(methods.keys())}")
    print()

    # Determine number of frames
    length_of_radar_scans = seq.length
    if MAX_FRAMES is not None:
        length_of_radar_scans = min(length_of_radar_scans, MAX_FRAMES)
    print(f"Matching every {MATCHING_STEP}th image (from frame {START_FRAME}, up to {length_of_radar_scans})")
    print("=" * 80)
    print()

    idx = START_FRAME + MATCHING_STEP

    while idx < length_of_radar_scans:
        prev_idx = idx - MATCHING_STEP
        pair_dir = Path(OUTPUT_DIR) / str(prev_idx) / seq_name

        print(f"\n--- Pair: {prev_idx} -> {idx} ---")
        print(f"  Output: {pair_dir}")

        img1, img2, raw1, raw2, gt_transform, results = run_pair(seq, prev_idx, idx, methods)

        # Print GT info
        gt_yaw = np.arctan2(gt_transform[1, 0], gt_transform[0, 0])
        if gt_yaw < 0:
            gt_yaw += 2 * np.pi
        print(f"  GT Rot: {gt_yaw:.4f} rad ({np.degrees(gt_yaw):.2f} deg), "
              f"GT Tx: {gt_transform[0, 3]:.3f} m, GT Ty: {gt_transform[1, 3]:.3f} m")

        save_pair_output(pair_dir, prev_idx, idx, img1, img2, raw1, raw2, gt_transform, results)
        print(f"  -> Saved to {pair_dir}/")

        idx += MATCHING_STEP

    print("\nDone.")


if __name__ == "__main__":
    main()
