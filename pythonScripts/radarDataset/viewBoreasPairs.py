################################################################################
#
# Boreas Pair Viewer - Automated pair-by-pair registration with image saving
#
# Usage:
#     python viewBoreasPairs.py
#
# Saves display images to displayImages/ folder:
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
from typing import Tuple

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from boreasDatasetLoader import (
    BoreasSequence,
    load_sequence,
    load_single_sequence,
    get_affine_matrix,
    transform_diff,
)
from boreasRegistrationMethods import FS2DRegistration


# ============================================================================
# CONFIGURATION - Edit these to test different settings
# ============================================================================
DATA_DIR = "/home/tim-external/dataFolder/radar_boreas"
SEQUENCE_NUMBER = 0
SEQUENCE_NAME = "boreas-2020-11-26-13-58" # Sequence name string, e.g. 'boreas-2020-11-26-13-58'
N = 128                          # Image grid size (N x N)
SIZE_OF_PIXEL = 2.0              # Meters per pixel
DEBUG_MODE = True
MATCHING_STEP = 1                # Match every Nth frame
MAX_FRAMES = None                # None = full sequence, or cap it
OUTPUT_DIR = "viewBoreasOutput"  # Blended images saved here
USE_DIRECT = True               # Use direct registration (1-angle) vs SO3 (multiple angles)
LEVEL_POTENTIAL_ROTATION = 0.1  # Persistence threshold for rotation peak filtering
# ============================================================================


def get_config_from_file():
    """Reload config from this file in case it was edited."""
    global DATA_DIR, SEQUENCE_NUMBER, SEQUENCE_NAME, N, SIZE_OF_PIXEL
    global MATCHING_STEP, MAX_FRAMES, OUTPUT_DIR, USE_DIRECT, LEVEL_POTENTIAL_ROTATION

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
        return default

    DATA_DIR = extract_var("DATA_DIR", DATA_DIR)
    SEQUENCE_NUMBER = extract_var("SEQUENCE_NUMBER", SEQUENCE_NUMBER)
    SEQUENCE_NAME = extract_var("SEQUENCE_NAME", SEQUENCE_NAME)
    N = extract_var("N", N)
    SIZE_OF_PIXEL = extract_var("SIZE_OF_PIXEL", SIZE_OF_PIXEL)
    MATCHING_STEP = extract_var("MATCHING_STEP", MATCHING_STEP)
    MAX_FRAMES = extract_var("MAX_FRAMES", MAX_FRAMES)
    OUTPUT_DIR = extract_var("OUTPUT_DIR", OUTPUT_DIR)
    USE_DIRECT = extract_var("USE_DIRECT", USE_DIRECT)
    LEVEL_POTENTIAL_ROTATION = extract_var("LEVEL_POTENTIAL_ROTATION", LEVEL_POTENTIAL_ROTATION)


DISPLAY_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "displayImages"


def run_pair(
    seq: BoreasSequence,
    idx1: int,
    idx2: int,
    method: FS2DRegistration,
    save_dir: Path,
) -> Tuple:
    """Register one pair and return (img1, img2, blended, result, gt_error)."""
    img1 = seq.get_cartesian_image(idx1, N, SIZE_OF_PIXEL)
    img2 = seq.get_cartesian_image(idx2, N, SIZE_OF_PIXEL)

    gt_transform = seq.get_gt_transform(idx1, idx2)
    gt_affine = get_affine_matrix(gt_transform)

    t0 = time.time()

    # Run registration with debug=True
    image_1 = img1.astype(np.float64).reshape(-1)
    image_2 = img2.astype(np.float64).reshape(-1)

    list_peaks = method.wrapper.register_all_solutions(
        image_1, image_2,
        cellSize=SIZE_OF_PIXEL,
        useGauss=method.use_gauss,
        debug=DEBUG_MODE,
        potentialNecessaryForPeak=method.potential_for_necessary_peak,
        multipleRadii=method.multiple_radii,
        useClahe=method.use_clahe,
        useHamming=method.use_hamming,
        useDirect=method.use_direct
    )

    # Find highest peak
    highest_peak = 0.0
    index_highest = 0
    for i, peak in enumerate(list_peaks):
        if peak.potentialTranslations[0].peakHeight > highest_peak:
            highest_peak = peak.potentialTranslations[0].peakHeight
            index_highest = i

    peak = list_peaks[index_highest]

    transform = np.eye(4)
    yaw = peak.potentialRotation.angle
    from scipy.spatial.transform import Rotation as R
    transform[:3, :3] = R.from_euler("z", yaw).as_matrix()
    tx, ty = peak.potentialTranslations[0].translationSI
    transform[:3, 3] = [tx, ty, 0.0]

    elapsed = time.time() - t0

    result = type('obj', (object,), {
        'transform': transform,
        'confidence': highest_peak,
        'method_name': 'fs2d',
        'computation_time': elapsed,
        'metadata': {
            'rotation_angle': yaw,
            'translation': (tx, ty),
            'peak_height': highest_peak,
            'num_solutions': len(list_peaks)
        }
    })()

    # Compute GT error
    est_affine = get_affine_matrix(transform)
    gt_error = transform_diff(gt_affine, est_affine)

    # Create blended image
    fs2d_affine = est_affine
    warped = cv2.warpPerspective(img2, fs2d_affine, (img1.shape[1], img1.shape[0]))
    blended = cv2.addWeighted((img1 * 255).astype(np.uint8), 0.5, (warped * 255).astype(np.uint8), 0.5, 0)

    return img1, img2, blended, result, gt_error


def main():
    # Reload config from file (in case it was edited)
    get_config_from_file()

    # Create display directory
    DISPLAY_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Config:")
    print(f"  DATA_DIR: {DATA_DIR}")
    print(f"  SEQUENCE_NUMBER: {SEQUENCE_NUMBER}")
    print(f"  SEQUENCE_NAME: {SEQUENCE_NAME}")
    print(f"  DEBUG_MODE: {DEBUG_MODE}")
    print(f"  N: {N}, SIZE_OF_PIXEL: {SIZE_OF_PIXEL}")
    print(f"  MATCHING_STEP: {MATCHING_STEP}, MAX_FRAMES: {MAX_FRAMES}")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"  DISPLAY_DIR: {DISPLAY_DIR}")
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

    # Setup method
    method_config = {
        "N": N,
        "use_clahe": True,
        "use_hamming": True,
        "potential_for_necessary_peak": 0.01,
        "multiple_radii": True,
        "use_gauss": False,
        "size_of_pixel": SIZE_OF_PIXEL,
        "use_direct": USE_DIRECT,
        "level_potential_rotation": LEVEL_POTENTIAL_ROTATION,
    }
    method = FS2DRegistration(method_config)
    print(f"Method: FS2D (N={N}, clahe={method.use_clahe}, hamming={method.use_hamming}, use_direct={method.use_direct}, level_potential_rotation={method.level_potential_rotation})")
    print()

    # Setup output directory
    method_key = "fs2d"
    save_dir = Path(
        f"{OUTPUT_DIR}/{SEQUENCE_NUMBER:02d}_{N:03d}_{int(SIZE_OF_PIXEL*100)}_{MATCHING_STEP}/{method_key}/"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {save_dir}")
    print()

    # Determine number of frames
    length_of_radar_scans = seq.length
    if MAX_FRAMES is not None:
        length_of_radar_scans = min(length_of_radar_scans, MAX_FRAMES)
    print(f"Matching every {MATCHING_STEP}th image (up to frame {length_of_radar_scans})")
    print(f"Display images saved to: {DISPLAY_DIR}/")
    print("Image files: image1.png, image2.png, blended.png (overwritten each pair)")
    print("=" * 80)
    print()

    idx = MATCHING_STEP

    while idx < length_of_radar_scans:
        prev_idx = idx - MATCHING_STEP

        print(f"\n--- Pair: {prev_idx} -> {idx} ---")

        # Run registration
        img1, img2, blended, result, gt_error = run_pair(seq, prev_idx, idx, method, save_dir)

        # Save display images (overwrite each time)
        cv2.imwrite(str(DISPLAY_DIR / "image1.png"), cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        cv2.imwrite(str(DISPLAY_DIR / "image2.png"), cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        cv2.imwrite(str(DISPLAY_DIR / "blended.png"), blended)

        # Print results
        gt_trans, gt_rot = gt_error
        gt_trans_norm = np.linalg.norm(gt_trans)
        print(f"  Rot: {result.metadata['rotation_angle'] * 180 / np.pi:.4f} deg")
        print(f"  Tx: {result.metadata['translation'][0]:.4f} m, Ty: {result.metadata['translation'][1]:.4f} m")
        print(f"  Confidence: {result.confidence:.4f}")
        print(f"  Time: {result.computation_time * 1000:.1f} ms")
        print(f"  GT RotErr: {abs(gt_rot):.4f} deg, GT TransErr: {gt_trans_norm:.4f} m")
        print(f"  -> Saved to {DISPLAY_DIR}/ (image1.png, image2.png, blended.png)")

        idx += MATCHING_STEP

    print("\nDone.")


if __name__ == "__main__":
    main()
