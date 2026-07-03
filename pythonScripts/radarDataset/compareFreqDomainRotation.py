#!/usr/bin/env python3
"""
Compare frequency-domain rotation vs spatial rotation for 2D registration.

Runs a single Boreas radar pair through both methods and prints a comparison
table of rotation angles, translations, peak heights, and timing.

Usage:
    python compareFreqDomainRotation.py

Edit CONFIG section below to change settings.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import cv2

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_script_dir))))
_install_lib = os.path.join(_root_dir, 'install', 'fsregistration', 'lib', 'fsregistration')
if os.path.isdir(_install_lib):
    sys.path.insert(0, _install_lib)
sys.path.insert(0, _script_dir)
from boreasDatasetLoader import load_sequence, get_affine_matrix, transform_diff

# ============================================================================
# CONFIG - Edit these
# ============================================================================
DATA_DIR = "/home/tim-external/dataFolder/radar_boreas"
SEQUENCE_NUMBER = 0
SEQUENCE_NAME = "boreas-2020-11-26-13-58"
N = 256
SIZE_OF_PIXEL = 0.5
FRAME_A = 45
FRAME_B = 50
LEVEL_POTENTIAL_ROTATION = 0.001
POTENTIAL_NECCESSARY_FOR_PEAK = 0.1
DEBUG_MODE = True  # writes CSVs for the notebook when True
OUTPUT_DIR = "freqRotationCompare"
# ============================================================================

PLOT_DATA_DIR = "/home/tim-external/ros_ws/src/fsregistration/plotting_results/2d/data"

def normalize_image(image):
    return image.astype(np.float64).reshape(-1)


def run_registration(wrapper, img1, img2, cell_size, debug=False):
    """Run register_all_solutions and return (result, elapsed)."""
    image_1 = normalize_image(img1)
    image_2 = normalize_image(img2)

    t0 = time.time()
    list_peaks = wrapper.register_all_solutions(
        image_1, image_2,
        cellSize=cell_size,
        useGauss=False,
        debug=debug,
        potentialNecessaryForPeak=POTENTIAL_NECCESSARY_FOR_PEAK,
        multipleRadii=True,
        useClahe=True,
        useHamming=True,
        useDirect=True,
        levelPotentialRotation=LEVEL_POTENTIAL_ROTATION,
    )
    elapsed = time.time() - t0

    rows = []
    for peak_idx, peak in enumerate(list_peaks):
        yaw = peak.potentialRotation.angle
        for t_idx, t in enumerate(peak.potentialTranslations):
            rows.append({
                'angle_idx': peak_idx,
                'angle': yaw,
                'angle_deg': np.degrees(yaw),
                'tx': t.translationSI[0],
                'ty': t.translationSI[1],
                'peak_height': t.peakHeight,
                'persistence': t.persistenceValue,
            })
    return rows, elapsed


def main():
    print("=" * 70)
    print("Freq-Domain Rotation vs Spatial Rotation Comparison")
    print("=" * 70)
    print(f"  Sequence: {SEQUENCE_NAME} (#{SEQUENCE_NUMBER})")
    print(f"  Frames:   {FRAME_A} → {FRAME_B}")
    print(f"  N:        {N}")
    print(f"  Pixel:    {SIZE_OF_PIXEL} m")
    print()

    # Load sequence
    seq = load_sequence(DATA_DIR, SEQUENCE_NUMBER)
    img1 = seq.get_cartesian_image(FRAME_A, N, SIZE_OF_PIXEL)
    img2 = seq.get_cartesian_image(FRAME_B, N, SIZE_OF_PIXEL)

    print(f"  Image shapes: {img1.shape}, {img2.shape}")
    print(f"  Image range:  [{img1.min():.4f}, {img1.max():.4f}] / [{img2.min():.4f}, {img2.max():.4f}]")
    print()

    # GT transform
    gt_transform = seq.get_gt_transform(FRAME_A, FRAME_B)
    gt_affine = get_affine_matrix(gt_transform)
    gt_x = gt_affine[0, 2]
    gt_y = gt_affine[1, 2]
    gt_yaw = np.degrees(np.arctan2(gt_affine[1, 0], gt_affine[0, 0]))
    print(f"  GT:          tx={gt_x:.3f}, ty={gt_y:.3f}, yaw={gt_yaw:.2f}°")
    print()

    from pybind_registration_2d import SoftRegistrationWrapper2D
    wrapper = SoftRegistrationWrapper2D(N)

    # Run registration
    print("Running registration (spatial rotation with pre-computed scan2 FFT)...")
    if DEBUG_MODE:
        os.makedirs(PLOT_DATA_DIR, exist_ok=True)

    rows, elapsed = run_registration(wrapper, img1, img2, SIZE_OF_PIXEL, debug=DEBUG_MODE)

    # --- Print results ---
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Computation time: {elapsed*1000:.2f} ms")
    print(f"  Rotation angles:  {len(set(r['angle_idx'] for r in rows))}")
    print(f"  Translation peaks: {len(rows)}")
    print()

    # Per-angle table
    print(f"  {'Angle index':<12} {'Angle (deg)':<14} {'#translations':<14}")
    print(f"  {'-'*12} {'-'*14} {'-'*14}")
    angles_sorted = sorted(set(r['angle_idx'] for r in rows), key=lambda idx: rows[0]['angle'])
    for aidx in sorted(set(r['angle_idx'] for r in rows)):
        arows = [r for r in rows if r['angle_idx'] == aidx]
        print(f"  {aidx:<12} {np.degrees(arows[0]['angle']):<14.2f} {len(arows):<14}")

    # --- Top solution ---
    print()
    print("--- Top translation peak (best confidence) ---")
    best = sorted(rows, key=lambda r: -r['peak_height'])[0]
    print(f"  Angle (deg):      {np.degrees(best['angle']):.2f}")
    print(f"  Translation X:    {best['tx']:.4f}")
    print(f"  Translation Y:    {best['ty']:.4f}")
    print(f"  Peak height:      {best['peak_height']:.6f}")
    print(f"  GT X:             {gt_x:.4f}")
    print(f"  GT Y:             {gt_y:.4f}")
    print(f"  GT Angle (deg):   {gt_yaw:.2f}")

    # --- Visual ---
    out_dir = Path(_script_dir) / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    best_row = sorted(rows, key=lambda r: -r['peak_height'])[0]
    transform = np.eye(4)
    transform[:3, :3] = R.from_euler("z", best_row['angle']).as_matrix()
    transform[:3, 3] = [best_row['tx'], -best_row['ty'], 0.0]
    affine = get_affine_matrix(transform)
    warped = cv2.warpPerspective(img2, affine, (img1.shape[1], img1.shape[0]))
    blended = cv2.addWeighted(img1, 0.5, warped, 0.5, 0)
    blended_path = out_dir / "blended.png"
    cv2.imwrite(str(blended_path), blended * 255)
    print(f"\n  Blended image saved to: {blended_path}")

    if DEBUG_MODE:
        print(f"  Debug CSVs written to: {PLOT_DATA_DIR}")
        print(f"  (Load plot_2d_registration.ipynb to visualize)")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
