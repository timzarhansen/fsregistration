"""
NDT P2D Parameter Sweep Benchmark

Tests all combinations of configurable NDT parameters on a single Boreas
sequence (default: seq 0).  Each combination gets its own output subdirectory
with results.csv and config.json in the same format as boreasBenchmark.py.

Usage:
    python boreasBenchmarkNDTSweep.py
"""

import csv
import json
import os
import random
import sys
import time
import itertools
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_script_dir))))
_install_lib = os.path.join(_root_dir, 'install', 'fsregistration', 'lib', 'fsregistration')
if os.path.isdir(_install_lib):
    sys.path.insert(0, _install_lib)

sys.path.insert(0, os.path.dirname(_script_dir))

from boreasDatasetLoader import (
    BoreasSequence,
    load_sequence,
    load_single_sequence,
    get_affine_matrix,
    transform_diff,
)
from boreasRegistrationMethods import RegistrationFactory


# ============================================================================
# CONFIG — Edit these to define your parameter sweep
# ============================================================================
DATA_DIR = "/home/tim-external/dataFolder/radar_boreas"
SEQUENCE = 0
OUTPUT_DIR = "benchmark_sweep"
START_FRAME = 0
MAX_FRAMES = None
NUM_WORKERS = 2
SAVE_BLENDED = False
USE_RAW_POINTCLOUD = True
RAW_INTENSITY_THRESHOLD = 0.3

PARAM_GRID = {
    "N": [128],
    "radius": [140.0],
    "matching_step": [3],
    "ndt_voxel_size": [2.0, 5.0, 10.0],
    "ndt_step_size": [0.05, 0.1, 0.5],
    "ndt_max_iteration": [50],
    "ndt_transformation_epsilon": [0.01],
    "ndt_scale": [1.0],
    "ndt_threshold_pct": [5.0],
    "ndt_z_scale": [0.0, 0.1, 1.0],
    "round": [False],
}
# ============================================================================


PARAM_ABBREV = {
    "ndt_voxel_size": "vox",
    "ndt_step_size": "step",
    "ndt_max_iteration": "iter",
    "ndt_transformation_epsilon": "eps",
    "ndt_scale": "scl",
    "ndt_threshold_pct": "thr",
    "ndt_z_scale": "z",
    "round": "mask",
    "radius": "R",
    "matching_step": "md",
    "N": "N",
}


# ============================================================================
# Helpers
# ============================================================================

def _format_value(v):
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def build_output_name(combo: dict, varied_keys: List[str], N: int,
                       size_of_pixel: float) -> str:
    step = combo["matching_step"]
    s_part = f"_s{step}" if "matching_step" not in varied_keys else ""
    parts = [f"seq{SEQUENCE:02d}_ndt_N{N:03d}_p{int(size_of_pixel * 100):04d}{s_part}"]
    for k in varied_keys:
        abbrev = PARAM_ABBREV.get(k, k)
        val_str = _format_value(combo[k])
        parts.append(f"{abbrev}{val_str}")
    return "_".join(parts)


def apply_circular_mask(image: np.ndarray) -> np.ndarray:
    N = image.shape[0]
    cy = cx = N // 2
    r = N // 2
    Y, X = np.ogrid[:N, :N]
    mask = (X - cx)**2 + (Y - cy)**2 <= r**2
    return image * mask


# ============================================================================
# Single-combo runner
# ============================================================================

_worker_seq_cache = None


def run_single_combo(args: tuple) -> Tuple[str, dict]:
    global _worker_seq_cache
    data_dir, seq_name, combo, varied_keys, output_dir, \
        start_frame, max_frames, save_blended = args

    if _worker_seq_cache is None:
        _worker_seq_cache = load_single_sequence(data_dir, seq_name)
    seq = _worker_seq_cache

    N = combo["N"]
    radius = combo["radius"]
    matching_step = combo["matching_step"]
    size_of_pixel = (2.0 * radius) / N

    method_config = {
        "N": N,
        "size_of_pixel": size_of_pixel,
        "ndt_voxel_size": combo["ndt_voxel_size"],
        "ndt_step_size": combo["ndt_step_size"],
        "ndt_max_iteration": combo["ndt_max_iteration"],
        "ndt_transformation_epsilon": combo["ndt_transformation_epsilon"],
        "ndt_scale": combo["ndt_scale"],
        "ndt_threshold_pct": combo["ndt_threshold_pct"],
        "ndt_z_scale": combo["ndt_z_scale"],
        "initial_guess": np.eye(4).tolist(),
    }

    out_name = build_output_name(combo, varied_keys, N, size_of_pixel)
    save_dir = Path(output_dir) / out_name
    save_dir.mkdir(parents=True, exist_ok=True)

    blended_dir = None
    if save_blended:
        blended_dir = save_dir / "blended"
        blended_dir.mkdir(parents=True, exist_ok=True)

    total_frames = seq.length
    if max_frames is not None:
        end_frame = min(start_frame + max_frames, total_frames)
    else:
        end_frame = total_frames

    num_pairs = 0
    if end_frame > start_frame + matching_step:
        num_pairs = max(0, (end_frame - start_frame - 1) // matching_step)

    print(f"  [{out_name}] {num_pairs} pairs, N={N}, radius={radius}m, "
          f"pixel={size_of_pixel:.3f}m")

    method = RegistrationFactory.create("ndt_p2d", method_config)

    results = []
    failures = []
    total_time = 0.0

    for pair_idx in range(num_pairs):
        prev_idx = start_frame + pair_idx * matching_step
        curr_idx = start_frame + (pair_idx + 1) * matching_step

        try:
            img_prev = seq.get_cartesian_image(prev_idx, N, size_of_pixel)
            img_curr = seq.get_cartesian_image(curr_idx, N, size_of_pixel)

            if combo["round"]:
                img_prev = apply_circular_mask(img_prev)
                img_curr = apply_circular_mask(img_curr)

            gt_transform = seq.get_gt_transform(prev_idx, curr_idx)
            gt_affine = get_affine_matrix(gt_transform)

            raw_prev = seq.get_raw_point_cloud(prev_idx, RAW_INTENSITY_THRESHOLD)
            raw_curr = seq.get_raw_point_cloud(curr_idx, RAW_INTENSITY_THRESHOLD)

            t0 = time.time()
            result = method.register(img_prev, img_curr, pcd1=raw_prev, pcd2=raw_curr)
            elapsed = time.time() - t0
            total_time += elapsed

            est_affine = get_affine_matrix(result.transform)
            trans_error, rot_error = transform_diff(gt_affine, est_affine)
            trans_norm = np.linalg.norm(trans_error)

            trans_x_error = trans_error[0]
            trans_y_error = trans_error[1]

            est_yaw = np.degrees(np.arctan2(est_affine[1, 0], est_affine[0, 0]))
            est_tx = est_affine[0, 2]
            est_ty = est_affine[1, 2]

            gt_yaw = np.degrees(np.arctan2(gt_affine[1, 0], gt_affine[0, 0]))
            gt_tx = gt_affine[0, 2]
            gt_ty = gt_affine[1, 2]

            row = {
                "pair_idx": pair_idx,
                "prev_frame": prev_idx,
                "curr_frame": curr_idx,
                "rot_error_deg": rot_error,
                "trans_error_m": trans_norm,
                "trans_x_error_m": trans_x_error,
                "trans_y_error_m": trans_y_error,
                "est_rot_deg": est_yaw,
                "est_tx_m": est_tx,
                "est_ty_m": est_ty,
                "est_confidence": result.confidence,
                "gt_rot_deg": gt_yaw,
                "gt_tx_m": gt_tx,
                "gt_ty_m": gt_ty,
                "computation_time_ms": elapsed * 1000,
            }
            results.append(row)

            if save_blended and blended_dir is not None:
                warp_affine = get_affine_matrix(result.transform,
                                                pixel_size=size_of_pixel, img_size=N)
                warped = cv2.warpPerspective(
                    img_curr, warp_affine,
                    (img_prev.shape[1], img_prev.shape[0])
                )
                blended = cv2.addWeighted(
                    (img_prev * 255).astype(np.uint8), 0.5,
                    (warped * 255).astype(np.uint8), 0.5, 0
                )
                cv2.imwrite(str(blended_dir / f"blended_{curr_idx:04d}.png"), blended)

            if hasattr(seq.sequence, 'radar_frames'):
                seq.sequence.radar_frames[prev_idx].unload_data()
                seq.sequence.radar_frames[curr_idx].unload_data()

        except Exception as e:
            failures.append({
                "pair_idx": pair_idx,
                "prev_frame": prev_idx,
                "curr_frame": curr_idx,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    # --- Summary ---
    summary = {
        "method": "ndt_p2d",
        "sequence": SEQUENCE,
        "N": N,
        "radius": radius,
        "pixel_size": size_of_pixel,
        "matching_step": matching_step,
        "start_frame": start_frame,
        "max_frames": max_frames,
        "total_frames": total_frames,
        "num_pairs_processed": len(results),
        "num_pairs_failed": len(failures),
        "total_run_time_seconds": total_time,
    }
    for k, v in combo.items():
        summary[f"param_{k}"] = v

    if results:
        rot_errors = [r["rot_error_deg"] for r in results]
        trans_errors = [r["trans_error_m"] for r in results]
        times = [r["computation_time_ms"] for r in results]
        confidences = [r["est_confidence"] for r in results]

        summary.update({
            "avg_rot_error_deg": float(np.mean(np.abs(rot_errors))),
            "std_rot_error_deg": float(np.std(np.abs(rot_errors))),
            "min_rot_error_deg": float(np.min(np.abs(rot_errors))),
            "max_rot_error_deg": float(np.max(np.abs(rot_errors))),
            "avg_trans_error_m": float(np.mean(trans_errors)),
            "std_trans_error_m": float(np.std(trans_errors)),
            "min_trans_error_m": float(np.min(trans_errors)),
            "max_trans_error_m": float(np.max(trans_errors)),
            "avg_confidence": float(np.mean(confidences)),
            "avg_time_ms": float(np.mean(times)),
            "median_time_ms": float(np.median(times)),
        })

    # Save config.json
    config_json = {
        "method": "ndt_p2d",
        "config": method_config,
        "benchmark_params": {
            "matching_step": matching_step,
            "start_frame": start_frame,
            "max_frames": max_frames,
            "save_blended": save_blended,
            "round": combo["round"],
            "use_raw_pointcloud": USE_RAW_POINTCLOUD,
            "raw_intensity_threshold": RAW_INTENSITY_THRESHOLD,
        },
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config_json, f, indent=2)

    # Save results.csv
    results_csv = save_dir / "results.csv"
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        for key, value in summary.items():
            writer.writerow([f"# {key}: {value}"])

        columns = [
            "pair_idx", "prev_frame", "curr_frame",
            "rot_error_deg", "trans_error_m", "trans_x_error_m", "trans_y_error_m",
            "est_rot_deg", "est_tx_m", "est_ty_m", "est_confidence",
            "gt_rot_deg", "gt_tx_m", "gt_ty_m",
            "computation_time_ms",
        ]
        writer.writerow(columns)
        for row in results:
            writer.writerow([row[col] for col in columns])

    if failures:
        with open(save_dir / "failures.log", "w") as f:
            for fail in failures:
                f.write(f"Pair {fail['pair_idx']} (frames {fail['prev_frame']} -> {fail['curr_frame']}): {fail['error']}\n")
                f.write(f"{fail['traceback']}\n\n")

    return str(results_csv), summary


# ============================================================================
# Main
# ============================================================================

def main():
    np.set_printoptions(precision=5, suppress=True)

    print("=" * 75)
    print("NDT P2D Parameter Sweep Benchmark")
    print("=" * 75)
    print(f"  Data:     {DATA_DIR}")
    print(f"  Sequence: {SEQUENCE}")
    print(f"  Output:   {OUTPUT_DIR}")
    print(f"  Start frame: {START_FRAME}")
    print(f"  Workers:  {NUM_WORKERS}")
    print()

    print("Loading sequence (to get sequence name)...")
    seq = load_sequence(DATA_DIR, SEQUENCE)
    seq_name = str(seq.sequence.ID)
    print(f"  Sequence: {seq_name}, {seq.length} radar scans")
    del seq
    print()

    keys = list(PARAM_GRID.keys())
    value_lists = list(PARAM_GRID.values())
    varied_keys = [k for k, v in PARAM_GRID.items() if len(v) > 1]

    combos = []
    for values in itertools.product(*value_lists):
        combos.append(dict(zip(keys, values)))
    random.shuffle(combos)

    print(f"  Grid keys: {keys}")
    print(f"  Varied keys (in dir name): {varied_keys}")
    print(f"  Total combinations: {len(combos)}")
    print()

    if not combos:
        print("ERROR: No parameter combinations to run.")
        sys.exit(1)

    for i, combo in enumerate(combos):
        out_name = build_output_name(
            combo, varied_keys, combo["N"],
            (2.0 * combo["radius"]) / combo["N"],
        )
        print(f"  [{i+1:3d}/{len(combos)}] {out_name}")
    print()

    worker_args = [
        (DATA_DIR, seq_name, combo, varied_keys, OUTPUT_DIR,
         START_FRAME, MAX_FRAMES, SAVE_BLENDED)
        for combo in combos
    ]

    t_start = time.time()
    if NUM_WORKERS <= 1:
        all_results = [run_single_combo(a) for a in worker_args]
    else:
        with Pool(processes=NUM_WORKERS) as pool:
            all_results = list(pool.imap_unordered(run_single_combo, worker_args))
    elapsed = time.time() - t_start

    print()
    print("=" * 75)
    print(f"All combinations done in {elapsed:.1f}s")
    print(f"  Successful: {len(all_results)}")
    print()

    print(f"{'Run':<5} {'AvgRot°':>8} {'AvgTrans(m)':>12} {'AvgTime(ms)':>11} {'Pairs':>6}  Name")
    print("-" * 85)
    for i, (csv_path, s) in enumerate(all_results):
        name = Path(csv_path).parent.name
        avg_rot = s.get("avg_rot_error_deg", float('nan'))
        avg_trans = s.get("avg_trans_error_m", float('nan'))
        avg_time = s.get("avg_time_ms", float('nan'))
        pairs = s.get("num_pairs_processed", 0)
        print(f"{i+1:<5} {avg_rot:>8.3f} {avg_trans:>12.4f} {avg_time:>11.1f} {pairs:>6}  {name}")

    print()
    print(f"Results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
