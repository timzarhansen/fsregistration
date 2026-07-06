################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Dan Barnes (dbarnes@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################

"""
Boreas Radar Benchmarking Script

Single-method, single-sequence benchmarking for the Boreas radar dataset.
Outputs a structured CSV with metadata header + per-pair results, suitable
for cross-method comparison and analysis in Python/Pandas/R.

Usage:
    python boreasBenchmark.py --method fs2d --sequence 0 --N 128 --size_of_pixel 0.5 \
        --matching_step 5 --output-dir benchmark_results <data_dir>

    python boreasBenchmark.py --method fs2d --sequence 0 --N 256 --size_of_pixel 0.25 \
        --matching_step 3 --start_frame 0 --max_frames 100 \
        --method-config "fs2d.use_direct=1 fs2d.level_potential_rotation=0.01" \
        --output-dir benchmark_results <data_dir>
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add paths
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_script_dir))))
_install_lib = os.path.join(_root_dir, 'install', 'fsregistration', 'lib', 'fsregistration')
if os.path.isdir(_install_lib):
    sys.path.insert(0, _install_lib)

from boreasDatasetLoader import (
    BoreasSequence,
    load_sequence,
    load_single_sequence,
    get_affine_matrix,
    transform_diff,
)
from boreasRegistrationMethods import RegistrationFactory, RegistrationResult


# ============================================================================
# Default configuration
# ============================================================================

DEFAULT_CONFIG = {
    "N": 128,
    "size_of_pixel": 0.5,
    "matching_step": 5,
    "start_frame": 0,
    "max_frames": None,
    "output_dir": "benchmark_results",
    "use_clahe": True,
    "use_hamming": True,
    "use_direct": True,
    "use_gauss": False,
    "multiple_radii": True,
    "potential_for_necessary_peak": 0.01,
    "level_potential_rotation": 0.001,
    "save_blended": False,
}


# ============================================================================
# Benchmark runner
# ============================================================================

def run_benchmark(
    seq: BoreasSequence,
    method_name: str,
    method_config: dict,
    sequence_number: int,
    matching_step: int,
    start_frame: int,
    max_frames: Optional[int],
    save_blended: bool,
    output_dir: str,
) -> Tuple[Path, dict]:
    """Run benchmark on a single sequence with a single method.

    Args:
        seq: BoreasSequence instance.
        method_name: Name of the registration method.
        method_config: Method configuration dict.
        sequence_number: Sequence number for output naming.
        matching_step: Match every Nth frame.
        start_frame: First frame index.
        max_frames: Cap sequence length (None = full).
        save_blended: Whether to save blended images.
        output_dir: Base output directory.

    Returns:
        Tuple of (results_csv_path, summary_dict).
    """
    N = method_config["N"]
    size_of_pixel = method_config["size_of_pixel"]

    # Determine number of frames
    total_frames = seq.length
    if max_frames is not None:
        end_frame = min(start_frame + max_frames, total_frames)
    else:
        end_frame = total_frames

    num_pairs = 0
    if end_frame > start_frame + matching_step:
        num_pairs = (end_frame - start_frame - matching_step) // matching_step + 1

    # Setup output directory
    px_int = int(size_of_pixel * 100)
    output_subdir = f"seq{0:02d}_{method_name}_N{N:03d}_p{px_int}_s{matching_step}"
    # Try to get sequence name for better naming
    try:
        seq_name = str(seq.sequence)
        output_subdir = f"seq{0:02d}_{method_name}_N{N:03d}_p{px_int}_s{matching_step}"
    except Exception:
        pass

    save_dir = Path(output_dir) / output_subdir
    save_dir.mkdir(parents=True, exist_ok=True)

    # Blended images directory
    blended_dir = None
    if save_blended:
        blended_dir = save_dir / "blended"
        blended_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving results to: {save_dir}")
    print(f"Sequence has {total_frames} radar scans (processing frames {start_frame} to {end_frame})")
    print(f"Matching every {matching_step}th frame -> {num_pairs} pairs")
    print(f"Method: {method_name}")
    print()

    # Initialize method
    method = RegistrationFactory.create(method_name, method_config)
    print(f"Method config: {method.config}")
    print()

    # Results storage
    results = []
    failures = []
    total_time = 0.0

    # Print progress header
    print(f"{'Pair':>6} {'Prev':>6} {'Curr':>6} {'RotErr°':>8} {'TransErr(m)':>12} "
          f"{'Conf':>6} {'Time(ms)':>9} {'Status':>8}")
    print("-" * 75)

    for pair_idx in range(num_pairs):
        prev_idx = start_frame + pair_idx * matching_step
        curr_idx = start_frame + (pair_idx + 1) * matching_step

        try:
            # Load images
            img_prev = seq.get_cartesian_image(prev_idx, N, size_of_pixel)
            img_curr = seq.get_cartesian_image(curr_idx, N, size_of_pixel)

            # Get GT transformation
            gt_transform = seq.get_gt_transform(prev_idx, curr_idx)
            gt_affine = get_affine_matrix(gt_transform)

            # Run registration
            t0 = time.time()
            result = method.register(img_prev, img_curr)
            elapsed = time.time() - t0
            total_time += elapsed

            # Compute estimated affine and errors
            est_affine = get_affine_matrix(result.transform)
            trans_error, rot_error = transform_diff(gt_affine, est_affine)
            trans_norm = np.linalg.norm(trans_error)

            # Extract error components
            trans_x_error = trans_error[0]
            trans_y_error = trans_error[1]

            # Extract estimated pose components
            est_yaw = np.degrees(np.arctan2(est_affine[1, 0], est_affine[0, 0]))
            est_tx = est_affine[0, 2]
            est_ty = est_affine[1, 2]

            # Extract GT pose components
            gt_yaw = np.degrees(np.arctan2(gt_affine[1, 0], gt_affine[0, 0]))
            gt_tx = gt_affine[0, 2]
            gt_ty = gt_affine[1, 2]

            # Store result
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

            # Save blended image if requested
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

            print(f"{pair_idx:6d} {prev_idx:6d} {curr_idx:6d} {rot_error:8.3f} {trans_norm:12.4f} "
                  f"{result.confidence:6.3f} {elapsed * 1000:9.1f} OK")

        except Exception as e:
            # Log failure but continue
            failures.append({
                "pair_idx": pair_idx,
                "prev_frame": prev_idx,
                "curr_frame": curr_idx,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            print(f"{pair_idx:6d} {prev_idx:6d} {curr_idx:6d} {'FAIL':>8} {'ERROR':>12} {'':>6} {'':>9} ERR")
            print(f"  -> {e}")

    print()
    print("-" * 75)
    print(f"Done. {len(results)} pairs processed, {len(failures)} failures.")
    print()

    # Compute summary statistics
    summary = {
        "method": method_name,
        "sequence": sequence_number,
        "sequence_name": str(seq.sequence) if hasattr(seq, 'sequence') else "unknown",
        "N": N,
        "size_of_pixel": size_of_pixel,
        "matching_step": matching_step,
        "start_frame": start_frame,
        "max_frames": max_frames,
        "total_frames": total_frames,
        "num_pairs_processed": len(results),
        "num_pairs_failed": len(failures),
        "total_run_time_seconds": total_time,
    }

    if results:
        rot_errors = [r["rot_error_deg"] for r in results]
        trans_errors = [r["trans_error_m"] for r in results]
        times = [r["computation_time_ms"] for r in results]
        confidences = [r["est_confidence"] for r in results]

        summary.update({
            "avg_rot_error_deg": np.mean(np.abs(rot_errors)),
            "std_rot_error_deg": np.std(np.abs(rot_errors)),
            "min_rot_error_deg": np.min(np.abs(rot_errors)),
            "max_rot_error_deg": np.max(np.abs(rot_errors)),
            "avg_trans_error_m": np.mean(trans_errors),
            "std_trans_error_m": np.std(trans_errors),
            "min_trans_error_m": np.min(trans_errors),
            "max_trans_error_m": np.max(trans_errors),
            "avg_confidence": np.mean(confidences),
            "avg_time_ms": np.mean(times),
            "median_time_ms": np.median(times),
        })

    # Save config.json
    config_json = {
        "method": method_name,
        "config": method_config,
        "benchmark_params": {
            "matching_step": matching_step,
            "start_frame": start_frame,
            "max_frames": max_frames,
            "save_blended": save_blended,
        },
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config_json, f, indent=2)

    # Save results.csv with metadata header
    results_csv = save_dir / "results.csv"
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)

        # Metadata header rows
        for key, value in summary.items():
            writer.writerow([f"# {key}: {value}"])

        # Column header
        columns = [
            "pair_idx", "prev_frame", "curr_frame",
            "rot_error_deg", "trans_error_m", "trans_x_error_m", "trans_y_error_m",
            "est_rot_deg", "est_tx_m", "est_ty_m", "est_confidence",
            "gt_rot_deg", "gt_tx_m", "gt_ty_m",
            "computation_time_ms",
        ]
        writer.writerow(columns)

        # Data rows
        for row in results:
            writer.writerow([row[col] for col in columns])

    # Save execution log if there were failures
    if failures:
        with open(save_dir / "failures.log", "w") as f:
            for fail in failures:
                f.write(f"Pair {fail['pair_idx']} (frames {fail['prev_frame']} -> {fail['curr_frame']}): {fail['error']}\n")
                f.write(f"{fail['traceback']}\n\n")

    # Print summary
    print(f"[{method_name}] Summary:")
    print(f"  Pairs processed: {summary['num_pairs_processed']}")
    print(f"  Pairs failed: {summary['num_pairs_failed']}")
    print(f"  Total time: {total_time:.1f}s ({total_time / max(len(results), 1):.3f}s per pair)")
    if results:
        print(f"  Avg rotation error: {summary['avg_rot_error_deg']:.3f} deg")
        print(f"  Std rotation error: {summary['std_rot_error_deg']:.3f} deg")
        print(f"  Avg translation error: {summary['avg_trans_error_m']:.4f} m")
        print(f"  Std translation error: {summary['std_trans_error_m']:.4f} m")
        print(f"  Avg confidence: {summary['avg_confidence']:.4f}")
        print(f"  Avg computation time: {summary['avg_time_ms']:.1f} ms")
    print(f"\nResults saved to: {save_dir}")

    return results_csv, summary


# ============================================================================
# CLI helpers
# ============================================================================

def _parse_value(v):
    """Parse a string value into the appropriate Python type."""
    if v.lower() in ("true", "1"):
        return True
    if v.lower() in ("false", "0"):
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _parse_method_config(specs: List[str]) -> dict:
    """Parse method config specs in format 'method_name.key=value'.

    Args:
        specs: List of config strings, e.g. ["fs2d.N=128", "fs2d.use_clahe=1"]

    Returns:
        Dict of method_name -> {key: value, ...}
    """
    method_configs = {}
    for spec in specs:
        parts = spec.split()
        for part in parts:
            if "." not in part:
                continue
            method_name, _, param = part.partition(".")
            k, _, v = param.partition("=")
            v = _parse_value(v)
            if method_name not in method_configs:
                method_configs[method_name] = {}
            method_configs[method_name][k] = v
    return method_configs


# ============================================================================
# Main
# ============================================================================

def main():
    np.set_printoptions(precision=5, suppress=True)

    parser = argparse.ArgumentParser(
        description="Boreas Radar Benchmarking - Single method, single sequence"
    )
    parser.add_argument("--method", type=str, required=True,
                        help="Registration method to benchmark. "
                             f"Available: {RegistrationFactory.list_methods()}")
    parser.add_argument("--sequence", type=int, required=True,
                        help="Sequence number from Boreas dataset.")
    parser.add_argument("--sequence-name", type=str, default=None,
                        help="Sequence name string (e.g. 'boreas-2020-11-26-13-58'). "
                             "When provided, loads only this sequence without scanning all sequences.")
    parser.add_argument("--N", type=int, default=DEFAULT_CONFIG["N"],
                        help=f"Image grid size (N x N). Default: {DEFAULT_CONFIG['N']}")
    parser.add_argument("--size_of_pixel", type=float, default=DEFAULT_CONFIG["size_of_pixel"],
                        help=f"Size of a pixel in meters. Default: {DEFAULT_CONFIG['size_of_pixel']}")
    parser.add_argument("--matching_step", type=int, default=DEFAULT_CONFIG["matching_step"],
                        help=f"Match every Nth frame. Default: {DEFAULT_CONFIG['matching_step']}")
    parser.add_argument("--start_frame", type=int, default=DEFAULT_CONFIG["start_frame"],
                        help=f"First frame index. Default: {DEFAULT_CONFIG['start_frame']}")
    parser.add_argument("--max_frames", type=int, default=DEFAULT_CONFIG["max_frames"],
                        help=f"Cap sequence length (None = full). Default: {DEFAULT_CONFIG['max_frames']}")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_CONFIG["output_dir"],
                        help=f"Output directory. Default: {DEFAULT_CONFIG['output_dir']}")
    parser.add_argument("--method-config", action="append", default=[],
                        help="Method config in format 'method_name.key=value'. "
                             "e.g., 'fs2d.N=128 fs2d.use_clahe=1 fs2d.potential_for_necessary_peak=0.01'")
    parser.add_argument("--save-blended", action="store_true",
                        help="Save blended images for each pair.")
    parser.add_argument("data_dir", type=str,
                        help="Path to Boreas radar data directory.")

    args = parser.parse_args()

    # Parse method configs from CLI
    cli_method_configs = _parse_method_config(args.method_config)

    # Build complete method config: start with defaults, then override with CLI values
    method_name = args.method
    method_config = {
        "N": args.N,
        "use_clahe": DEFAULT_CONFIG["use_clahe"],
        "use_hamming": DEFAULT_CONFIG["use_hamming"],
        "use_direct": DEFAULT_CONFIG["use_direct"],
        "use_gauss": DEFAULT_CONFIG["use_gauss"],
        "multiple_radii": DEFAULT_CONFIG["multiple_radii"],
        "potential_for_necessary_peak": DEFAULT_CONFIG["potential_for_necessary_peak"],
        "level_potential_rotation": DEFAULT_CONFIG["level_potential_rotation"],
        "size_of_pixel": args.size_of_pixel,
    }

    # Override with CLI values for this method
    for k, v in cli_method_configs.get(method_name, {}).items():
        method_config[k] = v

    # Print config summary
    print("=" * 80)
    print(f"Boreas Radar Benchmark")
    print("=" * 80)
    print(f"Method: {method_name}")
    print(f"Sequence: {args.sequence}")
    if args.sequence_name is not None:
        print(f"Sequence name: {args.sequence_name}")
    print(f"N: {args.N}")
    print(f"Size of pixel: {args.size_of_pixel} m")
    print(f"Matching step: {args.matching_step}")
    print(f"Start frame: {args.start_frame}")
    print(f"Max frames: {args.max_frames}")
    print(f"Save blended: {args.save_blended}")
    print(f"Output dir: {args.output_dir}")
    print(f"Data dir: {args.data_dir}")
    print(f"Method config: {method_config}")
    print("=" * 80)
    print()

    # Load sequence
    try:
        if args.sequence_name is not None:
            seq = load_single_sequence(args.data_dir, args.sequence_name)
            print(f"Loaded sequence '{args.sequence_name}': {seq.length} radar scans")
        else:
            seq = load_sequence(args.data_dir, args.sequence)
            print(f"Loaded sequence {args.sequence}: {seq.length} radar scans")
    except Exception as e:
        if args.sequence_name is not None:
            print(f"ERROR: Failed to load sequence '{args.sequence_name}': {e}")
        else:
            print(f"ERROR: Failed to load sequence {args.sequence}: {e}")
        sys.exit(1)

    # Run benchmark
    try:
        results_csv, summary = run_benchmark(
            seq=seq,
            method_name=method_name,
            method_config=method_config,
            sequence_number=args.sequence,
            matching_step=args.matching_step,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            save_blended=args.save_blended,
            output_dir=args.output_dir,
        )
    except Exception as e:
        print(f"ERROR: Benchmark failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
