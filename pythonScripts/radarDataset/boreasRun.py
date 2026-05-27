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
Entry point for Boreas registration framework.

Reads config (top of file) and/or CLI args, loads data, runs registration
methods, and saves results.

Usage:
    python boreasRun.py --method fs2d --sequence 0 --size_of_pixel 0.01 <data_dir>
    python boreasRun.py --method fs2d --method icp --compare --sequences 0,1 --size_of_pixel 0.01 <data_dir>
"""

import argparse
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List

from boreasDatasetLoader import (
    BoreasSequence,
    load_sequence,
    get_affine_matrix,
    transform_diff,
    matrix_to_transform,
    fuse_images,
)
from boreasRegistrationMethods import RegistrationFactory, RegistrationResult


# ============================================================================
# Default configuration
# ============================================================================

DEFAULT_CONFIG = {
    "sequences": [0],
    "N": 128,
    "size_of_pixel": 0.01,
    "matching_every_nth_image": 1,
    "max_frames": None,
    "output_dir": "saveResultsBoreas",
    "methods": ["fs2d"],
}


# ============================================================================
# Sequence runner
# ============================================================================

def run_sequence(args, method_configs: Dict[str, dict], seq: BoreasSequence, sequence_number: int):
    """Run a single sequence with one or more methods.

    Args:
        args: CLI arguments.
        method_configs: dict of method_name -> config dict.
        seq: BoreasSequence instance.
        sequence_number: Sequence number for output directory naming.
    """
    N = args.N
    size_of_pixel = args.size_of_pixel
    matching_every_nth = args.matching_every_nth_image
    max_frames = args.max_frames

    method_key = "_".join(sorted(method_configs.keys()))
    save_dir = Path(
        f"{args.output_dir}/{sequence_number:02d}_{N:03d}_{int(size_of_pixel*100)}_{matching_every_nth}/{method_key}/"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {save_dir}")

    length_of_radar_scans = seq.length
    if max_frames is not None:
        length_of_radar_scans = min(length_of_radar_scans, max_frames)
    print(f"Sequence has {seq.length} radar scans (using {length_of_radar_scans})")

    all_method_results = {}
    for method_name in method_configs:
        all_method_results[method_name] = {
            "transforms": [],
            "confidence": [],
            "errors": [],
            "times": [],
            "gt_poses": [],
            "est_poses": [],
        }

    first_img = seq.get_cartesian_image(0, N, size_of_pixel)
    images_over_time = [first_img * 255.0]

    cumulative_transforms = {}
    for method_name in method_configs:
        cumulative_transforms[method_name] = [np.eye(4)]

    print(f"Matching every {matching_every_nth}th image...")

    for index in range(1, length_of_radar_scans):
        if index >= matching_every_nth:
            if index % matching_every_nth == 0:
                prev_index = index - matching_every_nth

                img_prev = seq.get_cartesian_image(prev_index, N, size_of_pixel)
                img_curr = seq.get_cartesian_image(index, N, size_of_pixel)
                images_over_time.append(img_curr * 255.0)

                gt_transform = seq.get_gt_transform(prev_index, index)
                gt_affine = get_affine_matrix(gt_transform)

                for method_name, config in method_configs.items():
                    method = RegistrationFactory.create(method_name, config)
                    result = method.register(img_prev, img_curr)

                    prev_cumulative = cumulative_transforms[method_name][-1]
                    new_cumulative = prev_cumulative @ result.transform
                    cumulative_transforms[method_name].append(new_cumulative)

                    est_affine = get_affine_matrix(result.transform)
                    trans_error, rot_error = transform_diff(gt_affine, est_affine)

                    all_method_results[method_name]["transforms"].append(result.transform)
                    all_method_results[method_name]["confidence"].append(result.confidence)
                    all_method_results[method_name]["errors"].append((trans_error, rot_error))
                    all_method_results[method_name]["times"].append(result.computation_time)
                    all_method_results[method_name]["gt_poses"].append(gt_transform)
                    all_method_results[method_name]["est_poses"].append(new_cumulative)

                    print(f"  [{method_name}] idx={index}: rot_err={rot_error:.3f} deg, "
                          f"trans_err={np.linalg.norm(trans_error):.4f}m, "
                          f"time={result.computation_time*1000:.1f}ms, "
                          f"conf={result.confidence:.4f}")

                if "fs2d" in method_configs:
                    fs2d_results = all_method_results["fs2d"]
                    fs2d_result = fs2d_results["transforms"][-1]
                    fs2d_affine = get_affine_matrix(fs2d_result)
                    warped = cv2.warpPerspective(img_curr, fs2d_affine, (img_prev.shape[1], img_prev.shape[0]))
                    blended = cv2.addWeighted(img_prev, 0.5, warped, 0.5, 0)
                    cv2.imwrite(str(save_dir / f"blended_{index:04d}.png"), blended * 255.0)

    for method_name, data in all_method_results.items():
        errors = data["errors"]
        rotation_errors = [e[1] for e in errors]
        translation_errors = [np.linalg.norm(e[0]) for e in errors]

        with open(save_dir / f"{method_name}_rotation_error.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for val in rotation_errors:
                writer.writerow([val])

        with open(save_dir / f"{method_name}_translation_error.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for val in translation_errors:
                writer.writerow([val])

        with open(save_dir / f"{method_name}_gt_poses.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for mat in data["gt_poses"]:
                x, y, z, roll, pitch, yaw = matrix_to_transform(mat)
                writer.writerow([x, y, yaw])

        with open(save_dir / f"{method_name}_est_poses.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for mat in data["est_poses"]:
                x, y, z, roll, pitch, yaw = matrix_to_transform(mat)
                writer.writerow([x, y, yaw])

        if method_name == "fs2d" and len(images_over_time) > 1:
            fused = fuse_images(images_over_time, cumulative_transforms[method_name])
            if fused is not None:
                cv2.imwrite(str(save_dir / "fused_map.png"), fused)

        avg_rot = np.mean(np.abs(rotation_errors))
        avg_trans = np.mean(translation_errors)
        avg_time = np.mean(data["times"])
        print(f"\n[{method_name}] Summary:")
        print(f"  Avg rotation error: {avg_rot:.3f} deg")
        print(f"  Avg translation error: {avg_trans:.4f} m")
        print(f"  Avg computation time: {avg_time*1000:.1f} ms")

    if len(method_configs) > 1:
        with open(save_dir / "comparison.csv", "w", newline="") as f:
            writer = csv.writer(f)
            header = ["index"]
            for method_name in sorted(method_configs.keys()):
                header.extend([f"{method_name}_rot_error", f"{method_name}_trans_error",
                              f"{method_name}_time_ms", f"{method_name}_confidence"])
            header.append("gt_rot_error")
            header.append("gt_trans_error")
            writer.writerow(header)

            for i in range(len(errors)):
                row = [i]
                for method_name in sorted(method_configs.keys()):
                    data = all_method_results[method_name]
                    rot_e = np.abs(data["errors"][i][1])
                    trans_e = np.linalg.norm(data["errors"][i][0])
                    t_ms = data["times"][i] * 1000
                    conf = data["confidence"][i]
                    row.extend([rot_e, trans_e, t_ms, conf])
                gt_trans_e = np.linalg.norm(errors[i][0])
                gt_rot_e = np.abs(errors[i][1])
                row.extend([gt_rot_e, gt_trans_e])
                writer.writerow(row)

    print(f"\nDone. Results saved to: {save_dir}")


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


# ============================================================================
# Main
# ============================================================================

def main():
    import cv2

    np.set_printoptions(precision=5, suppress=True)

    parser = argparse.ArgumentParser(
        description="Extensible registration framework for Boreas radar dataset."
    )
    parser.add_argument("--method", action="append", required=True,
                        help="Registration method(s) to run. Can specify multiple for comparison. "
                             f"Available: {RegistrationFactory.list_methods()}")
    parser.add_argument("--method-config", action="append", default=[],
                        help="Method config in format 'method_name.key=value'. "
                             "e.g., 'fs2d.N=128 fs2d.use_clahe=1 fs2d.potential_for_necessary_peak=0.01'")
    parser.add_argument("--N", type=int, default=DEFAULT_CONFIG["N"],
                        help=f"Image grid size (N x N). Default: {DEFAULT_CONFIG['N']}")
    parser.add_argument("--size_of_pixel", type=float, default=DEFAULT_CONFIG["size_of_pixel"],
                        help=f"Size of a pixel in meters. Default: {DEFAULT_CONFIG['size_of_pixel']}")
    parser.add_argument("--matching_every_nth_image", type=int,
                        default=DEFAULT_CONFIG["matching_every_nth_image"],
                        help=f"Match every Nth image. Default: {DEFAULT_CONFIG['matching_every_nth_image']}")
    parser.add_argument("--sequence", type=int, default=None,
                        help="Single sequence number (deprecated, use --sequences).")
    parser.add_argument("--sequences", type=str, default=None,
                        help="Comma-separated list of sequence numbers. Default: 0")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Cap sequence length (None = full).")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_CONFIG["output_dir"],
                        help=f"Output directory. Default: {DEFAULT_CONFIG['output_dir']}")
    parser.add_argument("--compare", action="store_true",
                        help="Enable comparison mode when multiple methods are specified.")
    parser.add_argument("data_dir", type=str, help="Path to Boreas radar data directory.")

    args = parser.parse_args()

    # Resolve sequences
    if args.sequences is not None:
        sequences = [int(s.strip()) for s in args.sequences.split(",")]
    elif args.sequence is not None:
        sequences = [args.sequence]
    else:
        sequences = DEFAULT_CONFIG["sequences"]

    # Parse method configs
    method_configs = {}
    for spec in args.method_config:
        parts = spec.split()
        for part in parts:
            method_key, _, key_value = part.partition(".")
            if "." not in method_key:
                continue
            method_name, _, param = method_key.partition(".")
            k, _, v = param.partition("=")
            v = _parse_value(v)
            if method_name not in method_configs:
                method_configs[method_name] = {}
            method_configs[method_name][k] = v

    # Apply default config values if not overridden
    default_fs2d_config = {
        "N": args.N,
        "use_clahe": True,
        "use_hamming": True,
        "potential_for_necessary_peak": 0.01,
        "multiple_radii": True,
        "use_gauss": False,
        "size_of_pixel": args.size_of_pixel,
    }
    if "fs2d" not in method_configs:
        method_configs["fs2d"] = default_fs2d_config
    else:
        for k, v in default_fs2d_config.items():
            if k not in method_configs["fs2d"]:
                method_configs["fs2d"][k] = v

    print(f"Methods: {list(method_configs.keys())}")
    for name, cfg in method_configs.items():
        print(f"  {name}: {cfg}")

    for seq_num in sequences:
        print(f"\n{'='*60}")
        print(f"Running sequence {seq_num}")
        print(f"{'='*60}")

        seq = load_sequence(args.data_dir, seq_num)
        methods_str = "_".join(sorted(method_configs.keys()))

        # Temporarily override output_dir in args
        original_output_dir = args.output_dir
        args.output_dir = f"{original_output_dir}"

        run_sequence(args, method_configs, seq, seq_num)


if __name__ == "__main__":
    main()
