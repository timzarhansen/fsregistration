"""
Bremen-MSS 2D Benchmark — Single-sequence runner.

Processes one sequence, registering consecutive scans (matching_step=1)
and producing a CSV with per-pair error metrics.

Usage:
    python bremenMssBenchmark.py --method fs2d --sequence 1 \\
        --output-dir benchmark_results \\
        /home/tim-external/dataFolder/Bremen-MSS-Processed

    python bremenMssBenchmark.py --method icp --sequence 3 \\
        --N 256 --radius 22.5 \\
        --output-dir benchmark_results \\
        /home/tim-external/dataFolder/Bremen-MSS-Processed
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

# Add paths for pybind_registration_2d
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_script_dir))))
_install_lib = os.path.join(_root_dir, 'install', 'fsregistration', 'lib', 'fsregistration')
if os.path.isdir(_install_lib):
    sys.path.insert(0, _install_lib)

from bremenMssDatasetLoader import BremenMSSSequence, load_sequence, list_sequences
from boreasRegistrationMethods import RegistrationFactory, RegistrationResult
from scipy.spatial.transform import Rotation as R


# ============================================================================
# Benchmark runner
# ============================================================================

def run_benchmark(
    seq: BremenMSSSequence,
    method_name: str,
    method_config: dict,
    sequence_number: int,
    matching_step: int,
    start_frame: int,
    max_frames: Optional[int],
    save_blended: bool,
    output_dir: str,
    pcd1: Optional[np.ndarray] = None,
    pcd2: Optional[np.ndarray] = None,
) -> Tuple[Path, dict]:
    """Run benchmark on a single Bremen-MSS sequence with one method.

    Matches consecutive scans (step=1 by default) and computes
    error metrics from estimated vs GT relative transforms.

    Args:
        seq: BremenMSSSequence instance.
        method_name: Registration method name.
        method_config: Method configuration dict (must include N, radius, size_of_pixel).
        sequence_number: Sequence number for output naming.
        matching_step: Match every Nth scan (default 1 = consecutive).
        start_frame: First frame index.
        max_frames: Cap sequence length (None = full).
        save_blended: Whether to save blended images.
        output_dir: Base output directory.
        pcd1: Ignored (per-scan PCDs loaded individually for ICP/NDT).
        pcd2: Ignored.

    Returns:
        Tuple of (results_csv_path, summary_dict).
    """
    N = method_config.get("N", 256)
    radius = method_config.get("radius", 22.5)
    size_of_pixel = (2.0 * radius) / N

    # Determine number of frames
    total_frames = seq.length
    if max_frames is not None:
        end_frame = min(start_frame + max_frames, total_frames)
    else:
        end_frame = total_frames

    num_pairs = 0
    if end_frame > start_frame + matching_step:
        num_pairs = max(0, (end_frame - start_frame - 1) // matching_step)

    # Setup output directory
    px_int = int(size_of_pixel * 100)
    seq_label = f"seq{sequence_number:02d}"
    output_subdir = f"{seq_label}_{method_name}_N{N:03d}_p{px_int}_s{matching_step}"
    save_dir = Path(output_dir) / output_subdir
    save_dir.mkdir(parents=True, exist_ok=True)

    # Blended images directory
    blended_dir = None
    if save_blended:
        blended_dir = save_dir / "blended"
        blended_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving results to: {save_dir}")
    print(f"Sequence has {total_frames} scans (processing scans {start_frame} to {end_frame})")
    print(f"Matching every {matching_step}th scan -> {num_pairs} pairs")
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
          f"{'BestRot°':>8} {'BestTrans(m)':>12} "
          f"{'Conf':>6} {'Time(ms)':>9} {'Status':>8}")
    print("-" * 90)

    for pair_idx in range(num_pairs):
        prev_idx = start_frame + pair_idx * matching_step
        curr_idx = start_frame + (pair_idx + 1) * matching_step

        try:
            # Load images
            img_prev = seq.get_cartesian_image(prev_idx, N, size_of_pixel)
            img_curr = seq.get_cartesian_image(curr_idx, N, size_of_pixel)

            # Get GT transformation
            gt_transform = seq.get_gt_transform(prev_idx, curr_idx)

            # Load per-scan point clouds for ICP/NDT methods
            reg_kwargs = {}
            try:
                pcd_prev = seq.get_raw_point_cloud(prev_idx)
                pcd_curr = seq.get_raw_point_cloud(curr_idx)
                if pcd_prev is not None and pcd_curr is not None:
                    reg_kwargs["pcd1"] = pcd_prev
                    reg_kwargs["pcd2"] = pcd_curr
            except Exception:
                pass  # fall back to image-based registration

            # Run registration
            t0 = time.time()
            result = method.register(img_prev, img_curr, **reg_kwargs)
            elapsed = time.time() - t0
            total_time += elapsed

            # FS2D returns transform in C++ convention. Fix to standard SE3.
            est_transform = fix_fs2d_transform(result.transform)

            # Compute error directly in SE3 space
            rot_error, trans_norm = compute_se3_error(est_transform, gt_transform)
            trans_x_error = est_transform[0, 3] - gt_transform[0, 3]
            trans_y_error = est_transform[1, 3] - gt_transform[1, 3]

            # Extract estimated pose components (from fixed transform)
            est_yaw = np.degrees(np.arctan2(est_transform[1, 0], est_transform[0, 0]))
            est_tx = est_transform[0, 3]
            est_ty = est_transform[1, 3]

            # Extract GT pose components (directly from 4x4 GT)
            gt_yaw = np.degrees(np.arctan2(gt_transform[1, 0], gt_transform[0, 0]))
            gt_tx = gt_transform[0, 3]
            gt_ty = gt_transform[1, 3]

            # Find best solution among all candidates (closest to GT)
            best_rot_error = float('inf')
            best_trans_error = float('inf')
            best_rot_deg = 0.0
            best_tx = 0.0
            best_ty = 0.0
            all_solutions = result.metadata.get("all_solutions", [])
            for sol in all_solutions:
                sol_fixed = fix_fs2d_transform(sol)
                s_rot_err, s_trans_norm = compute_se3_error(sol_fixed, gt_transform)
                s_rot_deg = np.degrees(np.arctan2(sol_fixed[1, 0], sol_fixed[0, 0]))
                s_tx = sol_fixed[0, 3]
                s_ty = sol_fixed[1, 3]
                if abs(s_rot_err) < abs(best_rot_error) or (abs(s_rot_err) == abs(best_rot_error) and s_trans_norm < best_trans_error):
                    best_rot_error = s_rot_err
                    best_trans_error = s_trans_norm
                    best_rot_deg = s_rot_deg
                    best_tx = s_tx
                    best_ty = s_ty

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
                "best_rot_error_deg": best_rot_error,
                "best_trans_error_m": best_trans_error,
                "best_rot_deg": best_rot_deg,
                "best_tx_m": best_tx,
                "best_ty_m": best_ty,
                "num_solutions": len(all_solutions),
            }
            results.append(row)

            # Save blended image if requested
            if save_blended and blended_dir is not None:
                warp_affine = get_bremenmss_affine(est_transform,
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
                  f"{best_rot_error:8.3f} {best_trans_error:12.4f} "
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
        "sequence_name": f"sequence_{sequence_number}",
        "N": N,
        "radius": radius,
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
            "best_rot_error_deg", "best_trans_error_m",
            "best_rot_deg", "best_tx_m", "best_ty_m",
            "num_solutions",
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
# Math helpers (copied from boreasDatasetLoader for self-contained usage)
# ============================================================================

def fix_fs2d_transform(transform: np.ndarray) -> np.ndarray:
    """Convert FS2D result.transform (C++ convention) to standard SE3.

    The C++ code returns translationSI = (-dy, -dx) where (dx, dy) is the
    phase correlation shift in the rotation-aligned frame. The correct SE3
    transform is: T = [R_z(yaw) | t] with t = -R_z(yaw) * (translationSI.y, translationSI.x)
    """
    yaw = np.arctan2(transform[1, 0], transform[0, 0])
    tx, ty = transform[0, 3], transform[1, 3]
    R_mat = R.from_euler("z", yaw).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = -R_mat @ np.array([ty, tx, 0.0])
    return T


def compute_se3_error(T_est: np.ndarray, T_gt: np.ndarray) -> tuple:
    """Compute 2D rotation (deg) and translation (m) error between two SE3 transforms.

    Only compares the yaw (z-axis rotation) and xy-translation, since FS2D
    is a 2D registration method while GT may contain pitch/roll components.
    """
    # Rotation error: compare yaw only
    yaw_est = np.arctan2(T_est[1, 0], T_est[0, 0])
    yaw_gt = np.arctan2(T_gt[1, 0], T_gt[0, 0])
    rot_error_deg = float(np.degrees(abs(yaw_est - yaw_gt)))
    rot_error_deg = (rot_error_deg + 180) % 360 - 180
    if rot_error_deg < 0:
        rot_error_deg = -rot_error_deg

    # Translation error: 2D Euclidean distance (ignore z)
    trans_error_m = float(np.linalg.norm(T_est[:2, 3] - T_gt[:2, 3]))

    return rot_error_deg, trans_error_m


def get_bremenmss_affine(transform_se3: np.ndarray,
                          pixel_size: float = 1.0,
                          img_size: int = 0) -> np.ndarray:
    """Convert standard SE3 transform to image warp affine for Bremen-MSS.

    Bremen-MSS images map local-frame (lx, ly) directly to pixel (gx, gy):
      lx = (gx - N/2) * pixel_size
      ly = (gy - N/2) * pixel_size
    So no axis swapping is needed (unlike Boreas).
    """
    # Inverse: maps destination (prev) pixels to source (curr) coordinates
    T_inv = np.linalg.inv(transform_se3)
    result = np.eye(3)
    result[:2, :2] = T_inv[:2, :2]
    result[0, 2] = T_inv[0, 3] / pixel_size
    result[1, 2] = T_inv[1, 3] / pixel_size
    # Rotation center compensation
    if img_size > 0:
        c = img_size / 2.0
        T_c = np.eye(3)
        T_c[:2, 2] = [c, c]
        T_c_inv = np.eye(3)
        T_c_inv[:2, 2] = [-c, -c]
        result = T_c @ result @ T_c_inv
    return result


# ============================================================================
# CLI
# ============================================================================

def _parse_value(v):
    if v.lower() == "true":
        return True
    if v.lower() == "false":
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
    """Parse method config specs in format 'method_name.key=value'."""
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


def main():
    np.set_printoptions(precision=5, suppress=True)

    parser = argparse.ArgumentParser(
        description="Bremen-MSS 2D Benchmark - Single method, single sequence"
    )
    parser.add_argument("--method", type=str, required=True,
                        help="Registration method to benchmark. "
                             f"Available: {RegistrationFactory.list_methods()}")
    parser.add_argument("--sequence", type=int, required=True,
                        help="Sequence number (1-13).")
    parser.add_argument("--N", type=int, default=256,
                        help="Image grid size (N x N). Default: 256")
    parser.add_argument("--radius", type=float, default=22.5,
                        help="Scene radius in meters (pixel_size = 2*radius/N). Default: 22.5")
    parser.add_argument("--matching_step", type=int, default=1,
                        help="Match every Nth scan. Default: 1 (consecutive)")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="First scan index. Default: 0")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Cap sequence length (None = full).")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Output directory. Default: benchmark_results")
    parser.add_argument("--method-config", action="append", default=[],
                        help="Method config in format 'method_name.key=value'.")
    parser.add_argument("--save-blended", action="store_true",
                        help="Save blended images for each pair.")
    parser.add_argument("data_dir", type=str,
                        help="Path to Bremen-MSS-Processed data directory.")

    args = parser.parse_args()

    # Parse method configs from CLI
    cli_method_configs = _parse_method_config(args.method_config)

    # Build complete method config
    method_name = args.method
    method_config = {
        "N": args.N,
        "radius": args.radius,
        "size_of_pixel": (2.0 * args.radius) / args.N,
    }

    # Override with CLI values for this method
    for k, v in cli_method_configs.get(method_name, {}).items():
        method_config[k] = v

    # Print config summary
    print("=" * 80)
    print(f"Bremen-MSS 2D Benchmark")
    print("=" * 80)
    print(f"Method: {method_name}")
    print(f"Sequence: {args.sequence}")
    print(f"N: {args.N}")
    print(f"Radius: {args.radius} m (pixel_size: {(2.0 * args.radius) / args.N:.3f} m)")
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
    seq_name = f"sequence_{args.sequence}"
    seq_dir = os.path.join(args.data_dir, seq_name)
    if not os.path.isdir(seq_dir):
        print(f"ERROR: Sequence directory not found: {seq_dir}")
        sys.exit(1)

    try:
        seq = load_sequence(seq_dir)
        print(f"Loaded sequence {args.sequence:02d}: {seq.length} scans")
    except Exception as e:
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
