"""
Parallel Lidar Simulation Benchmark Script

Processes multiple dataset directories in parallel, each handled by a
worker thread.  Outputs per-sequence results (same CSV format as the
Bremen-MSS and Boreas benchmarks).

All registration methods (fs2d, icp, sift, kaze, akaze, surf, imreg,
ndt_p2d, loftr, eloftr, lightglue) work transparently through the
RegistrationFactory.

Usage:
    # All datasets, FS2D, 4 workers
    python lidarSimBenchmarkParallel.py --method fs2d \\
        --N 256 --radius 15 --num-workers 4 \\
        --output-dir benchmark_results /path/to/datasets_parent

    # Specific datasets (sorted by name, index 1-3)
    python lidarSimBenchmarkParallel.py --method icp \\
        --sequences 1-3 --num-workers 2 \\
        --output-dir results /path/to/datasets_parent

    # Quick test (5 pairs per dataset)
    python lidarSimBenchmarkParallel.py --method sift \\
        --max_frames 5 --num-workers 2 \\
        --output-dir test_results /path/to/datasets_parent
"""

import argparse
import os
import sys
import time
import traceback
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np

from lidarSimDatasetLoader import (LidarSimSequence, NOISE_LEVELS,
                                   list_sequences)
from bremenMssBenchmark import run_benchmark


# ============================================================================
# Helpers
# ============================================================================

def parse_sequence_spec(spec: str) -> List[int]:
    """Parse sequence specification into a list of indices (1-based)."""
    if "-" in spec and not spec.startswith("--"):
        parts = spec.split("-")
        if len(parts) == 2:
            try:
                return list(range(int(parts[0]), int(parts[1]) + 1))
            except ValueError:
                pass
    if "," in spec:
        try:
            return [int(s.strip()) for s in spec.split(",") if s.strip()]
        except ValueError:
            pass
    try:
        return [int(spec)]
    except ValueError:
        raise ValueError(
            f"Invalid sequence spec: '{spec}'. "
            f"Use 'all', range (1-5), comma list (1,2,3), or single number."
        )


def worker_process(args: tuple) -> Tuple[int, bool, str, dict]:
    """Process a single dataset directory in a worker thread."""
    seq_num, seq_name, seq_path, data_dir, method_name, method_config, \
        save_blended, output_dir, max_frames = args

    try:
        pid = os.getpid()
        print(f"[Worker {pid}] Loading sequence {seq_num} ({seq_name})...")
        seq = LidarSimSequence(
            seq_path, noise_level=method_config.get("noise_level", "None"))
        print(f"[Worker {pid}] Sequence {seq_num}: {seq.length} scans")

        csv_path, summary = run_benchmark(
            seq=seq,
            method_name=method_name,
            method_config=method_config,
            sequence_number=seq_num,
            matching_step=1,                # consecutive pairs only
            start_frame=0,
            max_frames=max_frames,
            save_blended=save_blended,
            output_dir=output_dir,
        )

        print(f"[Worker {pid}] Sequence {seq_num} done: {csv_path}")
        return seq_num, True, str(csv_path), summary

    except Exception as e:
        print(f"[ERROR] Sequence {seq_num} failed: {e}")
        traceback.print_exc()
        return seq_num, False, str(e), {}


# ============================================================================
# Main
# ============================================================================

def main():
    np.set_printoptions(precision=5, suppress=True)

    parser = argparse.ArgumentParser(
        description="Parallel Lidar Simulation Benchmark"
    )
    parser.add_argument("--method", type=str, required=True,
                        help="Registration method to benchmark.")
    parser.add_argument("--sequences", type=str, default="all",
                        help="Sequence spec: 'all', '1-5', '1,2,3', or '1'.")
    parser.add_argument("--N", type=int, default=256,
                        help="Image grid size (N x N). Default: 256")
    parser.add_argument("--radius", type=float, default=15.0,
                        help="Scene radius in meters (pixel_size = 2*radius/N). Default: 15")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel worker processes. Default: 4")
    parser.add_argument("--noise-level", type=str, default="None",
                        choices=NOISE_LEVELS,
                        help="Noise model applied to points before image/pcd "
                             f"generation. Choices: {NOISE_LEVELS}")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Cap sequence length (None = full).")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Output directory. Default: benchmark_results")
    parser.add_argument("--method-config", action="append", default=[],
                        help="Method config in format 'method_name.key=value'.")
    parser.add_argument("--save-blended", action="store_true",
                        help="Save blended images for each pair.")
    parser.add_argument("data_dir", type=str,
                        help="Path to parent directory containing dataset subdirs.")

    args = parser.parse_args()

    # Discover datasets
    all_seqs = list_sequences(args.data_dir)
    if not all_seqs:
        print(f"ERROR: No simulation datasets found in {args.data_dir}")
        sys.exit(1)

    seq_map = {seq_num: (seq_name, seq_path)
               for seq_num, seq_name, seq_path in all_seqs}

    if args.sequences.lower() == "all":
        sequence_numbers = sorted(seq_map.keys())
    else:
        sequence_numbers = parse_sequence_spec(args.sequences)

    valid_seqs = [s for s in sequence_numbers if s in seq_map]
    missing = [s for s in sequence_numbers if s not in seq_map]
    if missing:
        print(f"WARNING: Sequence(s) not found: {missing} "
              f"(available: {sorted(seq_map.keys())})")
    if not valid_seqs:
        print("ERROR: No valid sequences to process.")
        sys.exit(1)

    print(f"Sequences to process: {len(valid_seqs)} — {valid_seqs}")
    print(f"Method: {args.method}")
    print(f"N={args.N}, radius={args.radius} m "
          f"(pixel_size: {(2.0 * args.radius) / args.N:.3f} m)")
    print(f"Workers: {args.num_workers}")
    print(f"Max frames per seq: {args.max_frames if args.max_frames else 'full'}")
    print(f"Noise level: {args.noise_level}")
    print(f"Output: {args.output_dir}")
    print()

    # Build base method config
    method_config = {
        "N": args.N,
        "radius": args.radius,
        "size_of_pixel": (2.0 * args.radius) / args.N,
        "noise_level": args.noise_level,
    }

    # Parse method config overrides
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

    for spec in args.method_config:
        parts = spec.split()
        for part in parts:
            if "." not in part:
                continue
            mn, _, param = part.partition(".")
            if mn != args.method:
                continue
            k, _, v = param.partition("=")
            method_config[k] = _parse_value(v)

    # Build worker arguments
    worker_args = []
    for seq_num in valid_seqs:
        seq_name, seq_path = seq_map[seq_num]
        worker_args.append((
            seq_num, seq_name, seq_path, args.data_dir,
            args.method, method_config,
            args.save_blended, args.output_dir,
            args.max_frames,
        ))

    # Run in parallel
    t_start = time.time()
    results_list = []
    num_workers = min(args.num_workers, len(worker_args))

    if num_workers <= 1:
        # Sequential (useful for debugging)
        for wa in worker_args:
            results_list.append(worker_process(wa))
    else:
        with Pool(num_workers) as pool:
            results_list = pool.map(worker_process, worker_args)

    elapsed = time.time() - t_start

    # Summary
    succeeded = [r for r in results_list if r[1]]
    failed = [r for r in results_list if not r[1]]

    print()
    print("=" * 75)
    print(f"Benchmark complete: {len(succeeded)} OK, {len(failed)} failed "
          f"in {elapsed:.0f}s")
    print(f"Results in: {os.path.abspath(args.output_dir)}/")

    if failed:
        print()
        print("Failed sequences:")
        for seq_num, _, err, _ in failed:
            print(f"  seq {seq_num}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
