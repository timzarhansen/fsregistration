"""
Parallel Bremen-MSS 2D Benchmarking Script

Processes multiple sequences in parallel, each handled by a worker thread.
Outputs per-sequence results (same format as bremenMssBenchmark.py).

Usage:
    # All 13 sequences
    python bremenMssBenchmarkParallel.py --method fs2d --sequences all \\
        --N 256 --radius 22.5 --num-workers 4 \\
        --output-dir benchmark_results \\
        /home/tim-external/dataFolder/Bremen-MSS-Processed

    # Specific sequences
    python bremenMssBenchmarkParallel.py --method sift --sequences 1-5 \\
        --num-workers 3 --output-dir results \\
        /home/tim-external/dataFolder/Bremen-MSS-Processed

    # Quick test
    python bremenMssBenchmarkParallel.py --method fs2d --sequences 1-3 \\
        --max_frames 5 --num-workers 2 --output-dir test_results \\
        /home/tim-external/dataFolder/Bremen-MSS-Processed
"""

import argparse
import os
import sys
import time
import traceback
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np

from bremenMssBenchmark import run_benchmark
from bremenMssDatasetLoader import list_sequences, load_single_sequence


# ============================================================================
# Helpers
# ============================================================================

def parse_sequence_spec(spec: str) -> List[int]:
    """Parse a sequence specification into a list of sequence numbers.

    Accepts:
      - '1-5'         → range (inclusive)
      - '1,2,3,5'     → comma-separated list
      - '1'           → single number
    """
    if "-" in spec and not spec.startswith("--"):
        parts = spec.split("-")
        if len(parts) == 2:
            try:
                start, end = int(parts[0]), int(parts[1])
                return list(range(start, end + 1))
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
            f"Invalid sequence spec: '{spec}'. Use 'all', range (1-5), "
            f"comma list (1,2,3), or single number."
        )


def worker_process(args: tuple) -> Tuple[int, bool, str, dict]:
    """Process a single sequence in a worker thread."""
    seq_num, seq_name, seq_path, data_dir, method_name, method_config, \
        save_blended, output_dir = args

    try:
        pid = os.getpid()
        print(f"[Worker {pid}] Loading sequence {seq_num} ({seq_name})...")
        seq = load_single_sequence(data_dir, seq_name)
        print(f"[Worker {pid}] Sequence {seq_num}: {seq.length} scans")

        csv_path, summary = run_benchmark(
            seq=seq,
            method_name=method_name,
            method_config=method_config,
            sequence_number=seq_num,
            matching_step=1,              # always 1 for Bremen-MSS
            start_frame=0,
            max_frames=None,
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
        description="Parallel Bremen-MSS 2D Benchmarking"
    )
    parser.add_argument("--method", type=str, required=True,
                        help="Registration method to benchmark.")
    parser.add_argument("--sequences", type=str, default="all",
                        help="Sequence spec: 'all', '1-5', '1,2,3', or '1'.")
    parser.add_argument("--N", type=int, default=256,
                        help="Image grid size (N x N). Default: 256")
    parser.add_argument("--radius", type=float, default=22.5,
                        help="Scene radius in meters (pixel_size = 2*radius/N). Default: 22.5")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel worker processes. Default: 4")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Output directory. Default: benchmark_results")
    parser.add_argument("--method-config", action="append", default=[],
                        help="Method config in format 'method_name.key=value'.")
    parser.add_argument("--save-blended", action="store_true",
                        help="Save blended images for each pair.")
    parser.add_argument("data_dir", type=str,
                        help="Path to Bremen-MSS-Processed data directory.")

    args = parser.parse_args()

    # Discover all sequences
    all_seqs = list_sequences(args.data_dir)
    if not all_seqs:
        print(f"ERROR: No sequences found in {args.data_dir}")
        sys.exit(1)

    seq_map = {seq_num: (seq_name, seq_path) for seq_num, seq_name, seq_path in all_seqs}

    if args.sequences.lower() == "all":
        sequence_numbers = sorted(seq_map.keys())
    else:
        sequence_numbers = parse_sequence_spec(args.sequences)

    # Filter to available sequences
    valid_seqs = [s for s in sequence_numbers if s in seq_map]
    missing = [s for s in sequence_numbers if s not in seq_map]
    if missing:
        print(f"WARNING: Sequence(s) not found: {missing} (available: {sorted(seq_map.keys())})")
    if not valid_seqs:
        print("ERROR: No valid sequences to process.")
        sys.exit(1)

    print(f"Sequences to process: {len(valid_seqs)} — {valid_seqs}")
    print(f"Method: {args.method}")
    print(f"N={args.N}, radius={args.radius} m (pixel_size: {(2.0 * args.radius) / args.N:.3f} m)")
    print(f"Workers: {args.num_workers}")
    print(f"Output: {args.output_dir}")
    print()

    # Build base method config
    method_config = {
        "N": args.N,
        "radius": args.radius,
        "size_of_pixel": (2.0 * args.radius) / args.N,
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

    print(f"Method config: {method_config}")
    print()

    # Build worker arguments (each worker loads its own sequence)
    worker_args = [
        (
            seq_num,
            seq_map[seq_num][0],   # seq_name
            seq_map[seq_num][1],   # seq_path
            args.data_dir,
            args.method,
            method_config,
            args.save_blended,
            args.output_dir,
        )
        for seq_num in valid_seqs
    ]

    # Run in parallel
    t_start = time.time()

    if args.num_workers == 1:
        results = [worker_process(a) for a in worker_args]
    else:
        with Pool(processes=args.num_workers) as pool:
            results = pool.map(worker_process, worker_args)

    elapsed = time.time() - t_start

    # Report
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    print()
    print("=" * 75)
    print(f"All sequences done in {elapsed:.1f}s")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    if failed:
        print(f"  Failed sequences: {[r[0] for r in failed]}")
    print()

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
