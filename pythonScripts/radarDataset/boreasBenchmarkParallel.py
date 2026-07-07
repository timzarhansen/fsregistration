"""
Parallel Boreas Radar Benchmarking Script

Processes multiple sequences in parallel, each handled by a worker thread.
Outputs per-sequence results (same format as boreasBenchmark.py).

Usage:
    # Single machine, all sequences
    python boreasBenchmarkParallel.py --method fs2d --sequences all \\
        --N 128 --size_of_pixel 0.5 --matching_step 5 --num-workers 4 \\
        --output-dir benchmark_results <data_dir>

    # Multi-machine: machine 1 covers sequences 0-15
    python boreasBenchmarkParallel.py --method fs2d --sequences 0-15 \\
        --N 128 --num-workers 4 --output-dir results_m1 <data_dir>

    # Machine 2 covers sequences 16-30
    python boreasBenchmarkParallel.py --method fs2d --sequences 16-30 \\
        --N 128 --num-workers 4 --output-dir results_m2 <data_dir>

    # Quick test: 5 sequences, 10 pairs each
    python boreasBenchmarkParallel.py --method fs2d --sequences 0-4 \\
        --N 64 --max_frames 10 --matching_step 1 --num-workers 2 \\
        --output-dir test_results <data_dir>
"""

import argparse
import os
import sys
import time
import traceback
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_script_dir))))
_install_lib = os.path.join(_root_dir, 'install', 'fsregistration', 'lib', 'fsregistration')
if os.path.isdir(_install_lib):
    sys.path.insert(0, _install_lib)

from pyboreas import BoreasDataset

from boreasBenchmark import run_benchmark
from boreasDatasetLoader import load_single_sequence


# ============================================================================
# Helpers
# ============================================================================

def parse_sequence_spec(spec: str) -> List[int]:
    """Parse a sequence specification into a list of sequence numbers.

    Accepts:
      - '0-15'       → range (inclusive)
      - '0,1,2,5'    → comma-separated list
      - '0'          → single number
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
            f"Invalid sequence spec: '{spec}'. Use 'all', range (0-15), "
            f"comma list (0,1,2), or single number."
        )


def worker_process(args: tuple) -> Tuple[int, bool, str, dict]:
    """Process a single sequence in a worker thread.

    Each worker loads its own BoreasSequence by name (avoids scanning
    all sequences in BoreasDataset.__init__).
    """
    seq_num, seq_name, data_dir, method_name, method_config, matching_step, \
        start_frame, max_frames, save_blended, output_dir = args

    try:
        pid = os.getpid()
        print(f"[Worker {pid}] Loading sequence {seq_num} ({seq_name})...")
        seq = load_single_sequence(data_dir, seq_name)
        print(f"[Worker {pid}] Sequence {seq_num}: {seq.length} frames")

        csv_path, summary = run_benchmark(
            seq=seq,
            method_name=method_name,
            method_config=method_config,
            sequence_number=seq_num,
            matching_step=matching_step,
            start_frame=start_frame,
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
        description="Parallel Boreas Radar Benchmarking"
    )
    parser.add_argument("--method", type=str, required=True,
                        help="Registration method to benchmark.")
    parser.add_argument("--sequences", type=str, default="all",
                        help="Sequence spec: 'all', '0-15', '0,1,2', or '0'.")
    parser.add_argument("--N", type=int, default=128,
                        help="Image grid size (N x N). Default: 128")
    parser.add_argument("--size_of_pixel", type=float, default=0.5,
                        help="Size of a pixel in meters. Default: 0.5")
    parser.add_argument("--matching_step", type=int, default=5,
                        help="Match every Nth frame. Default: 5")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="First frame index. Default: 0")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Cap sequence length (None = full).")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel worker processes. Default: 4")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Output directory. Default: benchmark_results")
    parser.add_argument("--method-config", action="append", default=[],
                        help="Method config in format 'method_name.key=value'.")
    parser.add_argument("--save-blended", action="store_true",
                        help="Save blended images for each pair.")
    parser.add_argument("data_dir", type=str,
                        help="Path to Boreas radar data directory.")

    args = parser.parse_args()

    # Build index→name mapping (scan BoreasDataset once here, not per worker)
    print("Scanning Boreas dataset to resolve sequence names...")
    bd = BoreasDataset(args.data_dir, split=None, verbose=False)
    all_sequences = len(bd.sequences)

    if args.sequences.lower() == "all":
        sequence_numbers = list(range(all_sequences))
    else:
        sequence_numbers = parse_sequence_spec(args.sequences)

    seq_names = {
        i: str(bd.sequences[i].ID)
        for i in sequence_numbers
        if i < all_sequences
    }
    missing = [i for i in sequence_numbers if i not in seq_names]
    if missing:
        print(f"ERROR: Sequence indices not found: {missing} (dataset has {all_sequences} sequences)")
        sys.exit(1)
    del bd  # free memory, done with it

    print(f"Sequences to process: {len(sequence_numbers)} — {sequence_numbers}")
    print(f"Method: {args.method}")
    print(f"N={args.N}, size_of_pixel={args.size_of_pixel}, "
          f"matching_step={args.matching_step}")
    print(f"Workers: {args.num_workers}")
    print(f"Output: {args.output_dir}")
    print()

    # Build method config
    from boreasBenchmark import DEFAULT_CONFIG as _BDC
    method_config = {
        "N": args.N,
        "use_clahe": _BDC["use_clahe"],
        "use_hamming": _BDC["use_hamming"],
        "use_direct": _BDC["use_direct"],
        "use_gauss": _BDC["use_gauss"],
        "multiple_radii": _BDC["multiple_radii"],
        "potential_for_necessary_peak": _BDC["potential_for_necessary_peak"],
        "level_potential_rotation": _BDC["level_potential_rotation"],
        "use_weighted_peak_score": _BDC["use_weighted_peak_score"],
        "size_of_pixel": args.size_of_pixel,
    }

    # Parse method config overrides
    def _parse_value(v):
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

    # Build worker arguments
    worker_args = [
        (
            seq_num,
            seq_names[seq_num],
            args.data_dir,
            args.method,
            method_config,
            args.matching_step,
            args.start_frame,
            args.max_frames,
            args.save_blended,
            args.output_dir,
        )
        for seq_num in sequence_numbers
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

    # Exit with error if any sequences failed
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
