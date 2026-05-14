#!/usr/bin/python3
"""
Fix missing indices in a batch-processed output file.

Reads a list of missing indices, groups them into contiguous ranges,
recomputes the missing registrations in parallel, then merges with
the existing output file (sort + dedup).

Usage:
    python bashScripts/fix_missing_indices.py \
        --config configFiles/predatorNothingMac.yaml \
        --noise-level high \
        --data-type train \
        --model-type geotransformer \
        --missing-indices-file bashScripts/missing/outfile_geotransformer_high_train_missing.csv \
        --num-workers 16
"""

import argparse
import csv
import os
import sys
import subprocess
import time
import tempfile
from pathlib import Path


def parse_missing_indices(filepath):
    """Read comma-separated indices from a text file."""
    with open(filepath, 'r') as f:
        content = f.read().strip()
    indices = set()
    for token in content.split(','):
        token = token.strip()
        if token:
            indices.add(int(token))
    return sorted(indices)


def group_into_ranges(indices, max_range_size=100):
    """Group sorted indices into contiguous ranges, max `max_range_size` per range."""
    if not indices:
        return []

    ranges = []
    start = indices[0]
    prev = indices[0]

    for idx in indices[1:]:
        if idx == prev + 1 and (idx - start + 1) <= max_range_size:
            prev = idx
        else:
            ranges.append((start, prev))
            start = idx
            prev = idx

    ranges.append((start, prev))
    return ranges


def process_range(args):
    """Process a single index range by invoking the testing script."""
    start_idx, end_idx, config, noise_level, data_type, script_path, output_dir, model_type, soft_params = args

    temp_dir = os.path.join(output_dir, 'fix_missing_tmp')
    os.makedirs(temp_dir, exist_ok=True)

    temp_csv = os.path.join(temp_dir, f'missing_{start_idx:05d}_{end_idx:05d}.csv')

    if model_type == 'soft':
        cmd = [
            sys.executable,
            script_path,
            config,
            str(soft_params['N']),
            str(soft_params['use_clahe']),
            str(soft_params['r_min']),
            str(soft_params['r_max']),
            str(soft_params['level_rotation']),
            str(soft_params['level_translation']),
            str(soft_params['normalization']),
            noise_level,
            data_type,
            '--start-index', str(start_idx),
            '--end-index', str(end_idx),
            '--output-file', temp_csv
        ]
    else:
        cmd = [
            sys.executable,
            script_path,
            config,
            noise_level,
            data_type,
            '--start-index', str(start_idx),
            '--end-index', str(end_idx),
            '--output-file', temp_csv
        ]

    expected_rows = end_idx - start_idx + 1
    print(f"[Range {start_idx}-{end_idx}] Processing {expected_rows} samples...")

    start_time = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time

        if result.returncode == 0:
            # Validate row count
            if os.path.exists(temp_csv):
                with open(temp_csv, 'r') as f:
                    actual_rows = sum(1 for _ in f) - 1  # minus header
                if actual_rows == expected_rows:
                    print(f"[Range {start_idx}-{end_idx}] OK - {actual_rows} rows in {elapsed:.1f}s")
                    return (start_idx, end_idx, temp_csv, True, elapsed)
                else:
                    print(f"[Range {start_idx}-{end_idx}] WARNING - got {actual_rows} rows, expected {expected_rows}")
            print(f"[Range {start_idx}-{end_idx}] FAILED after {elapsed:.1f}s")
            if result.stderr:
                print(f"  STDERR: {result.stderr[-500:]}")
            return (start_idx, end_idx, temp_csv, False, elapsed)
        else:
            print(f"[Range {start_idx}-{end_idx}] FAILED after {elapsed:.1f}s")
            if result.stderr:
                print(f"  STDERR: {result.stderr[-500:]}")
            return (start_idx, end_idx, temp_csv, False, elapsed)

    except Exception as e:
        print(f"[Range {start_idx}-{end_idx}] ERROR: {e}")
        return (start_idx, end_idx, temp_csv, False, 0)


def merge_and_validate(output_path, temp_csvs, model_type, noise_level, data_type):
    """Merge existing output with new temp CSVs, sort, dedup, validate."""
    # Read existing file
    existing_rows = []
    existing_fieldnames = []
    if os.path.exists(output_path):
        with open(output_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            existing_fieldnames = list(reader.fieldnames)
            existing_rows = list(reader)
        print(f"Loaded {len(existing_rows)} rows from existing output file")
    else:
        print(f"No existing output file found at {output_path}")

    # Read all temp CSVs
    new_rows = []
    for temp_csv in temp_csvs:
        if os.path.exists(temp_csv):
            with open(temp_csv, 'r', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                new_rows.extend(rows)
                print(f"  Loaded {len(rows)} rows from {os.path.basename(temp_csv)}")

    print(f"Loaded {len(new_rows)} new rows from temp files")

    # Combine
    all_rows = existing_rows + new_rows

    # Deduplicate by index (keep first occurrence)
    seen = set()
    deduped = []
    dup_count = 0
    for row in all_rows:
        idx = int(row['index'])
        if idx not in seen:
            seen.add(idx)
            deduped.append(row)
        else:
            dup_count += 1

    print(f"Duplicates removed: {dup_count}")

    # Sort by index
    deduped.sort(key=lambda x: int(x['index']))

    # Validate continuity
    indices = [int(r['index']) for r in deduped]
    min_idx = min(indices)
    max_idx = max(indices)
    expected = max_idx - min_idx + 1

    print(f"\nTotal rows: {len(deduped)}")
    print(f"Index range: {min_idx} to {max_idx}")

    if len(indices) != expected:
        missing = set(range(min_idx, max_idx + 1)) - set(indices)
        print(f"WARNING: Still missing {len(missing)} indices")
        if missing:
            print(f"  Missing (first 20): {sorted(missing)[:20]}")
    else:
        print("Index continuity verified: all indices present")

    # Write output
    if existing_fieldnames:
        fieldnames = existing_fieldnames
    elif new_rows:
        fieldnames = list(new_rows[0].keys())
    else:
        fieldnames = []

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deduped)

    print(f"\nFinal output saved to: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Fix missing indices in batch output')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--noise-level', type=str, required=True, choices=['None', 'low', 'high'])
    parser.add_argument('--data-type', type=str, required=True, choices=['train', 'val'])
    parser.add_argument('--model-type', type=str, required=True,
                        choices=['fpfh', 'hybridpoint', 'pointreggpt', 'geotransformer', 'regtr', 'icp', 'soft'])
    parser.add_argument('--missing-indices-file', type=str, required=True,
                        help='File with comma-separated missing indices')
    parser.add_argument('--num-workers', type=int, default=8, help='Parallel workers')
    parser.add_argument('--max-range-size', type=int, default=100, help='Max indices per range')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: outputFiles/{model_type})')
    parser.add_argument('--script-path', type=str, default=None,
                        help='Path to testing script (auto-detected if not provided)')
    parser.add_argument('--max-retries', type=int, default=2, help='Max retries per failed range')
    # SOFT-specific parameters
    parser.add_argument('--soft-N', type=int, default=128,
                        help='SOFT voxel grid dimension (default: 128)')
    parser.add_argument('--soft-use-clahe', type=int, default=0,
                        help='SOFT use CLAHE 0/1 (default: 0)')
    parser.add_argument('--soft-r-min', type=int, default=16,
                        help='SOFT minimum radius (default: 16)')
    parser.add_argument('--soft-r-max', type=int, default=48,
                        help='SOFT maximum radius (default: 48)')
    parser.add_argument('--soft-level-rotation', type=float, default=0.001,
                        help='SOFT rotation potential level (default: 0.001)')
    parser.add_argument('--soft-level-translation', type=float, default=0.001,
                        help='SOFT translation potential level (default: 0.001)')
    parser.add_argument('--soft-normalization', type=int, default=2,
                        help='SOFT normalization factor (default: 2)')

    args = parser.parse_args()

    # Auto-detect script path
    if args.script_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_map = {
            'fpfh': 'testingFPFHOnPredatorData.py',
            'hybridpoint': 'testingHybridPointOnPredatorData.py',
            'pointreggpt': 'testingPointRegGPTOnPredatorData.py',
            'geotransformer': 'testingGeoTransformerOnPredatorData.py',
            'regtr': 'testingRegTROnPredatorData.py',
            'icp': 'testingICPOnPredatorData.py',
            'soft': 'testingSoftOnPredatorData.py'
        }
        args.script_path = os.path.join(script_dir, '..', script_map[args.model_type])

    if not os.path.exists(args.script_path):
        print(f"ERROR: Script not found: {args.script_path}")
        sys.exit(1)

    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.join('outputFiles', args.model_type)

    # Output file path
    output_path = os.path.join(args.output_dir, f'outfile_{args.model_type}_{args.noise_level}_{args.data_type}.csv')

    # Parse missing indices
    print("=" * 60)
    print("Fix Missing Indices")
    print("=" * 60)
    missing = parse_missing_indices(args.missing_indices_file)
    print(f"Missing indices: {len(missing)}")
    if missing:
        print(f"  Range: {missing[0]} to {missing[-1]}")

    ranges = group_into_ranges(missing, args.max_range_size)
    print(f"Grouped into {len(ranges)} ranges")
    for start, end in ranges:
        print(f"  {start:05d}-{end:05d} ({end - start + 1} samples)")

    # Process ranges in waves with retry
    pending = list(ranges)
    successful_files = []
    retry_counts = {}

    wave = 0
    while pending:
        wave += 1
        print(f"\n{'=' * 60}")
        print(f"WAVE {wave}: Processing {len(pending)} ranges")
        print(f"{'=' * 60}")

        soft_params = {
            'N': args.soft_N,
            'use_clahe': args.soft_use_clahe,
            'r_min': args.soft_r_min,
            'r_max': args.soft_r_max,
            'level_rotation': args.soft_level_rotation,
            'level_translation': args.soft_level_translation,
            'normalization': args.soft_normalization
        }

        batch_args = [
            (start, end, args.config, args.noise_level, args.data_type,
             args.script_path, args.output_dir, args.model_type, soft_params)
            for start, end in pending
        ]

        from multiprocessing import Pool
        with Pool(args.num_workers) as pool:
            results = pool.map(process_range, batch_args)

        results.sort(key=lambda x: x[0])

        next_pending = []
        for start, end, temp_csv, success, elapsed in results:
            if success:
                successful_files.append(temp_csv)
            else:
                retry_key = (start, end)
                retry_counts[retry_key] = retry_counts.get(retry_key, 0) + 1
                if retry_counts[retry_key] < args.max_retries:
                    next_pending.append((start, end))
                    print(f"  [RETRY {retry_counts[retry_key]}/{args.max_retries}] Range {start}-{end}")
                else:
                    print(f"  [FAILED] Range {start}-{end} (max retries exceeded)")

        pending = next_pending

    # Summary
    print(f"\n{'=' * 60}")
    print("PROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Successful ranges: {len(successful_files)}")
    print(f"Failed ranges: {len(missing) - len(successful_files)}")

    if not successful_files:
        print("\nNo ranges were successfully processed. Nothing to merge.")
        sys.exit(1)

    # Merge
    print(f"\nMerging results...")
    merge_and_validate(output_path, successful_files, args.model_type, args.noise_level, args.data_type)

    # Cleanup temp files
    temp_dir = os.path.join(args.output_dir, 'fix_missing_tmp')
    if os.path.exists(temp_dir):
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)
        print(f"\nCleaned up temp directory: {temp_dir}")

    print("\nDone!")


if __name__ == '__main__':
    main()
