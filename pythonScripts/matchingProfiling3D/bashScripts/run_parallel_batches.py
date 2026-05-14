#!/usr/bin/env python3
"""
Parallel batch processor for FPFH registration testing.
Splits samples into batches and processes them in parallel.

Usage:
    python run_parallel_batches.py --config config.yaml --noise-level high --data-type train \
                                   --total-samples 20000 --batch-size 100 --num-workers 4
"""

import argparse
import os
import sys
import glob
import subprocess
import time
from pathlib import Path


def get_completed_batches(output_dir, batch_size, noise_level, data_type, model_type):
    """Find already completed batch files."""
    pattern = os.path.join(output_dir, f'batch_{model_type}_{noise_level}_{data_type}_*.csv')
    completed = set()
    
    for f in glob.glob(pattern):
        # Extract start index from filename: batch_fpfh_None_train_00000_00099.csv
        basename = os.path.basename(f)
        try:
            parts = basename.replace('.csv', '').split('_')
            # Format: batch_{model}_{noise}_{type}_{start}_{end}
            if len(parts) >= 6:
                start_idx = int(parts[4])
                end_idx = int(parts[5])
                # Only count if start index is aligned to batch_size
                if start_idx % batch_size == 0:
                    # Validate row count to detect partial/interrupted files
                    expected_rows = end_idx - start_idx + 1
                    with open(f, 'r') as csvf:
                        actual_rows = sum(1 for _ in csvf) - 1  # minus header
                    if actual_rows == expected_rows:
                        completed.add(start_idx)
                    else:
                        print(f"  [PARTIAL] {basename}: {actual_rows}/{expected_rows} rows - will reprocess")
        except (IndexError, ValueError):
            continue
    
    return completed


def create_batches(total_samples, batch_size):
    """Create list of batch ranges."""
    batches = []
    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size - 1, total_samples - 1)
        batches.append((start, end))
    return batches


def process_batch(args):
    """Process a single batch."""
    batch_id, start_idx, end_idx, config, noise_level, data_type, script_path, model_type, output_dir, soft_params = args

    output_file = os.path.join(output_dir, f'batch_{model_type}_{noise_level}_{data_type}_{start_idx:05d}_{end_idx:05d}.csv')

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
            '--output-file', output_file
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
            '--output-file', output_file
        ]
    
    print(f"[Batch {batch_id}] Processing samples {start_idx}-{end_idx}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"[Batch {batch_id}] Completed in {elapsed:.1f}s")
            return (batch_id, start_idx, end_idx, True, elapsed)
        else:
            print(f"[Batch {batch_id}] FAILED after {elapsed:.1f}s")
            if result.stderr:
                print(f"  STDERR: {result.stderr[-500:]}")  # Last 500 chars
            return (batch_id, start_idx, end_idx, False, elapsed)
    
    except Exception as e:
        print(f"[Batch {batch_id}] ERROR: {e}")
        return (batch_id, start_idx, end_idx, False, 0)


def main():
    parser = argparse.ArgumentParser(description='Parallel batch processor')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--noise-level', type=str, required=True,
                        choices=['None', 'low', 'high'],
                        help='Noise level')
    parser.add_argument('--data-type', type=str, required=True,
                        choices=['train', 'val'],
                        help='Dataset type')
    parser.add_argument('--total-samples', type=int, default=20000,
                        help='Total number of samples')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Samples per batch')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of parallel processes')
    parser.add_argument('--max-retries', type=int, default=3,
                          help='Max retries per failed batch')
    parser.add_argument('--model-type', type=str, default='fpfh',
                        choices=['fpfh', 'hybridpoint', 'pointreggpt', 'geotransformer', 'regtr', 'icp', 'soft'],
                        help='Model type (auto-selects script)')
    parser.add_argument('--script-path', type=str, default=None,
                        help='Path to testing script (auto-detected if not provided)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: outputFiles/{model_type})')
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
    
    # Auto-detect script path based on model type
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
    
    # Create output directory with method-specific subdirectory
    if args.output_dir is None:
        args.output_dir = os.path.join('outputFiles', args.model_type)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create batches
    all_batches = create_batches(args.total_samples, args.batch_size)
    print(f"\nTotal batches: {len(all_batches)}")
    print(f"Batch size: {args.batch_size} samples")
    print(f"Parallel workers: {args.num_workers}")
    
    # Check for completed batches
    completed = get_completed_batches(args.output_dir, args.batch_size, args.noise_level, args.data_type, args.model_type)
    if completed:
        print(f"\nFound {len(completed)} already completed batches")
        print(f"  Start indices: {sorted(completed)[:10]}{'...' if len(completed) > 10 else ''}")
    
    # Filter out completed batches
    pending_batches = [(i, start, end) for i, (start, end) in enumerate(all_batches) 
                       if start not in completed]
    
    if not pending_batches:
        print("\nAll batches already completed!")
        print("Run merge_and_deduplicate.py to combine results.")
        sys.exit(0)
    
    print(f"Pending batches: {len(pending_batches)}")
    
    # Process batches in waves
    successful_batches = []
    failed_batches = []
    retry_counts = {}  # Track retry count per batch
    
    wave = 0
    while pending_batches:
        wave += 1
        print(f"\n{'='*60}")
        print(f"WAVE {wave}: Processing {len(pending_batches)} batches")
        print(f"{'='*60}")
        
        # Prepare SOFT-specific parameters
        soft_params = {
            'N': args.soft_N,
            'use_clahe': args.soft_use_clahe,
            'r_min': args.soft_r_min,
            'r_max': args.soft_r_max,
            'level_rotation': args.soft_level_rotation,
            'level_translation': args.soft_level_translation,
            'normalization': args.soft_normalization
        }

        # Prepare batch arguments
        batch_args = [
            (batch_id, start, end, args.config, args.noise_level, args.data_type, args.script_path, args.model_type, args.output_dir, soft_params)
            for batch_id, start, end in pending_batches
        ]
        
        # Process in parallel
        from multiprocessing import Pool
        
        with Pool(args.num_workers) as pool:
            results = pool.map(process_batch, batch_args)
        
        # Sort results for consistent ordering
        results.sort(key=lambda x: x[0])
        
        # Separate successful and failed
        wave_pending = []
        for batch_id, start, end, success, elapsed in results:
            if success:
                successful_batches.append((batch_id, start, end))
            else:
                # Track retries
                retry_key = (start, end)
                retry_counts[retry_key] = retry_counts.get(retry_key, 0) + 1
                
                if retry_counts[retry_key] < args.max_retries:
                    wave_pending.append((batch_id, start, end))
                    print(f"  [RETRY {retry_counts[retry_key]}/{args.max_retries}] Batch {batch_id}: {start}-{end}")
                else:
                    failed_batches.append((batch_id, start, end))
                    print(f"  [FAILED] Batch {batch_id}: {start}-{end} (max retries exceeded)")
        
        pending_batches = wave_pending
    
    # Summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successful batches: {len(successful_batches)}")
    print(f"Failed batches: {len(failed_batches)}")
    
    if failed_batches:
        print(f"\nFailed batch ranges:")
        for batch_id, start, end in failed_batches[:10]:
            print(f"  Batch {batch_id}: {start}-{end}")
        if len(failed_batches) > 10:
            print(f"  ... and {len(failed_batches) - 10} more")
        print("\nYou can manually retry failed batches or adjust parameters.")
    
    # Check if we have enough data to merge
    total_processed = len(successful_batches) * args.batch_size
    print(f"\nTotal samples processed: ~{total_processed}")
    
    if successful_batches:
        print(f"\nRun the following to merge results:")
        print(f"  python3 merge_and_deduplicate.py --noise-level {args.noise_level} --data-type {args.data_type} --model-type {args.model_type}")
    
    # Exit with error if any batches failed
    sys.exit(1 if failed_batches else 0)


if __name__ == '__main__':
    main()
