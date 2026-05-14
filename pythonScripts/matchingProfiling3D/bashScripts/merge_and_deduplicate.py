#!/usr/bin/env python3
"""
Merge and deduplicate batch CSV files into a single final output.
Handles duplicate indices and ensures proper sorting.

Usage:
    python merge_and_deduplicate.py --output-dir outputFiles --noise-level high --data-type train
"""

import argparse
import csv
import os
import glob
import sys


def merge_batches(output_dir='outputFiles', noise_level='high', data_type='train', model_type='fpfh'):
    """Merge all batch CSV files into a single deduplicated output."""
    
    # Use method-specific subdirectory if not explicitly set
    if output_dir == 'outputFiles':
        output_dir = os.path.join(output_dir, model_type)
    
    # Find all batch files (exclude final output and retry logs)
    batch_pattern = os.path.join(output_dir, f'batch_{model_type}_{noise_level}_{data_type}_*.csv')
    batch_files = glob.glob(batch_pattern)
    
    if not batch_files:
        print("ERROR: No batch files found!")
        print(f"Searched for: {batch_pattern}")
        return False
    
    print(f"Found {len(batch_files)} batch files")
    
    # Read all batches
    all_rows = []
    row_counts = []
    
    for batch_file in sorted(batch_files):
        try:
            with open(batch_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                all_rows.extend(rows)
                row_counts.append(len(rows))
                print(f"  Loaded {len(rows)} samples from {os.path.basename(batch_file)}")
        except Exception as e:
            print(f"  ERROR reading {batch_file}: {e}")
    
    if not all_rows:
        print("ERROR: No data loaded from batch files!")
        return False
    
    print(f"\nTotal rows before dedup: {len(all_rows)}")
    
    # Deduplicate by index (keep first occurrence)
    seen_indices = set()
    deduped_rows = []
    duplicate_count = 0
    
    for row in all_rows:
        idx = int(row['index'])
        if idx not in seen_indices:
            seen_indices.add(idx)
            deduped_rows.append(row)
        else:
            duplicate_count += 1
    
    print(f"Duplicates removed: {duplicate_count}")
    print(f"Total rows after dedup: {len(deduped_rows)}")
    
    # Sort by index
    deduped_rows.sort(key=lambda x: int(x['index']))
    
    # Verify continuity
    indices = [int(row['index']) for row in deduped_rows]
    min_idx = min(indices)
    max_idx = max(indices)
    expected_count = max_idx - min_idx + 1
    
    if len(indices) != expected_count:
        missing = set(range(min_idx, max_idx + 1)) - set(indices)
        print(f"WARNING: Missing {len(missing)} indices in range [{min_idx}, {max_idx}]")
        if missing:
            print(f"  Missing indices (first 10): {sorted(missing)[:10]}")
    else:
        print(f"Index continuity verified: {min_idx} to {max_idx}")
    
    # Save final output
    output_path = os.path.join(output_dir, f'outfile_{model_type}_{noise_level}_{data_type}.csv')

    # Dynamic fieldnames from first batch file
    with open(sorted(batch_files)[0], 'r', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deduped_rows)
    
    print(f"\nFinal output saved to: {output_path}")
    print(f"Total samples: {len(deduped_rows)}")
    
    # Clean up batch files
    print(f"\nCleaning up batch files...")
    for batch_file in batch_files:
        os.remove(batch_file)
        print(f"  Removed: {os.path.basename(batch_file)}")
    
    # Clean up retry log
    retry_log = os.path.join(output_dir, f'retry_log_{noise_level}_{data_type}.txt')
    if os.path.exists(retry_log):
        os.remove(retry_log)
        print(f"  Removed: retry_log_{noise_level}_{data_type}.txt")
    
    print(f"Cleaned up {len(batch_files)} batch files")
    
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge batch CSV files')
    parser.add_argument('--output-dir', type=str, default='outputFiles',
                        help='Directory containing batch files')
    parser.add_argument('--noise-level', type=str, default='high',
                        choices=['None', 'low', 'high'],
                        help='Noise level (must match batch naming)')
    parser.add_argument('--data-type', type=str, default='train',
                        choices=['train', 'val'],
                        help='Dataset type')
    parser.add_argument('--model-type', type=str, default='fpfh',
                        choices=['fpfh', 'hybridpoint', 'pointreggpt', 'geotransformer', 'regtr', 'icp', 'soft'],
                        help='Model type for output filename')
    
    args = parser.parse_args()
    
    success = merge_batches(args.output_dir, args.noise_level, args.data_type, args.model_type)
    sys.exit(0 if success else 1)
