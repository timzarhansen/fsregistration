#!/bin/bash
# Quick test of batch processing with 2 batches (200 samples)

set -e

echo "=== Quick Test: Processing 2 batches (200 samples) ==="
echo ""

# Clean up any previous test
rm -f outputFiles/batch_None_train_*.csv
rm -f outputFiles/outfile_fpfh_test*.csv

# Run 2 batches (FPFH model)
python3 run_parallel_batches.py \
    --config configFiles/predatorNothingMac.yaml \
    --noise-level None \
    --data-type train \
    --total-samples 200 \
    --batch-size 100 \
    --num-workers 2

echo ""
echo "=== Merging test results ==="
python3 merge_and_deduplicate.py --noise-level None --data-type train

echo ""
echo "=== Test Complete ==="
echo "Check outputFiles/ for batch_None_train_*.csv and outfile_fpfh_None_train.csv"

echo ""
echo "=== Examples for other model types ==="
echo "# HybridPoint:"
echo "  python3 run_parallel_batches.py --model-type hybridpoint --noise-level None --data-type train --total-samples 200 --batch-size 100 --num-workers 2"
echo "  python3 merge_and_deduplicate.py --model-type hybridpoint --noise-level None --data-type train"
echo ""
echo "# PointRegGPT:"
echo "  python3 run_parallel_batches.py --model-type pointreggpt --noise-level None --data-type train --total-samples 200 --batch-size 100 --num-workers 2"
echo "  python3 merge_and_deduplicate.py --model-type pointreggpt --noise-level None --data-type train"
