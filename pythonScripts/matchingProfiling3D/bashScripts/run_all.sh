#!/bin/bash
#
# Run FPFH registration batch processing
#
# Usage:
#   ./run_all.sh [config] [noise-level] [data-type] [num-workers]
#
# Arguments:
#   config      - Path to config file (default: configFiles/predatorNothingMac.yaml)
#   noise-level - Noise level: None, low, high (default: high)
#   data-type   - Dataset type: train, val (default: train)
#   num-workers - Number of parallel processes (default: 4)
#

set -e

# Defaults
CONFIG="${1:-configFiles/predatorNothingMac.yaml}"
NOISE_LEVEL="${2:-high}"
DATA_TYPE="${3:-train}"
NUM_WORKERS="${4:-4}"
TOTAL_SAMPLES=20000
BATCH_SIZE=100

echo "=============================================="
echo "FPFH Registration Batch Processing"
echo "=============================================="
echo "Config: $CONFIG"
echo "Noise level: $NOISE_LEVEL"
echo "Data type: $DATA_TYPE"
echo "Total samples: $TOTAL_SAMPLES"
echo "Batch size: $BATCH_SIZE"
echo "Parallel workers: $NUM_WORKERS"
echo "=============================================="

# Check config exists
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

# Create output directory
mkdir -p outputFiles

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "Starting parallel batch processing..."
echo ""

# Run parallel batches
python3 "$SCRIPT_DIR/run_parallel_batches.py" \
    --config "$CONFIG" \
    --noise-level "$NOISE_LEVEL" \
    --data-type "$DATA_TYPE" \
    --total-samples "$TOTAL_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS"

PARALLEL_EXIT=$?

echo ""
echo "=============================================="
echo "Merging Results"
echo "=============================================="

# Merge results
python3 "$SCRIPT_DIR/merge_and_deduplicate.py" \
    --noise-level "$NOISE_LEVEL" \
    --data-type "$DATA_TYPE"

MERGE_EXIT=$?

echo ""
echo "=============================================="
echo "Processing Complete"
echo "=============================================="

if [ $MERGE_EXIT -eq 0 ]; then
    OUTPUT_FILE="outputFiles/outfile_fpfh_${NOISE_LEVEL}_${DATA_TYPE}.csv"
    if [ -f "$OUTPUT_FILE" ]; then
        LINES=$(wc -l < "$OUTPUT_FILE")
        echo "Final output: $OUTPUT_FILE"
        echo "Total samples: $((LINES - 1))"  # Subtract header
    fi
fi

# Exit with appropriate code
if [ $PARALLEL_EXIT -ne 0 ]; then
    echo "WARNING: Some batches failed during processing"
fi

exit $MERGE_EXIT
