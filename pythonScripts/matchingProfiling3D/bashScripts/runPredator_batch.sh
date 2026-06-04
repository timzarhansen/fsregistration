#!/usr/bin/env bash

set -euo pipefail

# ---- CONFIG ----
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="predator_env"
CONFIG="configFiles/predatorNothing.yaml"
NUM_WORKERS=8
BATCH_SIZE=100
TOTAL_SAMPLES_VAL=1331
TOTAL_SAMPLES_TRAIN=20642

cd "$PROJECT_DIR"

# ---- LOAD CONDA ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "Starting Predator batch processing..."
echo "=============================================="
echo "Config: $CONFIG"
echo "Num workers: $NUM_WORKERS"
echo "Batch size: $BATCH_SIZE"
echo "=============================================="

# Process all combinations
for noise_level in None low high; do
    for data_type in val train; do
        echo ""
        echo "=============================================="
        echo "Processing: $noise_level / $data_type"
        echo "=============================================="
        
        if [ "$data_type" = "val" ]; then
            TOTAL_SAMPLES=$TOTAL_SAMPLES_VAL
        else
            TOTAL_SAMPLES=$TOTAL_SAMPLES_TRAIN
        fi

        OUTPUT_FILE="outputFiles/predator/outfile_predator_${noise_level}_${data_type}.csv"
        if [ -f "$OUTPUT_FILE" ]; then
            ACTUAL_ROWS=$(($(wc -l < "$OUTPUT_FILE") - 1))
            if [ "$ACTUAL_ROWS" -eq "$TOTAL_SAMPLES" ]; then
                echo "SKIP: $noise_level / $data_type (already complete: $ACTUAL_ROWS rows)"
                continue
            fi
        fi
        
        python3 bashScripts/run_parallel_batches.py \
            --config "$CONFIG" \
            --noise-level "$noise_level" \
            --data-type "$data_type" \
            --total-samples "$TOTAL_SAMPLES" \
            --batch-size "$BATCH_SIZE" \
            --num-workers "$NUM_WORKERS" \
            --model-type predator
        
        python3 bashScripts/merge_and_deduplicate.py \
            --noise-level "$noise_level" \
            --data-type "$data_type" \
            --model-type predator
    done
done

echo ""
echo "=============================================="
echo "Predator batch processing complete!"
echo "=============================================="
