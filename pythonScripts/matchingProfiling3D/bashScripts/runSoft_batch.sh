#!/usr/bin/env bash

set -euo pipefail

# ---- CONFIG ----
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="ml"
CONFIG="configFiles/predatorNothingBenchmark.yaml"
NUM_WORKERS=8
BATCH_SIZE=100
TOTAL_SAMPLES_VAL=1331
TOTAL_SAMPLES_TRAIN=20642
# SOFT parameters
SOFT_N=32
SOFT_USE_CLAHE=0
SOFT_R_MIN=4
SOFT_R_MAX=28
SOFT_LEVEL_ROTATION=0.001
SOFT_LEVEL_TRANSLATION=0.001
SOFT_NORMALIZATION=2

cd "$PROJECT_DIR"

echo "Starting SOFT batch processing..."
echo "=============================================="
echo "Config: $CONFIG"
echo "Num workers: $NUM_WORKERS"
echo "Batch size: $BATCH_SIZE"
echo "SOFT N: $SOFT_N, r_min: $SOFT_R_MIN, r_max: $SOFT_R_MAX"
echo "SOFT level_rotation: $SOFT_LEVEL_ROTATION, level_translation: $SOFT_LEVEL_TRANSLATION"
echo "SOFT normalization: $SOFT_NORMALIZATION"
echo "=============================================="

# Process all combinations
for noise_level in low_gauss high_gauss low_salt_pepper high_salt_pepper None low high; do
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
        
        python3 bashScripts/run_parallel_batches.py \
            --config "$CONFIG" \
            --noise-level "$noise_level" \
            --data-type "$data_type" \
            --total-samples "$TOTAL_SAMPLES" \
            --batch-size "$BATCH_SIZE" \
            --num-workers "$NUM_WORKERS" \
            --model-type soft \
            --soft-N "$SOFT_N" \
            --soft-use-clahe "$SOFT_USE_CLAHE" \
            --soft-r-min "$SOFT_R_MIN" \
            --soft-r-max "$SOFT_R_MAX" \
            --soft-level-rotation "$SOFT_LEVEL_ROTATION" \
            --soft-level-translation "$SOFT_LEVEL_TRANSLATION" \
            --soft-normalization "$SOFT_NORMALIZATION"
        
        python3 bashScripts/merge_and_deduplicate.py \
            --noise-level "$noise_level" \
            --data-type "$data_type" \
            --model-type soft \
            --soft-N "$SOFT_N"
    done
done

echo ""
echo "=============================================="
echo "SOFT batch processing complete!"
echo "=============================================="
