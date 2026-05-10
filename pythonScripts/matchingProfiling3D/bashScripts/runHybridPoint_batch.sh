#!/usr/bin/env bash

set -euo pipefail

# ---- CONFIG ----
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="pointreggpt_env"
CONFIG="configFiles/predatorNothingMac.yaml"
NUM_WORKERS=8
BATCH_SIZE=100
TOTAL_SAMPLES_VAL=1331
TOTAL_SAMPLES_TRAIN=20642

cd "$PROJECT_DIR"

# ---- LOAD CONDA ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "Starting HybridPoint batch processing..."
echo "=============================================="
echo "Config: $CONFIG"
echo "Num workers: $NUM_WORKERS"
echo "Batch size: $BATCH_SIZE"
echo "=============================================="

# Process all combinations
for noise_level in  high; do
    for data_type in  train; do
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
            --model-type hybridpoint
        
        python3 bashScripts/merge_and_deduplicate.py \
            --noise-level "$noise_level" \
            --data-type "$data_type" \
            --model-type hybridpoint
    done
done

echo ""
echo "=============================================="
echo "HybridPoint batch processing complete!"
echo "=============================================="


#        python3 bashScripts/run_parallel_batches.py \
#            --config configFiles/predatorNothingMac.yaml \
#            --noise-level None \
#            --data-type val \
#            --total-samples 1331 \
#            --batch-size 100 \
#            --num-workers 2 \
#            --model-type hybridpoint