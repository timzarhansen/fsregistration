#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="/home/tim-external/ros_ws/src/fsregistration/pythonScripts/radarDataset"
ENV_NAME="ml"  # Change to your CUDA/ML conda environment name if needed

cd "$PROJECT_DIR"

# ---- LOAD CONDA (uncomment if using conda) ----
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate "$ENV_NAME"

echo "Starting boreas benchmark..."
echo "Data dir: /home/tim-external/dataFolder/radar_boreas"
echo ""

python3 boreasBenchmark.py \
    --method fs2d \
    --sequence 0 \
    --sequence-name "boreas-2020-11-26-13-58" \
    --N 128 \
    --size_of_pixel 0.5 \
    --max_frames 100 \
    --output-dir benchmark_results \
    /home/tim-external/dataFolder/radar_boreas

echo ""
echo "Benchmark complete."
