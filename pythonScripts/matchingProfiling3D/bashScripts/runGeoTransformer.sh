#!/usr/bin/env bash

set -euo pipefail

# ---- CONFIG ----
PROJECT_DIR="/Users/timhansen/Documents/opencodeTestProject/fsregistration/pythonScripts/matchingProfiling3D"
ENV_NAME="geo_env"

cd "$PROJECT_DIR"

# ---- LOAD CONDA ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "Starting jobs..."

# -----------------------
# LOW
# -----------------------
echo "Running LOW..."

python3 testingFPFHOnPredatorData.py configFiles/predatorNothingMac.yaml low train > low_train.log 2>&1 &
pid1=$!

python3 testingFPFHOnPredatorData.py configFiles/predatorNothingMac.yaml low val > low_val.log 2>&1 &
pid2=$!

wait $pid1 $pid2
echo "LOW done"

# -----------------------
# HIGH
# -----------------------
echo "Running HIGH..."

python3 testingFPFHOnPredatorData.py configFiles/predatorNothingMac.yaml high train > high_train.log 2>&1 &
pid3=$!

python3 testingFPFHOnPredatorData.py configFiles/predatorNothingMac.yaml high val > high_val.log 2>&1 &
pid4=$!

wait $pid3 $pid4
echo "HIGH done"

 -----------------------
 NONE
 -----------------------
echo "Running NONE..."

python3 testingFPFHOnPredatorData.py configFiles/predatorNothingMac.yaml None train > none_train.log 2>&1 &
pid5=$!

python3 testingFPFHOnPredatorData.py configFiles/predatorNothingMac.yaml None val > none_val.log 2>&1 &
pid6=$!

wait $pid5 $pid6
#wait $pid5 $pid3
echo "NONE done"

echo "All jobs finished."