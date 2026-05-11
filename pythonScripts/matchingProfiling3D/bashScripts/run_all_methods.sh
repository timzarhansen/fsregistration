#!/usr/bin/env bash

# Master orchestration script to run multiple registration methods
# Usage: ./run_all_methods.sh [methods...]
# Example: ./run_all_methods.sh fpfh icp regtr
#          ./run_all_methods.sh (runs all methods)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Available methods
ALL_METHODS="fpfh icp geotransformer regtr hybridpoint pointreggpt"

# Parse methods from arguments or use all
if [ $# -gt 0 ]; then
    METHODS="$@"
else
    METHODS="$ALL_METHODS"
fi

echo "=============================================="
echo "FSRegistration Batch Processing"
echo "=============================================="
echo "Methods to run: $METHODS"
echo "=============================================="

# Track PIDs for background jobs
declare -a PIDS

# Function to run a method
run_method() {
    local method=$1
    case $method in
        fpfh)
            echo "[START] FPFH"
            bash "$SCRIPT_DIR/runFPFH.sh" &
            PIDS+=($!)
            ;;
        icp)
            echo "[START] ICP"
            bash "$SCRIPT_DIR/runICP_batch.sh" &
            PIDS+=($!)
            ;;
        geotransformer)
            echo "[START] GeoTransformer"
            bash "$SCRIPT_DIR/runGeoTransformer_batch.sh" &
            PIDS+=($!)
            ;;
        regtr)
            echo "[START] RegTR"
            bash "$SCRIPT_DIR/runRegTR_batch.sh" &
            PIDS+=($!)
            ;;
        hybridpoint)
            echo "[START] HybridPoint"
            bash "$SCRIPT_DIR/runHybridPoint_batch.sh" &
            PIDS+=($!)
            ;;
        pointreggpt)
            echo "[START] PointRegGPT"
            bash "$SCRIPT_DIR/runPointRegGPT_batch.sh" &
            PIDS+=($!)
            ;;
        *)
            echo "[ERROR] Unknown method: $method"
            echo "Available methods: $ALL_METHODS"
            exit 1
            ;;
    esac
}

# Run all specified methods
for method in $METHODS; do
    run_method "$method"
done

echo ""
echo "All methods started in background."
echo "Waiting for completion..."
echo ""

# Wait for all background jobs
for pid in "${PIDS[@]}"; do
    wait "$pid" || true
done

echo ""
echo "=============================================="
echo "All methods completed!"
echo "=============================================="
echo ""
echo "Output files location:"
echo "  outputFiles/fpfh/outfile_*.csv"
echo "  outputFiles/icp/outfile_*.csv"
echo "  outputFiles/geotransformer/outfile_*.csv"
echo "  outputFiles/regtr/outfile_*.csv"
echo "  outputFiles/hybridpoint/outfile_*.csv"
echo "  outputFiles/pointreggpt/outfile_*.csv"
