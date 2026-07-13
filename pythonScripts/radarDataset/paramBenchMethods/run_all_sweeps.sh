#!/bin/bash
# Run all parameter sweep benchmarks in parallel.
# Each sweep outputs results to the configured OUTPUT_DIR (default: benchmark_sweep/).
# Press Ctrl+C to stop all sweeps.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

trap 'echo ""; echo "Interrupted! Killing all background processes..."; kill -INT 0; exit 1' SIGINT

echo "============================================"
echo "  Running all parameter sweep benchmarks"
echo "============================================"

declare -A SWEEPS
SWEEPS["ICP"]="boreasBenchmarkICPSweep.py"
SWEEPS["NDT P2D"]="boreasBenchmarkNDTSweep.py"
SWEEPS["LoFTR"]="boreasBenchmarkLoFTRSweep.py"
SWEEPS["EfficientLoFTR"]="boreasBenchmarkEfficientLoFTRSweep.py"
SWEEPS["LightGlue"]="boreasBenchmarkLightGlueSweep.py"
SWEEPS["Fourier-Mellin"]="boreasBenchmarkFourierMellinSweep.py"
SWEEPS["SIFT"]="boreasBenchmarkSIFTSweep.py"
# SWEEPS["SURF"]="boreasBenchmarkSURFSweep.py"
SWEEPS["KAZE"]="boreasBenchmarkKAZESweep.py"
SWEEPS["AKAZE"]="boreasBenchmarkAKAZESweep.py"
SWEEPS["FS2D"]="boreasBenchmarkFS2DSweep.py"

PIDS=()

for name in "${!SWEEPS[@]}"; do
    script="${SWEEPS[$name]}"
    echo "  Starting: $name ($script) in background"
    (
        echo "  [PID $$] $name started"
        python "$script"
        status=$?
        if [ $status -eq 0 ]; then
            echo "  [PID $$] $name finished (OK)"
        else
            echo "  [PID $$] $name FAILED (exit code $status)"
        fi
        exit $status
    ) &
    PIDS+=($!)
done

echo ""
echo "Waiting for all sweeps to complete..."
echo "Press Ctrl+C to stop all running sweeps."
echo ""

FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAILED=$((FAILED + 1))
done

echo ""
echo "============================================"
if [ "$FAILED" -eq 0 ]; then
    echo "  All sweeps completed successfully"
else
    echo "  $FAILED sweep(s) failed"
fi
echo "============================================"
