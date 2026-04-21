#!/bin/bash
# Setup script for GPU backend testing
# Usage: source scripts/setup_gpu_backend.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/cmake-build-debug"

# Add build directory to PYTHONPATH so softCorrelation_gpu_backend.py can be found
export PYTHONPATH="${BUILD_DIR}:${PYTHONPATH}"

echo "GPU Backend Setup Complete"
echo "PYTHONPATH: $PYTHONPATH"
echo ""
echo "To run tests:"
echo "  cd ${BUILD_DIR}"
echo "  ./test_correlation_comparison"
echo ""
echo "Or run directly:"
echo "  ${BUILD_DIR}/test_correlation_comparison"
