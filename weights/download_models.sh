#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="$SCRIPT_DIR"
TEMP_DIR=$(mktemp -d)

cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

echo "========================================="
echo "  Downloading 3D Registration Model Weights"
echo "========================================="
echo ""

# --- GeoTransformer ---
echo "[1/4] Downloading GeoTransformer..."
GEO_URL="https://github.com/qinzheng93/GeoTransformer/releases/latest/download/geotransformer-3dmatch.pth.tar"
GEO_TARGET="$WEIGHTS_DIR/geotransformer/geotransformer-3dmatch.pth"

if [ -f "$GEO_TARGET" ]; then
    echo "  Already exists: $GEO_TARGET (skipping)"
else
    mkdir -p "$WEIGHTS_DIR/geotransformer"
    wget --show-progress -q -O "$TEMP_DIR/geo.tar" "$GEO_URL"
    mv "$TEMP_DIR/geo.tar" "$GEO_TARGET"
    echo "  Downloaded: $GEO_TARGET"
fi
echo ""

# --- RegTR ---
echo "[2/4] Downloading RegTR..."
REGTR_URL="https://github.com/yewzijian/RegTR/releases/download/v1/trained_models.zip"
REGTR_TARGET="$WEIGHTS_DIR/regtr/regtr-3dmatch.pth"

if [ -f "$REGTR_TARGET" ]; then
    echo "  Already exists: $REGTR_TARGET (skipping)"
else
    mkdir -p "$WEIGHTS_DIR/regtr"
    wget --show-progress -q -O "$TEMP_DIR/regtr.zip" "$REGTR_URL"
    unzip -o -q "$TEMP_DIR/regtr.zip" -d "$TEMP_DIR/regtr_extracted"
    # Find the best model file in the 3dmatch checkpoint directory
    BEST_MODEL=$(find "$TEMP_DIR/regtr_extracted" -path "*/3dmatch/*" -name "model-*.pth" | head -1)
    if [ -n "$BEST_MODEL" ]; then
        mv "$BEST_MODEL" "$REGTR_TARGET"
        echo "  Downloaded: $REGTR_TARGET"
    else
        echo "  Warning: Could not find RegTR 3DMatch model in release"
    fi
fi
echo ""

# --- HybridPoint ---
echo "[3/4] Downloading HybridPoint..."
HYBRID_URL="https://raw.githubusercontent.com/liyih/HybridPoint/main/weights_for_hybrid/3dmatch.tar"
HYBRID_TARGET="$WEIGHTS_DIR/hybridpoint/hybridpoint-3dmatch.pth"

if [ -f "$HYBRID_TARGET" ]; then
    echo "  Already exists: $HYBRID_TARGET (skipping)"
else
    mkdir -p "$WEIGHTS_DIR/hybridpoint"
    wget --show-progress -q -O "$TEMP_DIR/hybrid.tar" "$HYBRID_URL"
    mv "$TEMP_DIR/hybrid.tar" "$HYBRID_TARGET"
    echo "  Downloaded: $HYBRID_TARGET"
fi
echo ""

# --- Predator ---
echo "[4/4] Downloading Predator..."
PREDATOR_URL="https://share.phys.ethz.ch/~gsg/Predator/weights.zip"
PREDATOR_TARGET="$WEIGHTS_DIR/predator/predator-indoor.pth"

if [ -f "$PREDATOR_TARGET" ]; then
    echo "  Already exists: $PREDATOR_TARGET (skipping)"
else
    mkdir -p "$WEIGHTS_DIR/predator"
    wget --no-check-certificate --show-progress -q -O "$TEMP_DIR/predator.zip" "$PREDATOR_URL"
    unzip -o -q "$TEMP_DIR/predator.zip" -d "$TEMP_DIR/predator_extracted"
    if [ -f "$TEMP_DIR/predator_extracted/weights/indoor.pth" ]; then
        mv "$TEMP_DIR/predator_extracted/weights/indoor.pth" "$PREDATOR_TARGET"
        echo "  Downloaded: $PREDATOR_TARGET"
    else
        echo "  Warning: Could not find Predator indoor weights in release"
    fi
fi
echo ""

echo "========================================="
echo "  Download Complete"
echo "========================================="

# Check what's available
echo ""
echo "Available weights:"
for method in geotransformer regtr predator hybridpoint; do
    if ls "$WEIGHTS_DIR/$method/"*.pth 1>/dev/null 2>&1; then
        echo "  [OK]   $method"
    else
        echo "  [MISS] $method"
    fi
done
