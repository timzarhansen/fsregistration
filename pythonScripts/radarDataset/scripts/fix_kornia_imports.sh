#!/usr/bin/env bash
# Fix kornia imports in cloned LoFTR/EfficientLoFTR repos
# kornia >= 0.6 no longer has kornia.utils.grid
# Use kornia.utils.create_meshgrid (deprecated re-export, works in all versions >= 0.4.1)
set -e

BASE_DIR="${1:-$(dirname "$0")/../otherMethods}"

find "$BASE_DIR" -name '*.py' -exec sed -i 's/from kornia\.utils\.grid import create_meshgrid/from kornia.utils import create_meshgrid/g' {} +
