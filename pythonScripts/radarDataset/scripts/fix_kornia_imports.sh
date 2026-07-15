#!/usr/bin/env bash
# Fix imports in cloned LoFTR/EfficientLoFTR repos for compatibility with kornia >= 0.6
set -e

BASE_DIR="${1:-$(dirname "$0")/../otherMethods}"

# 1. kornia >= 0.6 no longer has kornia.utils.grid
find "$BASE_DIR" -name '*.py' -exec sed -i \
  's/from kornia\.utils\.grid import create_meshgrid/from kornia.utils import create_meshgrid/g' {} +

# 2. kornia >= 0.8.3 deprecated kornia.utils.create_meshgrid in favor of kornia.geometry
find "$BASE_DIR" -name '*.py' -exec sed -i \
  's/from kornia\.utils import create_meshgrid/from kornia.geometry import create_meshgrid/g' {} +

# 3. pytorch_lightning is not installed during inference — wrap import in try/except
for f in "$BASE_DIR/LoFTR/src/utils/misc.py" "$BASE_DIR/EfficientLoFTR/src/utils/misc.py"; do
  if [ -f "$f" ] && grep -q "^from pytorch_lightning\.utilities import rank_zero_only$" "$f"; then
    sed -i \
      's/^from pytorch_lightning\.utilities import rank_zero_only$/try:\n    from pytorch_lightning.utilities import rank_zero_only\nexcept ImportError:\n    rank_zero_only = None/g' "$f"
  fi
done
