# HybridPoint Wrapper

A self-contained wrapper for the HybridPoint point cloud registration method that doesn't modify the GeoTransformer library.

## Overview

This module provides a clean separation between HybridPoint and GeoTransformer, allowing both to be used independently without conflicts. It wraps the HybridPoint implementation for the 3DMatch dataset.

## Structure

```
hybridpoint_wrapper/
├── __init__.py          # Module exports
├── keypoints_detect.py  # ISS keypoint detection algorithm
├── data.py              # Data loading and collation functions
├── model.py             # HybridPoint model architecture
└── backbone.py          # KPConvFPN backbone network
```

## Usage

```python
import sys
sys.path.insert(0, 'path/to/hybridpoint_wrapper')

from hybridpoint_wrapper.model import create_model
from hybridpoint_wrapper.data import registration_collate_fn_stack_mode
```

## Dependencies

- GeoTransformer (for base modules and operations)
- PyTorch
- NumPy
- SciPy

## Original HybridPoint

This wrapper is based on the original HybridPoint project:
- Paper: [HybridPoint: Point Cloud Registration Based on Hybrid Point Sampling and Matching](https://arxiv.org/abs/2303.16526)
- Repository: [yihengli620/HybridPoint](https://github.com/yihengli620/HybridPoint)

## License

See the original HybridPoint repository for licensing information.
