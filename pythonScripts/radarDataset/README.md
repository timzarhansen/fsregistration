## Overview

`boreasRegistrationFramework.py` processes Boreas radar sequences by converting polar scans to
cartesian images and registering consecutive frames. It outputs per-frame transformation errors,
GT/estimated poses, blended images, and fused maps.

The script supports multiple registration methods (FS2D, ICP, Fourier-Mellin, etc.) via a common
interface, allowing easy comparison by running multiple methods on the same sequence with `--compare`.

## Setup

### Dependencies

- `pyboreas` — Boreas dataset loader
- `pybind_registration_2d` — compiled from `src/fsregistration/src/pybind_registration_2d.cpp`
- `cv2`, `numpy`, `scipy`

Build the C++ wrapper via colcon:

```bash
colcon build --packages-select fsregistration
source install/setup.bash
```

### Usage

```bash
python boreasRegistrationFramework.py --method fs2d --sequence 0 --size_of_pixel 0.01 <data_dir>
python boreasRegistrationFramework.py --method fs2d --method icp --compare --sequence 0 --size_of_pixel 0.01 <data_dir>
```

Method-specific config via `--method-config`:

```bash
python boreasRegistrationFramework.py --method fs2d \
    --method-config "fs2d.N=256 fs2d.potential_for_necessary_peak=0.001" \
    --sequence 0 --size_of_pixel 0.01 <data_dir>
```

### Adding a new method

1. Inherit `BaseRegistrationMethod` and implement `register(self, img1, img2)` → `RegistrationResult`
2. Register it: `RegistrationFactory.register("my_method", MyMethod)`
3. Run: `--method my_method`

## Plan

1. ICP
2. Fourier-mellin
3. NDT P2D
4. NDT D2D
5. SIFT
6. SURF
7. KAZE
8. AKAZE
9. LoFTR
10. E-LoFTR






