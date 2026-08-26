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
5. SIFT
6. SURF
7. KAZE
8. AKAZE
9. LoFTR
10. E-LoFTR

## FS2D parameters (`viewBoreasPairs.py` / `comparisonBoreasPais.py`)

All FS2D-specific settings live at the top of the scripts in the `FS2D-specific config`
section. The config dict passed to `RegistrationFactory.create("fs2d", ...)` forwards them
through `boreasRegistrationMethods.FS2DRegistration` to the C++ wrapper
(`pybind_registration_2d`) and finally `softRegistrationClass`.

| Config constant | Config key | Default | Meaning |
|---|---|---|---|
| `NUM_ANGLES` | `num_angles` | `-1` | Angular resolution of the 1D correlation curve in direct mode (`-1` = auto, i.e. `N`). Only used when `USE_DIRECT = True`. |
| `R_MIN` | `r_min` | `0.0` | Min radial frequency radius of the FS2D rotation descriptor, in FFT grid units (pixels). **`0.0` = auto: N-dependent default** (`1 + floor(N*0.05)`), i.e. the original behavior. |
| `R_MAX` | `r_max` | `0.0` | Max radial frequency radius, in FFT grid units (pixels). **`0.0` = auto: N-dependent default** (`N/2 - floor(N*0.05)`). |

### The `0.0 = auto` convention

`r_min`/`r_max` are **absolute pixel values of the N×N FFT spectrum** (same convention as the
3D wrapper's `r_min`/`r_max`). Setting them keeps the descriptor band fixed regardless of N;
leaving them `0.0` keeps the original hardcoded, N-dependent defaults:

- `r_min = 0.0` → `minR = 1 + floor(N * 0.05)` (≈ 5% of N above DC)
- `r_max = 0.0` → `maxR = N / 2 - floor(N * 0.05)` (≈ 5% of N below Nyquist)

Example for N=256: `R_MIN = 32`, `R_MAX = 96` samples the band from radius 32 to 96 px.

Notes:
- `multipleRadii = False` collapses the band to a single ring `maxR-1` **only when both
  `r_min` and `r_max` are left on auto** (matches the original behavior).
- Changing `r_min`/`r_max`/`num_angles` only affects the **rotation** step; the translation
  step (phase correlation) is unchanged.

> ⚠️ `r_min`/`r_max` are passed as keyword arguments to the pybind wrapper. They are only
> accepted **after rebuilding** the C++ module: `colcon build --packages-select fsregistration`
> (then `source install/setup.bash`). With an outdated `.so` the FS2D method raises a TypeError.






