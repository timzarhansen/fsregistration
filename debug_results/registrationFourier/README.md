# registrationFourier

Main registration algorithms and utilities for point cloud registration using Fourier-based methods.

## Structure

```
registrationFourier/
├── 3D/                    # 3D point cloud registration
│   ├── Odometry/          # Odometry-specific utilities
│   ├── predatorMatching/  # Predator dataset matching
│   └── xyz file operations/ # XYZ file parsing
├── FMT/                   # Fourier-Mellin Transform registration
├── ICRA2024Scripts/       # Scripts for ICRA 2024 paper
├── RadarStuff/            # Radar-specific processing
├── datasetReleaseJournal/ # Dataset release scripts
├── resultsJournalFMS2D/   # Results analysis for FMS 2D
├── resultsOfManyMatching/ # Matching results analysis
├── archive/               # ⚠️ Deprecated scripts (DO NOT USE)
└── [core modules]         # Core registration algorithms
```

## Core Modules

| File | Purpose |
|------|---------|
| `rotation_matrix.py` | Euler angle to rotation matrix conversion |
| `rotation_matrix_zyz.py` | ZYZ rotation matrix |
| `angles_r.py` | Rotation matrix to Euler angles |
| `fast_2d_peak_find.py` | 2D peak detection |
| `hipass_filter.py` | High-pass filter for Fourier-Mellin |
| `get_voxel_data.py` | Voxel grid population |
| `fourie_coeff.py` | Fourier coefficient calculation |
| `sampled_f_theta_phi.py` | Spherical harmonic sampling |
| `wignerd_function.py` | Wigner D-function calculation |
| `ylm_of_tp.py` | Spherical harmonics |
| `calculate_score_of_rotation.py` | Rotation scoring |
| `weighting_function.py` | Weighting for registration |

## Subdirectories

### 3D/
3D point cloud registration algorithms and utilities.

### FMT/
Fourier-Mellin Transform based registration.

### ICRA2024Scripts/
Scripts used for ICRA 2024 paper generation.

### RadarStuff/
Radar-specific point cloud processing and visualization.

### datasetReleaseJournal/
Scripts for dataset release and journal publication.

### resultsJournalFMS2D/
Results analysis and visualization for FMS 2D methods.

### resultsOfManyMatching/
Analysis of multiple matching results.

### archive/
⚠️ **Deprecated** - Legacy scripts from old development phases. Do not use.

## Usage Example

```python
from rotation_matrix import rotation_matrix
from angles_r import angles_r

# Create rotation matrix from Euler angles
R = rotation_matrix(roll, pitch, yaw)

# Extract Euler angles from rotation matrix
roll, pitch, yaw = angles_r(R)
```

## Dependencies

- numpy
- scipy
- matplotlib
- open3d
- pandas
- sympy
