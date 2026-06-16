# Plan: Split boreasRegistrationFramework.py into 3 Files

## Goal
Split the monolithic `boreasRegistrationFramework.py` into three focused files:
1. **boreasDatasetLoader.py** — data loading and preprocessing
2. **boreasRegistrationMethods.py** — registration algorithm implementations
3. **boreasRun.py** — entry point with config and CLI orchestration

---

## File 1: boreasDatasetLoader.py

**Responsibility:** Load the Boreas dataset and produce representations needed by registration methods (cartesian images, point clouds, raw radar scans).

### Exports
- `load_sequence(data_dir: str, sequence: int) -> BoreasSequence`
- `BoreasSequence` dataclass with:
  - `radar_frames` — list of raw radar frame objects (pose, polar data)
  - `get_cartesian_image(index, N, size_of_pixel) -> np.ndarray`
  - `get_point_cloud(index, N, size_of_pixel, threshold) -> np.ndarray` (for ICP/NDT)
  - `get_gt_transform(prev_index, curr_index) -> np.ndarray`
  - `length` — number of radar scans

### Functions to move here
- `get_image_from_sequence()` → becomes `BoreasSequence.get_cartesian_image()`
- `get_affine_matrix()`
- `transform_diff()`
- `matrix_to_transform()`
- `fuse_images()`
- `get_gt_transform()` (new helper)

### Imports
- `numpy`, `cv2`, `scipy.spatial.transform.Rotation`
- `pyboreas.BoreasDataset`
- `sdk.radar.load_radar`, `sdk.radar.radar_polar_to_cartesian`

---

## File 2: boreasRegistrationMethods.py

**Responsibility:** All registration algorithm classes. Each receives images (or point clouds) and returns a `RegistrationResult`.

### Exports
- `RegistrationResult` dataclass:
  - `transform: np.ndarray` (4x4 relative transformation)
  - `confidence: float` (peak height / correlation score)
  - `method_name: str`
  - `computation_time: float` (seconds)
  - `metadata: dict` (method-specific info)
- `BaseRegistrationMethod` ABC with `register(img1, img2) -> RegistrationResult`
- `FS2DRegistration` — wraps `SoftRegistrationWrapper2D`
- `ICPRegistration` — stub (placeholder)
- `FourierMellinRegistration` — stub (placeholder)
- `NDTRegistration` — stub (placeholder)
- `SIFTRegistration` — stub (placeholder)
- `RegistrationFactory` — registry pattern for dynamic method creation

### Imports
- `numpy`, `scipy`, `cv2`, `time`, `dataclasses`, `abc`, `typing`
- `pybind_registration_2d` (for FS2D only)

---

## File 3: boreasRun.py

**Responsibility:** Orchestration. Reads config (top of file) and/or CLI args, loads data, runs methods, saves results.

### Config block at top of file
```python
DEFAULT_CONFIG = {
    "sequences": [0],                       # which sequences to test
    "N": 128,                               # image grid size
    "size_of_pixel": 0.01,                  # meters per pixel
    "matching_every_nth_image": 1,          # match every Nth image
    "max_frames": None,                     # cap sequence length (None = full)
    "output_dir": "saveResultsBoreas",      # output directory
    "methods": ["fs2d"],                    # default method(s)
}
```

### Functions to move here
- `run_sequence(args, method_configs, seq, bd, radar_file_path)`
- `_parse_value(v)`
- `main()`
- `if __name__ == "__main__": main()`

### CLI args
- `--method` — registration method(s), can specify multiple for comparison
- `--method-config` — method config as `method_name.key=value`
- `--sequence` — sequence number (deprecated if `--sequences` used)
- `--N` — image grid size
- `--size_of_pixel` — meters per pixel
- `--matching_every_nth_image` — match every Nth image
- `--max-frames` — cap sequence length
- `--compare` — enable comparison mode with multiple methods
- `--sequences` — comma-separated list of sequences
- `data_dir` — path to Boreas radar data directory

### Imports
- `boreasDatasetLoader` (load_sequence, BoreasSequence)
- `boreasRegistrationMethods` (RegistrationFactory, RegistrationResult)
- `numpy`, `argparse`, `csv`, `pathlib`, `typing`

---

## Import graph

```
boreasRun.py
    ├── boreasDatasetLoader (load_sequence, BoreasSequence)
    └── boreasRegistrationMethods (RegistrationFactory, RegistrationResult)

boreasDatasetLoader.py
    ├── numpy, cv2, scipy
    ├── pyboreas
    └── sdk.radar

boreasRegistrationMethods.py
    ├── numpy, scipy, cv2, time
    └── pybind_registration_2d (FS2D only)
```

No circular dependencies. Each file is self-contained.

---

## Implementation steps

1. **Create `boreasDatasetLoader.py`**
   - Extract `BoreasSequence` dataclass with all data access methods
   - Move utility functions: `get_affine_matrix`, `transform_diff`, `matrix_to_transform`, `fuse_images`
   - Add `get_gt_transform` helper
   - Add `get_point_cloud` placeholder for ICP/NDT methods

2. **Create `boreasRegistrationMethods.py`**
   - Move `RegistrationResult` dataclass
   - Move `BaseRegistrationMethod` ABC
   - Move all method classes: `FS2DRegistration`, `ICPRegistration`, `FourierMellinRegistration`, `NDTRegistration`, `SIFTRegistration`
   - Move `RegistrationFactory` and auto-registration logic

3. **Create `boreasRun.py`**
   - Add `DEFAULT_CONFIG` block at top of file
   - Move `run_sequence()` — the main per-sequence loop
   - Move `_parse_value()` helper
   - Move `main()` and CLI argument parsing
   - Add `--sequences` support for running multiple sequences
   - Add `--max-frames` support for capping sequence length

4. **Verify**
   - Run `python boreasRun.py --method fs2d --sequence 0 --N 128 --size_of_pixel 0.01 <data_dir>`
   - Compare output CSVs and blended images against original `boreasRegistrationFramework.py` output
   - Test multi-method comparison: `--method fs2d --method icp --compare` (once ICP is implemented)

5. **Cleanup (optional, later)**
   - Remove or deprecate `boreasRegistrationFramework.py`
   - Update README.md to reference new file structure and entry point

---

## Decision points

- [ ] Should `boreasRun.py` also support a YAML/JSON config file in addition to CLI?
- [ ] Should point cloud extraction go in the loader or in the ICP method itself? (Recommendation: loader, so ICP just receives a point cloud)
- [ ] Any other methods to pre-stub? (e.g., feature-based, learning-based)
- [ ] Should fused map generation be per-method or only for fs2d? (Current: only fs2d to avoid excessive computation)

---

## Expected outcome

After the split:
- Each file has a single clear responsibility (data / algorithms / orchestration)
- Adding a new registration method only requires editing `boreasRegistrationMethods.py`
- Changing sequences, grid size, or other parameters only requires editing the config block in `boreasRun.py` or passing CLI args
- The original `boreasRegistrationFramework.py` output is preserved exactly
