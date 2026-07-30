"""
Lidar simulation dataset loader for registration benchmarking.

Loads datasets produced by the scan_data_gazebo_sim recorder and provides
an interface compatible with the Bremen-MSS / Boreas benchmark registration
methods (same duck-type interface as BremenMSSSequence).

Each dataset directory (YYYYMMDD_HHMMSS/) contains:
  - scans/*.npy          (720 × float32 polar range arrays)
  - poses.csv            (id, time_sec, x, y, yaw)
  - transforms.csv       (id0, id1, dx, dy, dtheta)  -- computed on shutdown
  - metadata.yaml        (num_beams, angle_min/max, range_max, ...)
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml


class LidarSimSequence:
    """Adapter for a single gazebo-sim dataset sequence.

    Provides the duck-type interface expected by
    bremenMssBenchmark.run_benchmark() (length, get_cartesian_image,
    get_gt_transform, get_raw_point_cloud).
    """

    def __init__(self, seq_dir: str):
        self.seq_dir = Path(seq_dir)
        if not self.seq_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {seq_dir}")

        # --- Metadata ---
        meta_path = self.seq_dir / "metadata.yaml"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.yaml not found in {seq_dir}")
        with open(meta_path) as f:
            self._meta = yaml.safe_load(f)

        # --- Poses ---
        poses_path = self.seq_dir / "poses.csv"
        if not poses_path.exists():
            raise FileNotFoundError(f"poses.csv not found in {seq_dir}")
        # Columns: id, time_sec, x, y, yaw
        self._poses = np.loadtxt(poses_path, delimiter=",", skiprows=1,
                                 usecols=(0, 1, 2, 3, 4))
        self._length = len(self._poses)

        # --- Scan parameters ---
        self._num_beams = int(self._meta.get("num_beams", 720))
        self._angle_min = float(self._meta.get("angle_min_rad", -np.pi))
        self._angle_max = float(self._meta.get("angle_max_rad", np.pi))
        self._angle_inc = float(self._meta.get(
            "angle_increment_rad",
            (self._angle_max - self._angle_min) / self._num_beams,
        ))
        self._range_max = float(self._meta.get("range_max_m", 30.0))
        self._range_min = float(self._meta.get("range_min_m", 0.1))
        self._angles = np.linspace(
            self._angle_min, self._angle_max,
            self._num_beams, endpoint=False,
        ).astype(np.float64)

        # --- Scans directory ---
        self._scans_dir = self.seq_dir / "scans"
        if not self._scans_dir.is_dir():
            raise FileNotFoundError(f"scans/ directory not found in {seq_dir}")

    # ================================================================
    # Interface required by benchmark
    # ================================================================

    @property
    def length(self) -> int:
        return self._length

    def get_cartesian_image(
        self, idx: int, N: int = 256, size_of_pixel: float = None
    ) -> np.ndarray:
        """Convert polar range scan to N×N cartesian occupancy image.

        Each beam (range, angle) is projected to a point in the xy-plane,
        scattered onto the pixel grid, then Gaussian-blurred for robustness.
        The result is a float64 array in [0, 1].

        Args:
            idx: Scan index (0-based).
            N: Image grid size (N×N).
            size_of_pixel: Meters per pixel. Defaults to 2*15/N (15 m radius
                           coverage, enough for the maze worlds).

        Returns:
            N×N float64 array in [0, 1].
        """
        if size_of_pixel is None:
            size_of_pixel = (2.0 * 15.0) / N   # 15 m radius default

        ranges = self._load_scan(idx)

        # Filter valid returns
        valid = (
            np.isfinite(ranges)
            & (ranges >= self._range_min)
            & (ranges <= self._range_max)
        )
        angles = self._angles[valid]
        ranges = ranges[valid]

        # Convert polar → cartesian
        x = ranges * np.cos(angles)   # forward (robot x)
        y = ranges * np.sin(angles)   # left    (robot y)

        # Maple to pixel grid
        half = N / 2.0
        px = ((x / size_of_pixel) + half).astype(np.int32)
        py = ((y / size_of_pixel) + half).astype(np.int32)

        inside = (px >= 0) & (px < N) & (py >= 0) & (py < N)
        px, py = px[inside], py[inside]

        img = np.zeros((N, N), dtype=np.float64)
        np.add.at(img, (py, px), 1)        # accumulate overlapping hits

        # Normalise to [0, 1]
        max_val = img.max()
        if max_val > 0:
            img /= max_val

        # Gentle blur for feature-based methods
        if N >= 20:
            k = max(3, (N // 20) | 1)      # odd kernel ≈ N/20
            img = cv2.GaussianBlur(img, (k, k), 0)

        return img

    def get_gt_transform(self, prev_idx: int, curr_idx: int) -> np.ndarray:
        """Ground-truth relative transform between two consecutive scans.

        Builds 4×4 world transforms from (x, y, yaw) in poses.csv and
        returns inv(T_prev) * T_curr — same convention as bremenMss.

        Args:
            prev_idx: Previous scan index (0-based).
            curr_idx: Current scan index (0-based).

        Returns:
            4×4 homogeneous transform matrix (float64).
        """
        def _pose_to_SE3(x, y, yaw):
            T = np.eye(4, dtype=np.float64)
            c, s = np.cos(yaw), np.sin(yaw)
            T[0, 0] = c; T[0, 1] = -s; T[0, 3] = x
            T[1, 0] = s; T[1, 1] =  c; T[1, 3] = y
            return T

        row_p = self._poses[prev_idx]
        row_c = self._poses[curr_idx]

        T_prev = _pose_to_SE3(row_p[2], row_p[3], row_p[4])
        T_curr = _pose_to_SE3(row_c[2], row_c[3], row_c[4])

        return np.linalg.inv(T_prev) @ T_curr

    def get_raw_point_cloud(
        self, idx: int, threshold: float = 0.0
    ) -> Optional[np.ndarray]:
        """Convert polar scan to (x, y, intensity) point cloud.

        Args:
            idx: Scan index (0-based).
            threshold: Intensity threshold (no intensity data in sim,
                       so threshold has no effect — included for interface
                       compatibility).

        Returns:
            N×3 array of (x, y, intensity=1.0) points.
        """
        _ = threshold   # unused for simulated LiDAR (no intensity)
        ranges = self._load_scan(idx)

        valid = (
            np.isfinite(ranges)
            & (ranges >= self._range_min)
            & (ranges <= self._range_max)
        )
        angles = self._angles[valid]
        ranges = ranges[valid]

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        intensity = np.ones_like(x)

        return np.column_stack([x, y, intensity]).astype(np.float64)

    # ================================================================
    # Internal helpers
    # ================================================================

    def _load_scan(self, idx: int) -> np.ndarray:
        """Load a single scan .npy file, return float64 ranges."""
        scan_path = self._scans_dir / f"{idx:06d}.npy"
        if not scan_path.exists():
            raise FileNotFoundError(f"Scan file not found: {scan_path}")
        return np.load(str(scan_path)).astype(np.float64)


# ====================================================================
# Discovery & loading helpers (same interface as bremenMssDatasetLoader)
# ====================================================================

def list_sequences(data_dir: str) -> List[Tuple[int, str, str]]:
    """Discover all simulation dataset directories.

    Scans for subdirectories that contain metadata.yaml + poses.csv + scans/.

    Args:
        data_dir: Path to the parent directory containing dataset subdirs.

    Returns:
        List of (seq_number, seq_name, seq_path) tuples, sorted by name.
    """
    data_path = Path(data_dir)
    sequences = []

    for d in sorted(data_path.iterdir()):
        if not d.is_dir():
            continue
        # Must contain the minimal signature of a valid dataset
        if not (d / "metadata.yaml").is_file():
            continue
        if not (d / "poses.csv").is_file():
            continue
        if not (d / "scans").is_dir():
            continue

        seq_name = d.name
        seq_path = str(d)
        # Use the directory index among valid datasets as sequence number
        sequences.append((len(sequences) + 1, seq_name, seq_path))

    return sequences


def load_single_sequence(data_dir: str, seq_name: str) -> LidarSimSequence:
    """Load a single simulation dataset by name.

    Args:
        data_dir: Parent directory containing dataset subdirectories.
        seq_name: Dataset directory name, e.g. '20260729_152030'.

    Returns:
        LidarSimSequence instance.
    """
    seq_path = Path(data_dir) / seq_name
    if not seq_path.is_dir():
        raise FileNotFoundError(f"Dataset not found: {seq_path}")

    return LidarSimSequence(str(seq_path))
