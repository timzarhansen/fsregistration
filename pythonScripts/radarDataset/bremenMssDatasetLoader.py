"""
Bremen-MSS dataset loader for registration benchmarking.

Loads pre-processed Bremen-MSS sequences (PNG images + PCD point clouds
+ info.yaml start_pose files) and provides an interface compatible with
the Boreas benchmark registration methods.

Each sequence directory contains per-scan files:
  - scan_XXX_image.png    (256x256 cartesian image, 45m grid extent)
  - scan_XXX_cloud.pcd    (point cloud in scan-start local frame)
  - scan_XXX_info.yaml    (start_pose GT position+orientation)

The GT relative transform between scans is computed from start_pose:
  T_gt = inv(T_world_prev) * T_world_curr
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import yaml

# Open3D for PCD loading
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    o3d = None
    OPEN3D_AVAILABLE = False


def _quat_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert quaternion (w, x, y, z convention from Eigen) to 3x3 rotation matrix."""
    return np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),         1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),         2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
    ])


def _pos_quat_to_transform(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Build 4x4 homogeneous transform from position (3,) and quaternion (4,)."""
    T = np.eye(4)
    T[:3, :3] = _quat_to_rotation_matrix(quat[0], quat[1], quat[2], quat[3])
    T[:3, 3] = pos
    return T


class BremenMSSSequence:
    """Adapter for a single Bremen-MSS processed sequence.

    Provides a duck-type-compatible interface with BoreasSequence so that
    the existing boreasBenchmark.run_benchmark() can be used directly.
    """

    def __init__(self, seq_dir: str):
        self.seq_dir = Path(seq_dir)
        if not self.seq_dir.is_dir():
            raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")

        # Discover scans: find all _image.png files and extract indices
        png_files = sorted(self.seq_dir.glob("scan_*_image.png"))
        pattern = re.compile(r"scan_(\d+)_image\.png$")

        self._scan_indices: List[int] = []
        for f in png_files:
            m = pattern.match(f.name)
            if m:
                self._scan_indices.append(int(m.group(1)))

        self._scan_indices.sort()
        self._length = len(self._scan_indices)

        if self._length == 0:
            raise ValueError(f"No scan_*_image.png files found in {seq_dir}")

        # Cache info.yaml data (scan_index -> dict)
        self._info_cache: dict = {}

    @property
    def length(self) -> int:
        """Number of scans in this sequence."""
        return self._length

    def _get_info(self, idx: int) -> dict:
        """Load and cache info.yaml for a scan index."""
        if idx not in self._info_cache:
            scan_i = self._scan_indices[idx]
            info_path = self.seq_dir / f"scan_{scan_i:03d}_info.yaml"
            with open(info_path, "r") as f:
                self._info_cache[idx] = yaml.safe_load(f)
        return self._info_cache[idx]

    def get_cartesian_image(self, idx: int, N: int = None, size_of_pixel: float = None) -> np.ndarray:
        """Load scan image as float64 array in [0, 1].

        Args:
            idx: Scan index (0-based into the sorted list).
            N: If provided, resize image to N x N (ignored if None).
            size_of_pixel: Ignored (kept for interface compatibility with BoreasSequence).

        Returns:
            N x N float64 array in [0, 1].
        """
        scan_i = self._scan_indices[idx]
        img_path = self.seq_dir / f"scan_{scan_i:03d}_image.png"

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Failed to load image: {img_path}")

        img = img.astype(np.float64) / 255.0

        # Resize if requested (native is 256x256)
        if N is not None and N != img.shape[0]:
            img = cv2.resize(img, (N, N), interpolation=cv2.INTER_LINEAR)

        return img

    def get_gt_transform(self, prev_idx: int, curr_idx: int) -> np.ndarray:
        """Compute ground truth relative transform between two scans.

        Reads start_pose from info.yaml for both scans, computes:
            T_gt = inv(T_world_prev) * T_world_curr

        Args:
            prev_idx: Previous scan index (0-based).
            curr_idx: Current scan index (0-based).

        Returns:
            4x4 homogeneous transformation matrix.
        """
        info_prev = self._get_info(prev_idx)
        info_curr = self._get_info(curr_idx)

        def _parse_pose(info: dict):
            pos = info["start_pose"]["position"]
            ori = info["start_pose"]["orientation"]
            return np.array([pos["x"], pos["y"], pos["z"]]), \
                   np.array([ori["x"], ori["y"], ori["z"], ori["w"]])

        pos_prev, quat_prev = _parse_pose(info_prev)
        pos_curr, quat_curr = _parse_pose(info_curr)

        T_w_prev = _pos_quat_to_transform(pos_prev, quat_prev)
        T_w_curr = _pos_quat_to_transform(pos_curr, quat_curr)

        T_gt = np.linalg.inv(T_w_prev) @ T_w_curr
        return T_gt.astype(np.float64)

    def get_raw_point_cloud(self, idx: int, threshold: float = 0.0) -> Optional[np.ndarray]:
        """Load point cloud from PCD file.

        Args:
            idx: Scan index (0-based).
            threshold: Intensity threshold for filtering (0.0 = all points).

        Returns:
            Nx3 array of (x, y, intensity) or None if Open3D unavailable.
        """
        if not OPEN3D_AVAILABLE:
            return None

        scan_i = self._scan_indices[idx]
        pcd_path = self.seq_dir / f"scan_{scan_i:03d}_cloud.pcd"

        if not pcd_path.exists():
            return None

        pcd = o3d.io.read_point_cloud(str(pcd_path))
        points = np.asarray(pcd.points)  # (N, 3) = (x, y, z)
        intensities = np.asarray(pcd.colors)  # may be empty

        # Check for intensity field in PCD
        # PointXYZI stores intensity in the point struct; Open3D may put it in colors
        if intensities.shape[0] == points.shape[0]:
            # Use first channel of colors as intensity
            intensity_vals = intensities[:, 0]
        else:
            # No intensity available
            intensity_vals = np.ones(points.shape[0])

        # Build result: (x, y, intensity)
        result = np.column_stack([points[:, 0], points[:, 1], intensity_vals])

        # Apply intensity threshold
        if threshold > 0.0:
            mask = result[:, 2] >= threshold
            result = result[mask]

        return result

    @property
    def sequence(self):
        """Duck-type compatibility with BoreasSequence for naming purposes."""
        return self


class _SeqName:
    """Minimal object to provide .ID for compatibility with boreasBenchmark output naming."""
    def __init__(self, seq_number: int, seq_name: str):
        self.ID = seq_name
        self.number = seq_number


def load_sequence(seq_dir: str) -> BremenMSSSequence:
    """Load a Bremen-MSS sequence by directory path.

    Args:
        seq_dir: Path to the sequence directory.

    Returns:
        BremenMSSSequence instance.
    """
    return BremenMSSSequence(seq_dir)


def list_sequences(data_dir: str) -> List[tuple]:
    """Discover all Bremen-MSS sequences in a data directory.

    Scans for 'sequence_<N>' subdirectories, sorted by number.

    Args:
        data_dir: Path to Bremen-MSS-Processed root directory.

    Returns:
        List of (seq_number, seq_name, seq_path) tuples.
    """
    data_path = Path(data_dir)
    pattern = re.compile(r"sequence_(\d+)$")

    sequences = []
    for d in sorted(data_path.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m:
            seq_num = int(m.group(1))
            sequences.append((seq_num, d.name, str(d)))

    return sequences


def load_single_sequence(data_dir: str, seq_name: str) -> BremenMSSSequence:
    """Load a single Bremen-MSS sequence by name.

    Args:
        data_dir: Path to Bremen-MSS-Processed root directory.
        seq_name: Sequence directory name, e.g. 'sequence_5'.

    Returns:
        BremenMSSSequence instance.
    """
    seq_path = Path(data_dir) / seq_name
    if not seq_path.is_dir():
        raise FileNotFoundError(f"Sequence not found: {seq_path}")
    return BremenMSSSequence(str(seq_path))
