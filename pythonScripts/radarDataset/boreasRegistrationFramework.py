################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Dan Barnes (dbarnes@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################

"""
Extensible registration framework for Boreas radar dataset.

Supports multiple registration methods via a common interface.
Add a new method by implementing BaseRegistrationMethod and registering it.

Usage:
    python boreasRegistrationFramework.py --method fs2d --sequence 0 --size_of_pixel 0.01 <data_dir>
    python boreasRegistrationFramework.py --method icp --sequence 0 --size_of_pixel 0.01 <data_dir>
    python boreasRegistrationFramework.py --method fs2d --method icp --compare --sequence 0 --size_of_pixel 0.01 <data_dir>
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import csv
import time
import argparse
import os
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from sdk.radar import load_radar, radar_polar_to_cartesian
from pyboreas import BoreasDataset


# ============================================================================
# Result type
# ============================================================================

@dataclass
class RegistrationResult:
    """Unified output from any registration method."""
    transform: np.ndarray           # 4x4 relative transformation
    confidence: float               # peak height / correlation score
    method_name: str
    computation_time: float         # seconds
    metadata: dict = field(default_factory=dict)  # method-specific info


# ============================================================================
# Base class
# ============================================================================

class BaseRegistrationMethod(ABC):
    """Abstract base class for all registration methods."""

    @abstractmethod
    def __init__(self, config: dict):
        """Initialize method with configuration.

        Args:
            config: dict of method-specific parameters.
        """
        self.config = config
        self._name = self.__class__.__name__

    @abstractmethod
    def register(self, img1: np.ndarray, img2: np.ndarray) -> RegistrationResult:
        """Register two 2D cartesian images.

        Args:
            img1: First image (N x N).
            img2: Second image (N x N).

        Returns:
            RegistrationResult with the estimated transform.
        """
        ...

    def get_name(self) -> str:
        return self._name


# ============================================================================
# FS2D Registration (wraps existing pybind wrapper)
# ============================================================================

class FS2DRegistration(BaseRegistrationMethod):
    """SOFT-based FS2D registration via pybind_registration_2d."""

    def __init__(self, config: dict):
        super().__init__(config)
        from pybind_registration_2d import SoftRegistrationWrapper2D

        self.N = config.get("N", 128)
        self.use_clahe = config.get("use_clahe", True)
        self.use_hamming = config.get("use_hamming", True)
        self.potential_for_necessary_peak = config.get("potential_for_necessary_peak", 0.01)
        self.multiple_radii = config.get("multiple_radii", True)
        self.use_gauss = config.get("use_gauss", False)

        self.wrapper = SoftRegistrationWrapper2D(self.N)

    def register(self, img1: np.ndarray, img2: np.ndarray) -> RegistrationResult:
        t0 = time.time()

        image_1 = img1.astype(np.float64).reshape(-1)
        image_2 = img2.astype(np.float64).reshape(-1)
        cell_size = self.config.get("size_of_pixel", 0.01)

        list_peaks = self.wrapper.register_all_solutions(
            image_1, image_2,
            cellSize=cell_size,
            useGauss=self.use_gauss,
            debug=False,
            potentialNecessaryForPeak=self.potential_for_necessary_peak,
            multipleRadii=self.multiple_radii,
            useClahe=self.use_clahe,
            useHamming=self.use_hamming
        )

        # Find peak with highest translation peak height
        highest_peak = 0.0
        index_highest = 0
        for i, peak in enumerate(list_peaks):
            if peak.potentialTranslations[0].peakHeight > highest_peak:
                highest_peak = peak.potentialTranslations[0].peakHeight
                index_highest = i

        peak = list_peaks[index_highest]

        transform = np.eye(4)
        yaw = peak.potentialRotation.angle
        transform[:3, :3] = R.from_euler("z", yaw).as_matrix()
        tx, ty = peak.potentialTranslations[0].translationSI
        transform[:3, 3] = [tx, ty, 0.0]

        elapsed = time.time() - t0

        return RegistrationResult(
            transform=transform,
            confidence=highest_peak,
            method_name="fs2d",
            computation_time=elapsed,
            metadata={
                "rotation_angle": yaw,
                "translation": (tx, ty),
                "peak_height": highest_peak,
                "num_solutions": len(list_peaks)
            }
        )


# ============================================================================
# Placeholder method stubs
# ============================================================================

class ICPRegistration(BaseRegistrationMethod):
    """Open3D point-to-point ICP (placeholder).

    Would convert images to 2D point clouds and run registration_icp.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._name = "icp"

    def register(self, img1: np.ndarray, img2: np.ndarray) -> RegistrationResult:
        raise NotImplementedError(
            "ICPRegistration not yet implemented. "
            "Plan: extract 2D point cloud from images (threshold-based), "
            "then use o3d.pipelines.registration.registration_icp."
        )


class FourierMellinRegistration(BaseRegistrationMethod):
    """Fourier-Mellin Transform registration (placeholder).

    Uses log-polar FFT for rotation + translation estimation.
    Reference: implementation at debug_results/registrationFourier/FMT/register.py
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._name = "fourier_mellin"

    def register(self, img1: np.ndarray, img2: np.ndarray) -> RegistrationResult:
        raise NotImplementedError(
            "FourierMellinRegistration not yet implemented. "
            "Plan: port the FMT algorithm from debug_results/registrationFourier/FMT/register.py."
        )


class NDTRegistration(BaseRegistrationMethod):
    """Normal Distributions Transform registration (placeholder)."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._name = "ndt"

    def register(self, img1: np.ndarray, img2: np.ndarray) -> RegistrationResult:
        raise NotImplementedError(
            "NDTRegistration not yet implemented."
        )


class SIFTRegistration(BaseRegistrationMethod):
    """SIFT feature-based registration with RANSAC (placeholder)."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._name = "sift"

    def register(self, img1: np.ndarray, img2: np.ndarray) -> RegistrationResult:
        raise NotImplementedError(
            "SIFTRegistration not yet implemented. "
            "Plan: use cv2.SIFT_create(), find keypoints/descriptors, "
            "BFMatcher with FLANN, RANSAC for affine/homography estimation."
        )


# ============================================================================
# Factory / Registry
# ============================================================================

class RegistrationFactory:
    """Factory for creating registration methods by name."""

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, method_class: type):
        """Register a method class under a name."""
        if not issubclass(method_class, BaseRegistrationMethod):
            raise TypeError(f"{method_class} must be a subclass of BaseRegistrationMethod")
        cls._registry[name] = method_class

    @classmethod
    def create(cls, name: str, config: dict) -> BaseRegistrationMethod:
        """Instantiate a method by name."""
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown method '{name}'. Available: {available}")
        return cls._registry[name](config)

    @classmethod
    def list_methods(cls) -> List[str]:
        return sorted(cls._registry.keys())


# Auto-register available methods
RegistrationFactory.register("fs2d", FS2DRegistration)
# RegistrationFactory.register("icp", ICPRegistration)          # uncomment when implemented
# RegistrationFactory.register("fourier_mellin", FourierMellinRegistration)  # uncomment when implemented
# RegistrationFactory.register("sift", SIFTRegistration)         # uncomment when implemented


# ============================================================================
# Utility functions (from boreasTestFileFS2D.py)
# ============================================================================

def get_image_from_sequence(seq, index, cart_resolution, cart_pixel_width):
    frame = seq.get_radar(index)
    img = frame.polar_to_cart(
        cart_resolution=cart_resolution,
        cart_pixel_width=cart_pixel_width,
        in_place=False,
    )
    return img


def get_affine_matrix(input_matrix):
    """Extract 2D affine transform from 4x4 matrix."""
    input_matrix = np.linalg.inv(input_matrix)
    result = np.eye(3)
    result[:2, :2] = input_matrix[:2, :2]
    result[0, 2] = -input_matrix[1, 3]
    result[1, 2] = input_matrix[0, 3]
    return result


def transform_diff(matrix1, matrix2):
    """Compute translation and rotation difference between two 3x3 affine matrices."""
    t1 = np.array([matrix1[0, 2], matrix1[1, 2]])
    t2 = np.array([matrix2[0, 2], matrix2[1, 2]])
    trans_diff = t2 - t1

    r1 = matrix1[:2, :2]
    r2 = matrix2[:2, :2]
    angle_diff = np.degrees(
        np.arctan2(r2[1, 0], r2[0, 0]) - np.arctan2(r1[1, 0], r1[0, 0])
    )
    return trans_diff, angle_diff


def fuse_images(images_over_time, estimated_transformations):
    """Fuse multiple images using their absolute transformation matrices."""
    assert len(images_over_time) == len(estimated_transformations)
    if not images_over_time:
        return None

    # Calculate canvas boundaries
    min_x = float("inf")
    max_x = -float("inf")
    min_y = float("inf")
    max_y = -float("inf")

    for i in range(len(images_over_time)):
        img = images_over_time[i]
        tmat = get_affine_matrix(estimated_transformations[i])
        h, w = img.shape[:2]
        corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float64).reshape(-1, 1, 2)

        if tmat.shape == (3, 3):
            transformed = cv2.perspectiveTransform(corners, tmat)
        else:
            transformed = cv2.transform(corners, tmat)

        transformed = transformed.squeeze()
        min_x = min(min_x, int(np.min(transformed[:, 0])))
        max_x = max(max_x, int(np.max(transformed[:, 0])))
        min_y = min(min_y, int(np.min(transformed[:, 1])))
        max_y = max(max_y, int(np.max(transformed[:, 1])))

    canvas_w = (max_x - min_x) + 1
    canvas_h = (max_y - min_y) + 1
    tx, ty = -min_x, -min_y

    adjusted = []
    for tmat in estimated_transformations:
        t = get_affine_matrix(tmat)
        if t.shape == (3, 3):
            trans = np.eye(3)
            trans[0, 2] = tx
            trans[1, 2] = ty
            t = trans @ t
        else:
            t = t.copy()
            t[:2, 2] += [tx, ty]
        adjusted.append(t)

    warped = []
    for i in range(len(images_over_time)):
        img = images_over_time[i]
        if adjusted[i].shape == (3, 3):
            w = cv2.warpPerspective(img, adjusted[i], (canvas_w, canvas_h))
        else:
            w = cv2.warpAffine(img, adjusted[i][:2], (canvas_w, canvas_h))
        warped.append(w)

    if warped:
        return np.mean(np.stack(warped).astype(np.float64), axis=0).astype(np.uint8)
    return np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)


def matrix_to_transform(matrix):
    """Convert 4x4 matrix to [x, y, z, roll, pitch, yaw]."""
    translation = matrix[:3, 3]
    rotation = R.from_matrix(matrix[:3, :3])
    euler = rotation.as_euler("xyz", degrees=False)
    return np.asarray([translation[0], translation[1], translation[2],
                       euler[0], euler[1], euler[2]])


# ============================================================================
# Main loop
# ============================================================================

def run_sequence(args, method_class, method_configs, seq, bd, radar_file_path):
    """Run a single sequence with one or more methods.

    Args:
        args: CLI arguments.
        method_class: The method name to use (for output directory naming).
        method_configs: dict of method_name -> config dict.
        seq: Boreas sequence.
        bd: BoreasDataset.
        radar_file_path: path to save results.
    """
    N = args.N
    size_of_pixel = args.size_of_pixel
    matching_every_nth = args.matching_every_nth_image
    sequence_number = args.sequence

    # Determine output directory
    method_key = "_".join(sorted(method_configs.keys()))
    save_dir = Path(f"saveResultsBoreas/{sequence_number:02d}_{N:03d}_{int(size_of_pixel*100)}_{matching_every_nth}/{method_key}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {save_dir}")

    length_of_radar_scans = len(seq.radar_frames)
    print(f"Sequence has {length_of_radar_scans} radar scans")

    # Results storage
    all_method_results = {}  # method_name -> list of results
    for method_name in method_configs:
        all_method_results[method_name] = {
            "transforms": [],
            "confidence": [],
            "errors": [],
            "times": [],
            "gt_poses": [],
            "est_poses": [],
        }

    # Initialize: first frame is the reference
    first_img = get_image_from_sequence(seq, 0, size_of_pixel, N)
    images_over_time = [first_img * 255.0]

    # Cumulative transforms per method: [method_name] -> list of 4x4 matrices
    cumulative_transforms = {}
    for method_name in method_configs:
        cumulative_transforms[method_name] = [np.eye(4)]

    print(f"Matching every {matching_every_nth}th image...")

    for index in range(1, length_of_radar_scans):
        if index >= matching_every_nth:
            if index % matching_every_nth == 0:
                prev_index = index - matching_every_nth

                # Get images
                img_prev = get_image_from_sequence(seq, prev_index, size_of_pixel, N)
                img_curr = get_image_from_sequence(seq, index, size_of_pixel, N)
                images_over_time.append(img_curr * 255.0)

                # Get GT transformation
                gt_transform = np.matmul(
                    np.linalg.inv(np.asarray(seq.get_radar(prev_index).pose, dtype=np.float64)),
                    np.asarray(seq.get_radar(index).pose, dtype=np.float64)
                )
                gt_affine = get_affine_matrix(gt_transform)

                # Run each method
                for method_name, config in method_configs.items():
                    method = RegistrationFactory.create(method_name, config)
                    result = method.register(img_prev, img_curr)

                    # Update cumulative transform
                    prev_cumulative = cumulative_transforms[method_name][-1]
                    new_cumulative = prev_cumulative @ result.transform
                    cumulative_transforms[method_name].append(new_cumulative)

                    est_affine = get_affine_matrix(result.transform)

                    # Compute errors
                    trans_error, rot_error = transform_diff(gt_affine, est_affine)

                    # Store results
                    all_method_results[method_name]["transforms"].append(result.transform)
                    all_method_results[method_name]["confidence"].append(result.confidence)
                    all_method_results[method_name]["errors"].append((trans_error, rot_error))
                    all_method_results[method_name]["times"].append(result.computation_time)
                    all_method_results[method_name]["gt_poses"].append(gt_transform)
                    all_method_results[method_name]["est_poses"].append(new_cumulative)

                    print(f"  [{method_name}] idx={index}: rot_err={rot_error:.3f} deg, "
                          f"trans_err={np.linalg.norm(trans_error):.4f}m, "
                          f"time={result.computation_time*1000:.1f}ms, "
                          f"conf={result.confidence:.4f}")

                # Save blended images for fs2d
                if "fs2d" in method_configs:
                    fs2d_result = all_method_results["fs2d"][-1]
                    fs2d_affine = get_affine_matrix(fs2d_result.transform)
                    warped = cv2.warpPerspective(img_curr, fs2d_affine, (img_prev.shape[1], img_prev.shape[0]))
                    blended = cv2.addWeighted(img_prev, 0.5, warped, 0.5, 0)
                    cv2.imwrite(str(save_dir / f"blended_{index:04d}.png"), blended * 255.0)

    # Save CSV results per method
    for method_name, data in all_method_results.items():
        errors = data["errors"]
        rotation_errors = [e[1] for e in errors]
        translation_errors = [np.linalg.norm(e[0]) for e in errors]

        # Rotation error CSV
        with open(save_dir / f"{method_name}_rotation_error.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for val in rotation_errors:
                writer.writerow([val])

        # Translation error CSV
        with open(save_dir / f"{method_name}_translation_error.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for val in translation_errors:
                writer.writerow([val])

        # GT poses CSV
        with open(save_dir / f"{method_name}_gt_poses.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for mat in data["gt_poses"]:
                x, y, z, roll, pitch, yaw = matrix_to_transform(mat)
                writer.writerow([x, y, yaw])

        # Estimated poses CSV
        with open(save_dir / f"{method_name}_est_poses.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for mat in data["est_poses"]:
                x, y, z, roll, pitch, yaw = matrix_to_transform(mat)
                writer.writerow([x, y, yaw])

        # Fusion (only for fs2d to avoid excessive computation)
        if method_name == "fs2d" and len(images_over_time) > 1:
            fused = fuse_images(images_over_time, cumulative_transforms[method_name])
            if fused is not None:
                cv2.imwrite(str(save_dir / "fused_map.png"), fused)

        # Summary
        avg_rot = np.mean(np.abs(rotation_errors))
        avg_trans = np.mean(translation_errors)
        avg_time = np.mean(data["times"])
        print(f"\n[{method_name}] Summary:")
        print(f"  Avg rotation error: {avg_rot:.3f} deg")
        print(f"  Avg translation error: {avg_trans:.4f} m")
        print(f"  Avg computation time: {avg_time*1000:.1f} ms")

    # If multiple methods, write comparison CSV
    if len(method_configs) > 1:
        with open(save_dir / "comparison.csv", "w", newline="") as f:
            writer = csv.writer(f)
            header = ["index"]
            for method_name in sorted(method_configs.keys()):
                header.extend([f"{method_name}_rot_error", f"{method_name}_trans_error",
                              f"{method_name}_time_ms", f"{method_name}_confidence"])
            header.append("gt_rot_error")
            header.append("gt_trans_error")
            writer.writerow(header)

            for i in range(len(errors)):
                row = [i]
                for method_name in sorted(method_configs.keys()):
                    data = all_method_results[method_name]
                    rot_e = np.abs(data["errors"][i][1])
                    trans_e = np.linalg.norm(data["errors"][i][0])
                    t_ms = data["times"][i] * 1000
                    conf = data["confidence"][i]
                    row.extend([rot_e, trans_e, t_ms, conf])
                # GT errors (from the last method's GT, all share the same GT)
                gt_trans_e = np.linalg.norm(errors[i][0])
                gt_rot_e = np.abs(errors[i][1])
                row.extend([gt_rot_e, gt_trans_e])
                writer.writerow(row)

    print(f"\nDone. Results saved to: {save_dir}")


def main():
    np.set_printoptions(precision=5, suppress=True)

    parser = argparse.ArgumentParser(
        description="Extensible registration framework for Boreas radar dataset."
    )
    parser.add_argument("--method", action="append", required=True,
                        help="Registration method(s) to run. Can specify multiple for comparison. "
                             f"Available: {RegistrationFactory.list_methods()}")
    parser.add_argument("--method-config", action="append", default=[],
                        help="Method config in format 'method_name.key=value'. "
                             "e.g., 'fs2d.N=128 fs2d.use_clahe=1 fs2d.potential_for_necessary_peak=0.01'")
    parser.add_argument("--N", type=int, default=128, help="Image grid size (N x N).")
    parser.add_argument("--size_of_pixel", type=float, default=0.01, help="Size of a pixel in meters.")
    parser.add_argument("--matching_every_nth_image", type=int, default=1,
                        help="Match every Nth image.")
    parser.add_argument("--sequence", type=int, default=0, help="Sequence number from Boreas dataset.")
    parser.add_argument("--compare", action="store_true",
                        help="Enable comparison mode when multiple methods are specified.")
    parser.add_argument("data_dir", type=str, help="Path to Boreas radar data directory.")

    args = parser.parse_args()

    # Parse method configs
    method_configs = {}
    for spec in args.method_config:
        # Format: "method_name.key=value" or "method_name.key1=val1 key_name.key2=val2"
        parts = spec.split()
        for part in parts:
            method_key, _, key_value = part.partition(".")
            if "." not in method_key:
                continue
            method_name, _, param = method_key.partition(".")
            k, _, v = param.partition("=")
            v = _parse_value(v)
            if method_name not in method_configs:
                method_configs[method_name] = {}
            method_configs[method_name][k] = v

    # Apply default config values if not overridden
    default_fs2d_config = {
        "N": args.N,
        "use_clahe": True,
        "use_hamming": True,
        "potential_for_necessary_peak": 0.01,
        "multiple_radii": True,
        "use_gauss": False,
        "size_of_pixel": args.size_of_pixel,
    }
    if "fs2d" not in method_configs:
        method_configs["fs2d"] = default_fs2d_config
    else:
        # Merge defaults with user overrides
        for k, v in default_fs2d_config.items():
            if k not in method_configs["fs2d"]:
                method_configs["fs2d"][k] = v

    print(f"Methods: {list(method_configs.keys())}")
    for name, cfg in method_configs.items():
        print(f"  {name}: {cfg}")

    # Load dataset
    bd = BoreasDataset(args.data_dir, split=None, verbose=True)
    seq = bd.sequences[args.sequence]

    # Run
    methods_str = "_".join(sorted(method_configs.keys()))
    run_sequence(args, methods_str, method_configs, seq, bd, args.data_dir)


def _parse_value(v):
    """Parse a string value into the appropriate Python type."""
    if v.lower() in ("true", "1"):
        return True
    if v.lower() in ("false", "0"):
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


if __name__ == "__main__":
    main()
