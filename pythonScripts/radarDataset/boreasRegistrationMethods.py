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
Registration algorithm implementations for Boreas radar data.

Provides a common interface (BaseRegistrationMethod) for multiple registration
methods, with a factory pattern for dynamic creation.
"""

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import time
from scipy.spatial.transform import Rotation as R

# Add colcon install path for pybind_registration_2d
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_script_dir))))
_install_lib = os.path.join(_root_dir, 'install', 'fsregistration', 'lib', 'fsregistration')
if os.path.isdir(_install_lib):
    sys.path.insert(0, _install_lib)

# Try to import Open3D (needed for ICP, NDT)
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    o3d = None
    OPEN3D_AVAILABLE = False

# Try to import pybind_ndt (PCL NDT)
try:
    import pybind_ndt
    PCLNDT_AVAILABLE = True
except ImportError:
    pybind_ndt = None
    PCLNDT_AVAILABLE = False


@dataclass
class RegistrationResult:
    """Unified output from any registration method."""
    transform: np.ndarray           # 4x4 relative transformation
    confidence: float               # peak height / correlation score
    method_name: str
    computation_time: float         # seconds
    metadata: dict = field(default_factory=dict)  # method-specific info


def _image_to_pointcloud(img: np.ndarray, cell_size: float = 0.01,
                          scale: float = 1.0, threshold_pct: float = 5.0) -> "o3d.geometry.PointCloud":
    """Convert a 2D cartesian image to an Open3D point cloud.

    Maps pixel coordinates to metric space centered at origin,
    uses pixel intensity as the z-coordinate.

    Args:
        img: N x N float64 image in [0, 1].
        cell_size: Meters per pixel.
        scale: Multiplier for intensity as z-coordinate.
        threshold_pct: Percentile threshold below which points are filtered out.

    Returns:
        Open3D PointCloud.
    """
    h, w = img.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    points = np.stack([
        (x - w / 2) * cell_size,
        (y - h / 2) * cell_size,
        img.astype(np.float64) * scale
    ], axis=-1)

    mask = img > np.percentile(img, threshold_pct)
    point_coords = points[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_coords)
    return pcd


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
        self.use_direct = config.get("use_direct", False)
        self.level_potential_rotation = config.get("level_potential_rotation", 0.1)
        self.normalization = config.get("normalization", 1)
        self.use_weighted_peak_score = config.get("use_weighted_peak_score", True)
        self.use_phase_correlation = config.get("use_phase_correlation", False)

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
            useHamming=self.use_hamming,
            useDirect=self.use_direct,
            levelPotentialRotation=self.level_potential_rotation,
            normalization=self.normalization,
            usePhaseCorrelation=self.use_phase_correlation
        )

        best_score = 0.0
        best_rot_idx = 0
        best_trans = list_peaks[0].potentialTranslations[0]
        for i, peak in enumerate(list_peaks):
            for trans in peak.potentialTranslations:
                if self.use_weighted_peak_score:
                    score = trans.peakHeight * np.sqrt(peak.potentialRotation.peakCorrelation)
                else:
                    score = trans.peakHeight
                if score > best_score:
                    best_score = score
                    best_rot_idx = i
                    best_trans = trans
        highest_peak = best_score
        peak = list_peaks[best_rot_idx]

        transform = np.eye(4)
        yaw = peak.potentialRotation.angle
        transform[:3, :3] = R.from_euler("z", yaw).as_matrix()
        tx, ty = best_trans.translationSI
        transform[:3, 3] = [tx, -ty, 0.0]

        # Collect all candidate solutions (rotation peaks × translation candidates)
        all_solutions = []
        for rot_peak in list_peaks:
            rot_yaw = rot_peak.potentialRotation.angle
            rot_R = R.from_euler("z", rot_yaw).as_matrix()
            for trans in rot_peak.potentialTranslations:
                t = np.eye(4)
                t[:3, :3] = rot_R
                t[:3, 3] = [trans.translationSI[0], -trans.translationSI[1], 0.0]
                all_solutions.append(t)

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
                "all_solutions": all_solutions,
                "num_solutions": len(all_solutions)
            }
        )


class ICPRegistration(BaseRegistrationMethod):
    """Open3D point-to-point ICP registration.

    Converts images to 3D point clouds (x, y, intensity) and
    runs point-to-point ICP.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._name = "icp"
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D is required for ICPRegistration. Install with: pip install open3d")
        self.max_distance = config.get("icp_max_distance", config.get("max_distance", 0.05))
        self.max_iteration = config.get("icp_max_iteration", config.get("max_iteration", 100))
        self.scale = config.get("icp_scale", config.get("scale", 1.0))
        self.threshold_pct = config.get("icp_threshold_pct", config.get("threshold_pct", 5.0))
        self.initial_guess = config.get("initial_guess", np.eye(4))
        self.voxel_size = config.get("icp_voxel_size", config.get("voxel_size", 0.0))

    def register(self, img1: np.ndarray, img2: np.ndarray,
                 pcd1: np.ndarray = None, pcd2: np.ndarray = None) -> RegistrationResult:
        t0 = time.time()

        if pcd1 is not None and pcd2 is not None:
            source = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(pcd1[:, :3])
            target = o3d.geometry.PointCloud()
            target.points = o3d.utility.Vector3dVector(pcd2[:, :3])
        else:
            cell_size = self.config.get("size_of_pixel", 0.01)
            source = _image_to_pointcloud(img1, cell_size, self.scale, self.threshold_pct)
            target = _image_to_pointcloud(img2, cell_size, self.scale, self.threshold_pct)

        if self.voxel_size > 0:
            source = source.voxel_down_sample(self.voxel_size)
            target = target.voxel_down_sample(self.voxel_size)

        if len(source.points) < 3 or len(target.points) < 3:
            elapsed = time.time() - t0
            return RegistrationResult(
                transform=np.eye(4),
                confidence=0.0,
                method_name="icp",
                computation_time=elapsed,
                metadata={"error": "too few points", "source_points": len(source.points), "target_points": len(target.points)}
            )

        reg = o3d.pipelines.registration.registration_icp(
            source, target, self.max_distance, self.initial_guess,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.max_iteration)
            )

        # Convert Open3D point-cloud frame to vehicle frame (x-forward, y-right user convention)
        # Boreas radar standard: x = r*cos(θ) (forward), y = r*sin(θ) (left for CCW)
        # Our pc frame: pc_x = y_Boreas (left), pc_y = x_Boreas (forward)
        # R_veh = R_pc (invariant for z-axis rotation), t_veh_x = -pc_ty, t_veh_y = -pc_tx
        T_pc = reg.transformation
        transform = np.eye(4)
        transform[:3, :3] = T_pc[:3, :3]
        transform[0, 3] = -T_pc[1, 3]   # veh_x = -pc_y
        transform[1, 3] = -T_pc[0, 3]   # veh_y = -pc_x

        fitness = reg.fitness
        rmse = reg.inlier_rmse
        confidence = fitness / (1.0 + rmse) if rmse > 0 else 0.0
        elapsed = time.time() - t0

        return RegistrationResult(
            transform=transform,
            confidence=confidence,
            method_name="icp",
            computation_time=elapsed,
            metadata={"fitness": fitness, "rmse": rmse, "num_points_source": len(source.points), "num_points_target": len(target.points)}
        )


class FourierMellinRegistration(BaseRegistrationMethod):
    """Fourier-Mellin Transform registration via imreg_dft.

    Uses log-polar FFT for rotation + translation estimation.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._name = "fourier_mellin"

    def register(self, img1: np.ndarray, img2: np.ndarray) -> RegistrationResult:
        import imreg_dft as ird

        t0 = time.time()

        I1 = img1.astype(np.float64)
        I2 = img2.astype(np.float64)

        cell_size = self.config.get("size_of_pixel", 0.01)

        result = ird.similarity(I1, I2)

        angle = float(result["angle"])
        tx, ty = result["tvec"]
        success = float(result.get("success", 0.0))

        transform = np.eye(4)
        transform[:3, :3] = R.from_euler("z", np.radians(angle)).as_matrix()
        transform[:3, 3] = [tx * cell_size, -ty * cell_size, 0.0]

        elapsed = time.time() - t0

        return RegistrationResult(
            transform=transform,
            confidence=success,
            method_name="fourier_mellin",
            computation_time=elapsed,
            metadata={
                "rotation_angle_deg": angle,
                "translation_px": (tx, ty),
            },
        )


class NDT_P2DRegistration(BaseRegistrationMethod):
    """Normal Distributions Transform registration using PCL's NDT via pybind_ndt.

    Falls back to Open3D GeneralizedICP if pybind_ndt is unavailable.
    Uses raw polar point clouds when available (USE_RAW_POINTCLOUD).
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._name = "ndt_p2d"
        self.resolution = config.get("ndt_voxel_size", config.get("voxel_size", 5.0))
        self.max_iteration = config.get("ndt_max_iteration", config.get("max_iteration", 35))
        self.transformation_epsilon = config.get("ndt_transformation_epsilon", config.get("transformation_epsilon", 0.01))
        self.step_size = config.get("ndt_step_size", config.get("step_size", 0.1))
        self.scale = config.get("ndt_scale", config.get("scale", 1.0))
        self.threshold_pct = config.get("ndt_threshold_pct", config.get("threshold_pct", 5.0))

    def register(self, img1: np.ndarray, img2: np.ndarray,
                 pcd1: np.ndarray = None, pcd2: np.ndarray = None) -> RegistrationResult:
        t0 = time.time()
        cell_size = self.config.get("size_of_pixel", 0.01)

        if pcd1 is not None and pcd2 is not None:
            source_pts = np.zeros((len(pcd1), 3), dtype=np.float64)
            source_pts[:, :2] = pcd1[:, :2]
            target_pts = np.zeros((len(pcd2), 3), dtype=np.float64)
            target_pts[:, :2] = pcd2[:, :2]
        else:
            pcd1_o3d = _image_to_pointcloud(img1, cell_size, self.scale, self.threshold_pct)
            pcd2_o3d = _image_to_pointcloud(img2, cell_size, self.scale, self.threshold_pct)
            source_pts = np.asarray(pcd1_o3d.points, dtype=np.float64)
            target_pts = np.asarray(pcd2_o3d.points, dtype=np.float64)
            source_pts[:, 2] = 0.0
            target_pts[:, 2] = 0.0

        if len(source_pts) < 10 or len(target_pts) < 10:
            elapsed = time.time() - t0
            return RegistrationResult(
                transform=np.eye(4), confidence=0.0, method_name="ndt_p2d",
                computation_time=elapsed,
                metadata={"error": "too few points", "source_points": len(source_pts), "target_points": len(target_pts)}
            )

        if PCLNDT_AVAILABLE:
            ndt = pybind_ndt.PCLNDTWrapper()
            initial_guess = self.config.get("initial_guess", np.eye(4)).astype(np.float64)
            result = ndt.align(
                source_pts, target_pts,
                resolution=self.resolution,
                step_size=self.step_size,
                transformation_epsilon=self.transformation_epsilon,
                max_iterations=self.max_iteration,
                initial_guess=initial_guess.ravel()
            )
            T_pc = np.array(result.transformation)
            fitness = result.fitness
            convergence = result.has_converged
            n_iter = result.final_num_iteration
        else:
            # Fallback: Open3D generalized ICP
            if not OPEN3D_AVAILABLE:
                raise ImportError("Neither pybind_ndt nor Open3D available for NDT_P2DRegistration")
            import open3d as o3d
            source_o3d = o3d.geometry.PointCloud()
            source_o3d.points = o3d.utility.Vector3dVector(source_pts)
            target_o3d = o3d.geometry.PointCloud()
            target_o3d.points = o3d.utility.Vector3dVector(target_pts)
            reg = o3d.pipelines.registration.registration_generalized_icp(
                source_o3d, target_o3d, self.resolution, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=self.max_iteration,
                    relative_rmse=self.transformation_epsilon
                )
            )
            T_pc = reg.transformation
            fitness = reg.fitness
            convergence = True
            n_iter = 0

        # Convert PCL/Open3D point-cloud frame to vehicle frame
        # Boreas radar standard: x = r*cos(θ) (forward), y = r*sin(θ) (left for CCW)
        # Our pc frame: pc_x = y_Boreas (left), pc_y = x_Boreas (forward)
        # R_veh = R_pc (invariant for z-axis rotation), t_veh_x = -pc_ty, t_veh_y = -pc_tx
        transform = np.eye(4)
        transform[:3, :3] = T_pc[:3, :3]
        transform[0, 3] = -T_pc[1, 3]   # veh_x = -pc_y
        transform[1, 3] = -T_pc[0, 3]   # veh_y = -pc_x

        confidence = 1.0 / (1.0 + fitness) if fitness > 0 else 0.0
        elapsed = time.time() - t0

        return RegistrationResult(
            transform=transform,
            confidence=confidence,
            method_name="ndt_p2d",
            computation_time=elapsed,
            metadata={
                "fitness": fitness, "converged": convergence, "iterations": n_iter,
                "resolution": self.resolution,
                "num_points_source": len(source_pts), "num_points_target": len(target_pts)
            }
        )


class SIFTRegistration(BaseRegistrationMethod):
    """SIFT feature-based registration with RANSAC.

    Detects SIFT keypoints, matches with FLANN + Lowe ratio test,
    and estimates 2D rotation + translation via RANSAC.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._name = "sift"
        self.nfeatures = config.get("sift_nfeatures", config.get("nfeatures", 0))
        self.n_octave_layers = config.get("sift_n_octave_layers", config.get("n_octave_layers", 3))
        self.contrast_threshold = config.get("sift_contrast_threshold", config.get("contrast_threshold", 0.04))
        self.edge_threshold = config.get("sift_edge_threshold", config.get("edge_threshold", 10))
        self.sigma = config.get("sift_sigma", config.get("sigma", 1.6))
        self.ratio_threshold = config.get("sift_ratio_threshold", config.get("ratio_threshold", 0.75))
        self.ransac_threshold = config.get("sift_ransac_threshold", config.get("ransac_threshold", 3.0))
        self.confidence = config.get("sift_ransac_confidence", config.get("ransac_confidence", 0.99))

    def register(self, img1: np.ndarray, img2: np.ndarray) -> RegistrationResult:
        import cv2
        t0 = time.time()

        img1_u8 = (img1 * 255).astype(np.uint8)
        img2_u8 = (img2 * 255).astype(np.uint8)

        detector = cv2.SIFT_create(
            nfeatures=self.nfeatures,
            nOctaveLayers=self.n_octave_layers,
            contrastThreshold=self.contrast_threshold,
            edgeThreshold=self.edge_threshold,
            sigma=self.sigma
        )
        kp1, des1 = detector.detectAndCompute(img1_u8, None)
        kp2, des2 = detector.detectAndCompute(img2_u8, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            elapsed = time.time() - t0
            return RegistrationResult(
                transform=np.eye(4), confidence=0.0, method_name="sift",
                computation_time=elapsed,
                metadata={"error": "insufficient keypoints", "n1": len(kp1) if kp1 else 0, "n2": len(kp2) if kp2 else 0}
            )

        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

        knn_matches = matcher.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in knn_matches:
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            elapsed = time.time() - t0
            return RegistrationResult(
                transform=np.eye(4), confidence=0.0, method_name="sift",
                computation_time=elapsed,
                metadata={"error": "insufficient good matches", "num_matches": len(good_matches)}
            )

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        affine, inlier_mask = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            confidence=self.confidence
        )

        if affine is None:
            elapsed = time.time() - t0
            return RegistrationResult(
                transform=np.eye(4), confidence=0.0, method_name="sift",
                computation_time=elapsed,
                metadata={"error": "RANSAC failed to estimate transform"}
            )

        n_inliers = np.sum(inlier_mask) if inlier_mask is not None else 0
        angle_rad = np.arctan2(affine[1, 0], affine[0, 0])
        tx_px = affine[0, 2]
        ty_px = affine[1, 2]
        cell_size = self.config.get("size_of_pixel", 0.01)

        transform = np.eye(4)
        transform[:3, :3] = R.from_euler("z", angle_rad).as_matrix()
        transform[:3, 3] = [-tx_px * cell_size, ty_px * cell_size, 0.0]

        confidence = n_inliers / max(len(good_matches), 1)
        elapsed = time.time() - t0

        return RegistrationResult(
            transform=transform, confidence=confidence, method_name="sift",
            computation_time=elapsed,
            metadata={
                "keypoints_source": len(kp1), "keypoints_target": len(kp2),
                "good_matches": len(good_matches), "inliers": n_inliers,
                "rotation_deg": np.degrees(angle_rad), "translation": (tx_px, ty_px)
            }
        )


def _check_surf():
    """Check that SURF is available in the OpenCV build.

    SURF requires opencv-contrib-python and may not be present
    in all OpenCV distributions. Raises ImportError if unavailable.
    """
    import cv2
    if not hasattr(cv2, 'xfeatures2d'):
        raise ImportError(
            "SURF is not available in this OpenCV build. "
            "Install opencv-contrib-python: pip install opencv-contrib-python"
        )


class SURFRegistration(BaseRegistrationMethod):
    """SURF feature-based registration with RANSAC.

    Requires opencv-contrib-python. Fails hard if not available.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._name = "surf"
        self.hessian_threshold = config.get("surf_hessian_threshold", config.get("hessian_threshold", 400))
        self.n_octaves = config.get("surf_n_octaves", config.get("n_octaves", 4))
        self.n_octave_layers = config.get("surf_n_octave_layers", config.get("n_octave_layers", 3))
        self.extended = config.get("surf_extended", config.get("extended", True))
        self.upright = config.get("surf_upright", config.get("upright", False))
        self.ratio_threshold = config.get("surf_ratio_threshold", config.get("ratio_threshold", 0.75))
        self.ransac_threshold = config.get("surf_ransac_threshold", config.get("ransac_threshold", 3.0))
        self.confidence = config.get("surf_ransac_confidence", config.get("ransac_confidence", 0.99))
        # Validate availability at construction time
        _check_surf()

    def register(self, img1: np.ndarray, img2: np.ndarray) -> RegistrationResult:
        import cv2
        t0 = time.time()

        img1_u8 = (img1 * 255).astype(np.uint8)
        img2_u8 = (img2 * 255).astype(np.uint8)

        detector = cv2.xfeatures2d.SURF_create(
            hessianThreshold=self.hessian_threshold,
            nOctaves=self.n_octaves,
            nOctaveLayers=self.n_octave_layers,
            extended=self.extended,
            upright=self.upright
        )
        kp1, des1 = detector.detectAndCompute(img1_u8, None)
        kp2, des2 = detector.detectAndCompute(img2_u8, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            elapsed = time.time() - t0
            return RegistrationResult(
                transform=np.eye(4), confidence=0.0, method_name="surf",
                computation_time=elapsed,
                metadata={"error": "insufficient keypoints", "n1": len(kp1) if kp1 else 0, "n2": len(kp2) if kp2 else 0}
            )

        matcher = cv2.BFMatcher(cv2.NORM_L2)
        knn_matches = matcher.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in knn_matches:
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            elapsed = time.time() - t0
            return RegistrationResult(
                transform=np.eye(4), confidence=0.0, method_name="surf",
                computation_time=elapsed,
                metadata={"error": "insufficient good matches", "num_matches": len(good_matches)}
            )

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        affine, inlier_mask = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            confidence=self.confidence
        )

        if affine is None:
            elapsed = time.time() - t0
            return RegistrationResult(
                transform=np.eye(4), confidence=0.0, method_name="surf",
                computation_time=elapsed,
                metadata={"error": "RANSAC failed to estimate transform"}
            )

        n_inliers = np.sum(inlier_mask) if inlier_mask is not None else 0
        angle_rad = np.arctan2(affine[1, 0], affine[0, 0])
        tx_px = affine[0, 2]
        ty_px = affine[1, 2]
        cell_size = self.config.get("size_of_pixel", 0.01)

        transform = np.eye(4)
        transform[:3, :3] = R.from_euler("z", angle_rad).as_matrix()
        transform[:3, 3] = [-tx_px * cell_size, ty_px * cell_size, 0.0]

        confidence = n_inliers / max(len(good_matches), 1)
        elapsed = time.time() - t0

        return RegistrationResult(
            transform=transform, confidence=confidence, method_name="surf",
            computation_time=elapsed,
            metadata={
                "keypoints_source": len(kp1), "keypoints_target": len(kp2),
                "good_matches": len(good_matches), "inliers": n_inliers,
                "rotation_deg": np.degrees(angle_rad), "translation": (tx_px, ty_px)
            }
        )


class KAZERegistration(BaseRegistrationMethod):
    """KAZE feature-based registration with RANSAC."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._name = "kaze"
        self.extended = config.get("kaze_extended", config.get("extended", False))
        self.upright = config.get("kaze_upright", config.get("upright", False))
        self.threshold = config.get("kaze_threshold", config.get("threshold", 0.001))
        self.n_octaves = config.get("kaze_n_octaves", config.get("n_octaves", 4))
        self.n_octave_layers = config.get("kaze_n_octave_layers", config.get("n_octave_layers", 4))
        self.diffusivity = int(config.get("kaze_diffusivity", config.get("diffusivity", 2)))
        self.ratio_threshold = config.get("kaze_ratio_threshold", config.get("ratio_threshold", 0.75))
        self.ransac_threshold = config.get("kaze_ransac_threshold", config.get("ransac_threshold", 3.0))
        self.confidence = config.get("kaze_ransac_confidence", config.get("ransac_confidence", 0.99))

    def register(self, img1: np.ndarray, img2: np.ndarray) -> RegistrationResult:
        import cv2
        t0 = time.time()

        img1_u8 = (img1 * 255).astype(np.uint8)
        img2_u8 = (img2 * 255).astype(np.uint8)

        detector = cv2.KAZE_create(
            extended=self.extended,
            upright=self.upright,
            threshold=self.threshold,
            nOctaves=self.n_octaves,
            nOctaveLayers=self.n_octave_layers,
            diffusivity=self.diffusivity
        )
        kp1, des1 = detector.detectAndCompute(img1_u8, None)
        kp2, des2 = detector.detectAndCompute(img2_u8, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            elapsed = time.time() - t0
            return RegistrationResult(
                transform=np.eye(4), confidence=0.0, method_name="kaze",
                computation_time=elapsed,
                metadata={"error": "insufficient keypoints", "n1": len(kp1) if kp1 else 0, "n2": len(kp2) if kp2 else 0}
            )

        matcher = cv2.BFMatcher(cv2.NORM_L2)
        knn_matches = matcher.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in knn_matches:
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            elapsed = time.time() - t0
            return RegistrationResult(
                transform=np.eye(4), confidence=0.0, method_name="kaze",
                computation_time=elapsed,
                metadata={"error": "insufficient good matches", "num_matches": len(good_matches)}
            )

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        affine, inlier_mask = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            confidence=self.confidence
        )

        if affine is None:
            elapsed = time.time() - t0
            return RegistrationResult(
                transform=np.eye(4), confidence=0.0, method_name="kaze",
                computation_time=elapsed,
                metadata={"error": "RANSAC failed to estimate transform"}
            )

        n_inliers = np.sum(inlier_mask) if inlier_mask is not None else 0
        angle_rad = np.arctan2(affine[1, 0], affine[0, 0])
        tx_px = affine[0, 2]
        ty_px = affine[1, 2]
        cell_size = self.config.get("size_of_pixel", 0.01)

        transform = np.eye(4)
        transform[:3, :3] = R.from_euler("z", angle_rad).as_matrix()
        transform[:3, 3] = [-tx_px * cell_size, ty_px * cell_size, 0.0]

        confidence = n_inliers / max(len(good_matches), 1)
        elapsed = time.time() - t0

        return RegistrationResult(
            transform=transform, confidence=confidence, method_name="kaze",
            computation_time=elapsed,
            metadata={
                "keypoints_source": len(kp1), "keypoints_target": len(kp2),
                "good_matches": len(good_matches), "inliers": n_inliers,
                "rotation_deg": np.degrees(angle_rad), "translation": (tx_px, ty_px)
            }
        )


class AKAZERegistration(BaseRegistrationMethod):
    """AKAZE feature-based registration with RANSAC.

    Uses binary descriptors with Hamming distance matching.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._name = "akaze"
        self.descriptor_type = config.get("akaze_descriptor_type", config.get("descriptor_type", "MLDB"))
        self.descriptor_size = config.get("akaze_descriptor_size", config.get("descriptor_size", 0))
        self.descriptor_channels = config.get("akaze_descriptor_channels", config.get("descriptor_channels", 3))
        self.threshold = config.get("akaze_threshold", config.get("threshold", 0.001))
        self.n_octaves = config.get("akaze_n_octaves", config.get("n_octaves", 4))
        self.n_octave_layers = config.get("akaze_n_octave_layers", config.get("n_octave_layers", 4))
        self.diffusivity = int(config.get("akaze_diffusivity", config.get("diffusivity", 2)))
        self.ratio_threshold = config.get("akaze_ratio_threshold", config.get("ratio_threshold", 0.75))
        self.ransac_threshold = config.get("akaze_ransac_threshold", config.get("ransac_threshold", 3.0))
        self.confidence = config.get("akaze_ransac_confidence", config.get("ransac_confidence", 0.99))
        # Descriptor type resolved lazily in register() to avoid import-time dependency on cv2

    def register(self, img1: np.ndarray, img2: np.ndarray) -> RegistrationResult:
        import cv2
        t0 = time.time()

        _desc_map = {
            "KAZE": cv2.AKAZE_DESCRIPTOR_KAZE,
            "MLDB": cv2.AKAZE_DESCRIPTOR_MLDB,
        }
        desc_type = _desc_map.get(self.descriptor_type.upper(), cv2.AKAZE_DESCRIPTOR_MLDB)

        img1_u8 = (img1 * 255).astype(np.uint8)
        img2_u8 = (img2 * 255).astype(np.uint8)

        detector = cv2.AKAZE_create(
            descriptor_type=desc_type,
            descriptor_size=self.descriptor_size,
            descriptor_channels=self.descriptor_channels,
            threshold=self.threshold,
            nOctaves=self.n_octaves,
            nOctaveLayers=self.n_octave_layers,
            diffusivity=self.diffusivity
        )
        kp1, des1 = detector.detectAndCompute(img1_u8, None)
        kp2, des2 = detector.detectAndCompute(img2_u8, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            elapsed = time.time() - t0
            return RegistrationResult(
                transform=np.eye(4), confidence=0.0, method_name="akaze",
                computation_time=elapsed,
                metadata={"error": "insufficient keypoints", "n1": len(kp1) if kp1 else 0, "n2": len(kp2) if kp2 else 0}
            )

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn_matches = matcher.knnMatch(des1, des2, k=2)

        good_matches = []
        for m, n in knn_matches:
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            elapsed = time.time() - t0
            return RegistrationResult(
                transform=np.eye(4), confidence=0.0, method_name="akaze",
                computation_time=elapsed,
                metadata={"error": "insufficient good matches", "num_matches": len(good_matches)}
            )

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        affine, inlier_mask = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            confidence=self.confidence
        )

        if affine is None:
            elapsed = time.time() - t0
            return RegistrationResult(
                transform=np.eye(4), confidence=0.0, method_name="akaze",
                computation_time=elapsed,
                metadata={"error": "RANSAC failed to estimate transform"}
            )

        n_inliers = np.sum(inlier_mask) if inlier_mask is not None else 0
        angle_rad = np.arctan2(affine[1, 0], affine[0, 0])
        tx_px = affine[0, 2]
        ty_px = affine[1, 2]
        cell_size = self.config.get("size_of_pixel", 0.01)

        transform = np.eye(4)
        transform[:3, :3] = R.from_euler("z", angle_rad).as_matrix()
        transform[:3, 3] = [-tx_px * cell_size, ty_px * cell_size, 0.0]

        confidence = n_inliers / max(len(good_matches), 1)
        elapsed = time.time() - t0

        return RegistrationResult(
            transform=transform, confidence=confidence, method_name="akaze",
            computation_time=elapsed,
            metadata={
                "keypoints_source": len(kp1), "keypoints_target": len(kp2),
                "good_matches": len(good_matches), "inliers": n_inliers,
                "rotation_deg": np.degrees(angle_rad), "translation": (tx_px, ty_px)
            }
        )


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
RegistrationFactory.register("icp", ICPRegistration)
RegistrationFactory.register("ndt_p2d", NDT_P2DRegistration)
RegistrationFactory.register("fourier_mellin", FourierMellinRegistration)
RegistrationFactory.register("sift", SIFTRegistration)
RegistrationFactory.register("surf", SURFRegistration)
RegistrationFactory.register("kaze", KAZERegistration)
RegistrationFactory.register("akaze", AKAZERegistration)
