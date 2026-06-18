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


@dataclass
class RegistrationResult:
    """Unified output from any registration method."""
    transform: np.ndarray           # 4x4 relative transformation
    confidence: float               # peak height / correlation score
    method_name: str
    computation_time: float         # seconds
    metadata: dict = field(default_factory=dict)  # method-specific info


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
            levelPotentialRotation=self.level_potential_rotation
        )

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
        transform[:3, 3] = [tx, -ty, 0.0]

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
    Reference: implementation at plotting_results/registrationFourier/FMT/register.py
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._name = "fourier_mellin"

    def register(self, img1: np.ndarray, img2: np.ndarray) -> RegistrationResult:
        raise NotImplementedError(
            "FourierMellinRegistration not yet implemented. "
            "Plan: port the FMT algorithm from plotting_results/registrationFourier/FMT/register.py."
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
