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
Boreas dataset loader and preprocessing utilities.

Loads Boreas radar sequences and provides cartesian images, point clouds,
and ground truth transforms for registration methods.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

from sdk.radar import load_radar, radar_polar_to_cartesian
from pyboreas import BoreasDataset
from pyboreas.data.sequence import Sequence


@dataclass
class BoreasSequence:
    """Wrapper around a Boreas dataset sequence for registration workflows."""

    sequence: object
    data_dir: str

    @property
    def length(self) -> int:
        return len(self.sequence.radar_frames)

    def get_cartesian_image(self, index: int, N: int, size_of_pixel: float) -> np.ndarray:
        """Convert polar radar scan to cartesian image.

        Args:
            index: Index of radar frame.
            N: Image grid size (N x N).
            size_of_pixel: Meters per pixel.

        Returns:
            Normalized cartesian image as float64 array in [0, 1].
        """
        frame = self.sequence.get_radar(index)
        img = frame.polar_to_cart(
            cart_resolution=size_of_pixel,
            cart_pixel_width=N,
            in_place=False,
        )
        return img

    def get_point_cloud(self, index: int, N: int, size_of_pixel: float, threshold: float = 0.01) -> np.ndarray:
        """Extract 2D point cloud from cartesian image.

        Args:
            index: Index of radar frame.
            N: Image grid size.
            size_of_pixel: Meters per pixel.
            threshold: Reflectivity threshold for point inclusion.

        Returns:
            Nx2 array of (x, y) coordinates in meters.
        """
        img = self.get_cartesian_image(index, N, size_of_pixel)
        ys, xs = np.where(img >= threshold)
        points = np.column_stack([
            (xs - N / 2) * size_of_pixel,
            (ys - N / 2) * size_of_pixel,
        ])
        return points

    def get_gt_transform(self, prev_index: int, curr_index: int) -> np.ndarray:
        """Compute ground truth relative transformation between two frames.

        Args:
            prev_index: Index of previous frame.
            curr_index: Index of current frame.

        Returns:
            4x4 relative transformation matrix.
        """
        return np.matmul(
            np.linalg.inv(np.asarray(self.sequence.get_radar(prev_index).pose, dtype=np.float64)),
            np.asarray(self.sequence.get_radar(curr_index).pose, dtype=np.float64),
        )


def load_sequence(data_dir: str, sequence: int) -> BoreasSequence:
    """Load a Boreas dataset sequence by index.

    Args:
        data_dir: Path to Boreas radar data directory.
        sequence: Sequence number (index into all sequences).

    Returns:
        BoreasSequence instance.
    """
    bd = BoreasDataset(data_dir, split=None, verbose=False)
    seq = bd.sequences[sequence]
    return BoreasSequence(sequence=seq, data_dir=data_dir)


def load_single_sequence(data_dir: str, sequence_name: str) -> BoreasSequence:
    """Load a single Boreas dataset sequence by name without scanning all sequences.

    Args:
        data_dir: Path to Boreas radar data directory.
        sequence_name: Sequence ID string, e.g. 'boreas-2020-11-26-13-58'.

    Returns:
        BoreasSequence instance.
    """
    seq = Sequence(data_dir, [sequence_name])
    return BoreasSequence(sequence=seq, data_dir=data_dir)


def get_affine_matrix(input_matrix: np.ndarray,
                       pixel_size: float = 1.0,
                       img_size: int = 0) -> np.ndarray:
    """Extract 2D affine transform from 4x4 matrix.

    Args:
        input_matrix: 4x4 world transform (translation in meters).
        pixel_size: Meters per pixel for converting translation to pixel units.
        img_size: Image dimension N for rotation center compensation.
                  0 = no compensation (backward compatible).

    Returns:
        3x3 affine matrix suitable for cv2.warpPerspective.
    """
    input_matrix = np.linalg.inv(input_matrix)
    result = np.eye(3)
    result[:2, :2] = input_matrix[:2, :2]
    result[0, 2] = -input_matrix[1, 3] / pixel_size
    result[1, 2] = input_matrix[0, 3] / pixel_size
    # Rotation center compensation (rotate around image center)
    if img_size > 0:
        c = img_size / 2.0
        T_c = np.eye(3)
        T_c[:2, 2] = [c, c]
        T_c_inv = np.eye(3)
        T_c_inv[:2, 2] = [-c, -c]
        result = T_c @ result @ T_c_inv
    return result


def transform_diff(matrix1: np.ndarray, matrix2: np.ndarray) -> tuple:
    """Compute translation and rotation difference between two 3x3 affine matrices.

    Args:
        matrix1: First 3x3 affine matrix.
        matrix2: Second 3x3 affine matrix.

    Returns:
        Tuple of (translation_diff, rotation_angle_diff_degrees).
    """
    t1 = np.array([matrix1[0, 2], matrix1[1, 2]])
    t2 = np.array([matrix2[0, 2], matrix2[1, 2]])
    trans_diff = t2 - t1

    r1 = matrix1[:2, :2]
    r2 = matrix2[:2, :2]
    angle_diff = np.degrees(
        np.arctan2(r2[1, 0], r2[0, 0]) - np.arctan2(r1[1, 0], r1[0, 0])
    )
    return trans_diff, angle_diff


def matrix_to_transform(matrix: np.ndarray) -> np.ndarray:
    """Convert 4x4 matrix to [x, y, z, roll, pitch, yaw]."""
    translation = matrix[:3, 3]
    rotation = R.from_matrix(matrix[:3, :3])
    euler = rotation.as_euler("xyz", degrees=False)
    return np.asarray([translation[0], translation[1], translation[2],
                       euler[0], euler[1], euler[2]])


def fuse_images(images_over_time: List[np.ndarray],
                estimated_transformations: List[np.ndarray],
                pixel_size: float = 1.0,
                img_size: int = 0) -> Optional[np.ndarray]:
    """Fuse multiple images using their absolute transformation matrices.

    Args:
        images_over_time: List of cartesian images.
        estimated_transformations: List of 4x4 absolute transform matrices.
        pixel_size: Meters per pixel for converting translation to pixel units.
        img_size: Image dimension N for rotation center compensation.

    Returns:
        Fused map as uint8 array, or None if no images.
    """
    assert len(images_over_time) == len(estimated_transformations)
    if not images_over_time:
        return None

    min_x = float("inf")
    max_x = -float("inf")
    min_y = float("inf")
    max_y = -float("inf")

    for i in range(len(images_over_time)):
        img = images_over_time[i]
        tmat = get_affine_matrix(estimated_transformations[i], pixel_size, img_size)
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
        t = get_affine_matrix(tmat, pixel_size, img_size)
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
