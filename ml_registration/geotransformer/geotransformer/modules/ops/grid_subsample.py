import torch
import numpy as np
import open3d as o3d

try:
    import importlib
    ext_module = importlib.import_module('geotransformer.ext')
    HAS_CPP_EXTENSION = True
except (ImportError, ModuleNotFoundError):
    HAS_CPP_EXTENSION = False


def _grid_subsampling_open3d_fallback(points, lengths, voxel_size):
    """Open3D-based fallback for grid subsampling."""
    if isinstance(points, torch.Tensor):
        points_np = points.cpu().numpy()
    else:
        points_np = points
    
    if isinstance(lengths, torch.Tensor):
        lengths_np = lengths.cpu().numpy()
    else:
        lengths_np = lengths
    
    s_points_list = []
    s_lengths_list = []
    
    start_idx = 0
    for length in lengths_np:
        cloud_points = points_np[start_idx:start_idx + length]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud_points)
        pcd_down = pcd.voxel_down_sample(voxel_size)
        s_points = np.asarray(pcd_down.points).astype(np.float32)
        s_points_list.append(s_points)
        s_lengths_list.append(len(s_points))
        start_idx += length
    
    s_points = np.vstack(s_points_list).astype(np.float32)
    s_lengths = torch.LongTensor(s_lengths_list)
    
    if isinstance(points, torch.Tensor):
        s_points = torch.from_numpy(s_points)
    
    return s_points, s_lengths


def grid_subsample(points, lengths, voxel_size):
    """Grid subsampling in stack mode.

    This function is implemented on CPU.

    Args:
        points (Tensor): stacked points. (N, 3)
        lengths (Tensor): number of points in the stacked batch. (B,)
        voxel_size (float): voxel size.

    Returns:
        s_points (Tensor): stacked subsampled points (M, 3)
        s_lengths (Tensor): numbers of subsampled points in the batch. (B,)
    """
    if HAS_CPP_EXTENSION:
        s_points, s_lengths = ext_module.grid_subsampling(points, lengths, voxel_size)
    else:
        s_points, s_lengths = _grid_subsampling_open3d_fallback(points, lengths, voxel_size)
    
    return s_points, s_lengths
