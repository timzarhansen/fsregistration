import torch
import numpy as np
import open3d as o3d

try:
    import importlib
    ext_module = importlib.import_module('geotransformer.ext')
    HAS_CPP_EXTENSION = True
except (ImportError, ModuleNotFoundError):
    HAS_CPP_EXTENSION = False


def _radius_neighbors_open3d_fallback(q_points, s_points, q_lengths, s_lengths, radius, neighbor_limit):
    """Open3D-based fallback for radius neighbors search using KDTreeFlann."""
    if isinstance(q_points, torch.Tensor):
        q_points_np = q_points.cpu().numpy()
    else:
        q_points_np = q_points
    
    if isinstance(s_points, torch.Tensor):
        s_points_np = s_points.cpu().numpy()
    else:
        s_points_np = s_points
    
    if isinstance(q_lengths, torch.Tensor):
        q_lengths_np = q_lengths.cpu().numpy()
    else:
        q_lengths_np = q_lengths
    
    if isinstance(s_lengths, torch.Tensor):
        s_lengths_np = s_lengths.cpu().numpy()
    else:
        s_lengths_np = s_lengths
    
    all_samples_neighbors = []
    
    q_start = 0
    s_start = 0
    for q_len, s_len in zip(q_lengths_np, s_lengths_np):
        q_cloud = q_points_np[q_start:q_start + q_len]
        s_cloud = s_points_np[s_start:s_start + s_len]
        
        cloud_pcd = o3d.geometry.PointCloud()
        cloud_pcd.points = o3d.utility.Vector3dVector(s_cloud)
        cloud_tree = o3d.geometry.KDTreeFlann(cloud_pcd)
        
        cloud_neighbor_indices = []
        for point in q_cloud:
            [_, idx, _] = cloud_tree.search_radius_vector_3d(point, radius)
            global_idx = [s_start + i for i in idx]
            cloud_neighbor_indices.append(global_idx)
        
        all_samples_neighbors.append({
            'neighbors': cloud_neighbor_indices,
            'q_len': q_len,
            's_len': s_len,
            's_start': s_start
        })
        
        q_start += q_len
        s_start += s_len
    
    if neighbor_limit > 0:
        global_max = neighbor_limit
    else:
        global_max = 0
        for sample in all_samples_neighbors:
            for n in sample['neighbors']:
                global_max = max(global_max, len(n))
    
    neighbor_indices_list = []
    for sample in all_samples_neighbors:
        q_len = sample['q_len']
        s_len = sample['s_len']
        s_start = sample['s_start']
        neighbors = sample['neighbors']
        
        indices = np.full((q_len, global_max), s_start + s_len, dtype=np.int64)
        for i, n in enumerate(neighbors):
            limit = min(len(n), global_max)
            indices[i, :limit] = n[:limit]
        
        neighbor_indices_list.append(indices)
    
    neighbor_indices = np.vstack(neighbor_indices_list).astype(np.int64)
    
    if isinstance(q_points, torch.Tensor):
        neighbor_indices = torch.from_numpy(neighbor_indices)
    
    return neighbor_indices


def radius_search(q_points, s_points, q_lengths, s_lengths, radius, neighbor_limit):
    r"""Computes neighbors for a batch of q_points and s_points, apply radius search (in stack mode).

    This function is implemented on CPU.

    Args:
        q_points (Tensor): the query points (N, 3)
        s_points (Tensor): the support points (M, 3)
        q_lengths (Tensor): the list of lengths of batch elements in q_points
        s_lengths (Tensor): the list of lengths of batch elements in s_points
        radius (float): maximum distance of neighbors
        neighbor_limit (int): maximum number of neighbors

    Returns:
        neighbors (Tensor): the k nearest neighbors of q_points in s_points (N, k).
            Filled with M if there are less than k neighbors.
    """
    if HAS_CPP_EXTENSION:
        neighbor_indices = ext_module.radius_neighbors(q_points, s_points, q_lengths, s_lengths, radius)
    else:
        neighbor_indices = _radius_neighbors_open3d_fallback(q_points, s_points, q_lengths, s_lengths, radius, neighbor_limit)
    
    if neighbor_limit > 0:
        neighbor_indices = neighbor_indices[:, :neighbor_limit]
    
    return neighbor_indices
