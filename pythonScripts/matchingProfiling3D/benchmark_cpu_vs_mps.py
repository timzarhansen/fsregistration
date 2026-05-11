#!/usr/bin/python3
"""
Benchmark script to compare CPU vs MPS performance for HybridPoint model.

Usage:
    python benchmark_cpu_vs_mps.py
"""

import os
import sys
import torch
import time
import numpy as np
import gc
import copy
import open3d as o3d
import transforms3d.quaternions as quat
import transforms3d.euler as euler

# Add paths (same as testingHybridPointOnPredatorData.py)
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
ml_registration_path = os.path.join(root_dir, 'ml_registration')

geo_base_path = os.path.join(ml_registration_path, 'geotransformer')
sys.path.insert(0, geo_base_path)

from geotransformer.utils.torch import release_cuda, to_device

geo_exp_path = os.path.join(geo_base_path, 'experiments', 'geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn')
sys.path.insert(0, geo_exp_path)

from config import make_cfg as make_geo_cfg

sys.path.insert(0, ml_registration_path)

from hybridpoint_wrapper.data import registration_collate_fn_stack_mode
from hybridpoint_wrapper.model import create_model

from dataloader_utils import PredatorDataLoader


def check_device_fallback(output_dict, expected_device):
    """Recursively check if any tensor fell back to CPU instead of expected device."""
    fallbacks = []
    for key, value in output_dict.items():
        if isinstance(value, torch.Tensor):
            if value.device != expected_device:
                fallbacks.append(f"  {key}: expected {expected_device}, got {value.device}")
        elif isinstance(value, dict):
            fallbacks.extend(check_device_fallback(value, expected_device))
    return fallbacks


def add_gaussian_noise_to_pointcloud(pcd, mean=0.0, std=0.01, seed=None):
    points = np.asarray(pcd.points)
    if seed is not None:
        np.random.seed(seed)
    pcdNew = o3d.geometry.PointCloud()
    pcdNew.points = o3d.utility.Vector3dVector(points + np.random.normal(mean, std, points.shape))
    return pcdNew


def add_salt_pepper_noise(pcd, percentage, min_x, max_x, min_y, max_y, min_z, max_z):
    points = np.asarray(pcd.points)
    mask = np.random.rand(len(points)) < percentage
    points[mask] = np.random.uniform(
        [min_x, min_y, min_z],
        [max_x, max_y, max_z],
        size=(np.sum(mask), 3)
    )
    pcdNew = o3d.geometry.PointCloud()
    pcdNew.points = o3d.utility.Vector3dVector(points)
    return pcdNew


def prepare_sample(loader, index):
    """Load and prepare a single sample from the dataset."""
    inputs = loader.get_by_index(index)

    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(inputs["src_pcd_raw"])
    pcd2.points = o3d.utility.Vector3dVector(inputs["tgt_pcd_raw"])

    meanNoise = 0.01
    percentageNoise = 0.01

    pcd1_noisy = add_gaussian_noise_to_pointcloud(pcd1, mean=0.0, std=meanNoise, seed=None)
    pcd2_noisy = add_gaussian_noise_to_pointcloud(pcd2, mean=0.0, std=meanNoise, seed=None)

    pcd1_noisy = add_salt_pepper_noise(
        pcd1_noisy, percentageNoise,
        np.min(pcd1.points, axis=0)[0], np.max(pcd1.points, axis=0)[0],
        np.min(pcd1.points, axis=0)[1], np.max(pcd1.points, axis=0)[1],
        np.min(pcd1.points, axis=0)[2], np.max(pcd1.points, axis=0)[2]
    )
    pcd2_noisy = add_salt_pepper_noise(
        pcd2_noisy, percentageNoise,
        np.min(pcd2.points, axis=0)[0], np.max(pcd2.points, axis=0)[0],
        np.min(pcd2.points, axis=0)[1], np.max(pcd2.points, axis=0)[1],
        np.min(pcd2.points, axis=0)[2], np.max(pcd2.points, axis=0)[2]
    )

    gtTransformation = np.eye(4)
    gtTransformation[:3, :3] = inputs["rot"]
    gtTransformation[:3, 3] = np.squeeze(np.asarray(inputs["trans"]))

    ref_points = np.asarray(pcd2_noisy.points).astype(np.float32)
    src_points = np.asarray(pcd1_noisy.points).astype(np.float32)
    ref_feats = np.ones_like(ref_points[:, :1]).astype(np.float32)
    src_feats = np.ones_like(src_points[:, :1]).astype(np.float32)

    data_dict = {
        "ref_points": ref_points,
        "src_points": src_points,
        "ref_feats": ref_feats,
        "src_feats": src_feats,
        "transform": gtTransformation.astype(np.float32)
    }
    return data_dict


def run_benchmark(model, data_dict, device, geo_cfg, neighbor_limits, voxel_size, search_radius, num_repetitions):
    """Run benchmark on a single sample for a given device."""
    model.to(device)
    model.eval()

    full_times = []
    model_times = []
    fallback_count = 0

    for rep in range(num_repetitions):
        gc.collect()
        if device.type == 'mps':
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

        # Full pipeline timing
        t0 = time.perf_counter()

        data_dict_processed = registration_collate_fn_stack_mode(
            [data_dict],
            num_stages=geo_cfg.backbone.num_stages,
            voxel_size=voxel_size,
            search_radius=search_radius,
            neighbor_limits=neighbor_limits
        )

        if device.type in ('cuda', 'mps'):
            data_dict_processed = to_device(data_dict_processed, device)

        # Model-only timing
        t1 = time.perf_counter()

        with torch.no_grad():
            output_dict = model(data_dict_processed)

        t2 = time.perf_counter()

        if device.type in ('cuda', 'mps'):
            output_dict = release_cuda(output_dict)

        t3 = time.perf_counter()

        full_times.append(t3 - t0)
        model_times.append(t2 - t1)

        if device.type == 'mps':
            fallbacks = check_device_fallback(output_dict, device)
            if fallbacks:
                fallback_count += 1

    return full_times, model_times, fallback_count


def median_std(times):
    return np.median(times), np.std(times)


def main():
    num_samples = 5
    num_warmup = 2
    num_repetitions = 3

    config_path = os.path.join(script_dir, 'configFiles', 'predatorNothingMac.yaml')
    loader = PredatorDataLoader(config_path, split='val')

    geo_cfg = make_geo_cfg()
    geo_cfg.coarse_matching.num_correspondences = 512

    weights_path = os.path.join(ml_registration_path, 'hybridpoint/weights_for_hybrid/3dmatch.tar')
    if not os.path.exists(weights_path):
        print(f"Error: HybridPoint pretrained weights not found at {weights_path}")
        sys.exit(1)

    device_cpu = torch.device('cpu')
    model = create_model(geo_cfg).to(device_cpu)

    try:
        state_dict = torch.load(weights_path, map_location=device_cpu)
        model.load_state_dict(state_dict["model"], strict=False)
        print(f"Loaded HybridPoint pretrained weights")
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)

    neighbor_limits = [38, 36, 36, 38]
    voxel_size = 0.025
    search_radius = geo_cfg.backbone.init_radius

    # Detect available devices
    devices_to_test = [device_cpu]
    if torch.backends.mps.is_available():
        devices_to_test.append(torch.device('mps'))
        print("MPS is available")
    else:
        print("MPS is NOT available on this system")

    print(f"\n{'='*60}")
    print(f"CPU vs MPS Benchmark")
    print(f"{'='*60}")
    print(f"Config:     configFiles/predatorNothing.yaml")
    print(f"Dataset:    val split, noise: low")
    print(f"Samples:    {num_samples}")
    print(f"Warmup:     {num_warmup}")
    print(f"Repetitions: {num_repetitions}")
    print(f"Devices:    {', '.join(str(d) for d in devices_to_test)}")
    print(f"{'='*60}\n")

    # Prepare samples
    print(f"Preparing {num_samples} samples...")
    samples = []
    for i in range(num_samples):
        data_dict = prepare_sample(loader, i)
        samples.append(data_dict)
    print(f"Done. Running benchmark...\n")

    # Benchmark each device
    results = {}
    for device in devices_to_test:
        device_label = str(device).upper()
        print(f"--- Benchmarking on {device_label} ---")

        all_full_times = []
        all_model_times = []
        total_fallbacks = 0

        # Warmup
        for i in range(num_warmup):
            try:
                _, _, fb = run_benchmark(
                    model, samples[i], device,
                    geo_cfg, neighbor_limits, voxel_size, search_radius, 1
                )
                if device.type == 'mps' and fb:
                    total_fallbacks += 1
            except Exception as e:
                print(f"  Warmup sample {i} failed: {e}")

        # Timed runs
        for i, data_dict in enumerate(samples):
            try:
                full_times, model_times, fb = run_benchmark(
                    model, data_dict, device,
                    geo_cfg, neighbor_limits, voxel_size, search_radius, num_repetitions
                )
                all_full_times.append(full_times)
                all_model_times.append(model_times)
                if device.type == 'mps' and fb:
                    total_fallbacks += fb
                print(f"  Sample {i}: full={np.median(full_times):.3f}s model={np.median(model_times):.3f}s")
            except Exception as e:
                print(f"  Sample {i} failed: {e}")

        results[device] = {
            'full_times': all_full_times,
            'model_times': all_model_times,
            'fallbacks': total_fallbacks
        }
        print()

    # Print summary
    print(f"{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")

    cpu_full_med = None
    cpu_model_med = None

    for device, res in results.items():
        device_label = str(device).upper()
        full_meds = [np.mean(t) for t in res['full_times']]
        model_meds = [np.mean(t) for t in res['model_times']]

        if not full_meds or not model_meds:
            print(f"{device_label}: No valid results")
            continue

        full_median, full_std = np.median(full_meds), np.std(full_meds)
        model_median, model_std = np.median(model_meds), np.std(model_meds)

        print(f"\n{device_label}:")
        print(f"  Full pipeline:  {full_median:.3f}s ± {full_std:.3f}s (median ± std)")
        print(f"  Model only:     {model_median:.3f}s ± {model_std:.3f}s")

        if device.type == 'mps':
            print(f"  Fallbacks:      {res['fallbacks']}/{num_samples * num_repetitions} runs had CPU fallback")

        if device.type == 'cpu':
            cpu_full_med = full_median
            cpu_model_med = model_median

    # Speedup comparison
    mps_results = results.get(torch.device('mps'))
    if mps_results and cpu_model_med:
        mps_model_meds = [np.mean(t) for t in mps_results['model_times']]
        if mps_model_meds:
            mps_model_median = np.median(mps_model_meds)
            speedup = cpu_model_med / mps_model_median if mps_model_median > 0 else float('inf')
            print(f"\n{'='*60}")
            print(f"Speedup (CPU / MPS): {speedup:.1f}x")
            print(f"{'='*60}")

            if speedup > 1.2:
                print(f"\nRecommendation: MPS is faster on this workload.")
            elif speedup > 0.8:
                print(f"\nRecommendation: MPS and CPU are comparable. No clear winner.")
            else:
                print(f"\nRecommendation: CPU is faster. MPS may be falling back to CPU frequently.")

    print()


if __name__ == '__main__':
    main()
