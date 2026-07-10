"""
NDT Parameter Diagnostic — intensity-as-z sweep.

Tests PCL NDT from identity on a consecutive pair (~3m motion).
Sweeps resolution × z_scale to find a working combination.

Usage:
    python diagnoseNDT.py
"""

import os
import sys
import time
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
_install_lib = "/home/tim-external/ros_ws/install/fsregistration/lib/fsregistration"
sys.path.insert(0, _install_lib)
_pyboreas_path = "/home/tim-external/ros_ws/install/asrl-pyboreas/lib/python3.12/site-packages"
if os.path.isdir(_pyboreas_path):
    sys.path.insert(0, _pyboreas_path)

from boreasDatasetLoader import load_single_sequence, get_affine_matrix, transform_diff
from boreasRegistrationMethods import RegistrationFactory

# ============================================================================
# CONFIG
# ============================================================================
DATA_DIR = "/home/tim-external/dataFolder/radar_boreas"
SEQUENCE_NAME = "boreas-2020-11-26-13-58"
N = 128
RADIUS = 150.0
SIZE_OF_PIXEL = (2.0 * RADIUS) / N
INTENSITY_THRESHOLD = 0.3

# Test pair: consecutive frames ~3m motion
IDX1, IDX2 = 3685, 3686

# Sweep parameters
NDT_RESOLUTIONS = [2.0, 5.0, 10.0, 20.0, 50.0]
NDT_Z_SCALES    = [0.0, 5.0, 10.0, 25.0, 50.0, 100.0, 150.0]
NDT_STEP_SIZES  = [0.1, 0.5, 1.0]

# ============================================================================
# HELPERS
# ============================================================================

def build_ndt_config(resolution, z_scale, step_size=0.1, max_iter=50):
    return {
        "size_of_pixel": SIZE_OF_PIXEL,
        "ndt_voxel_size": resolution,
        "ndt_max_iteration": max_iter,
        "ndt_transformation_epsilon": 0.01,
        "ndt_step_size": step_size,
        "ndt_z_scale": z_scale,
        "initial_guess": np.eye(4),
    }


def build_icp_config(max_distance=30.0, max_iter=200):
    return {
        "size_of_pixel": SIZE_OF_PIXEL,
        "icp_max_distance": max_distance,
        "icp_max_iteration": max_iter,
        "icp_scale": 1.0,
        "icp_threshold_pct": 20.0,
        "icp_voxel_size": 0.5,
        "initial_guess": np.eye(4),
    }


def evaluate(result, gt_affine):
    est_affine = get_affine_matrix(result.transform)
    gt_error = transform_diff(gt_affine, est_affine)
    gt_trans, gt_rot = gt_error
    return abs(gt_rot), np.linalg.norm(gt_trans), result.confidence, result.metadata


# ============================================================================
# MAIN
# ============================================================================

def main():
    seq = load_single_sequence(DATA_DIR, SEQUENCE_NAME)
    print(f"Sequence: {seq.length} scans")
    print(f"N={N}, pixel_size={SIZE_OF_PIXEL:.3f}m, radius={RADIUS}m")

    img1 = seq.get_cartesian_image(IDX1, N, SIZE_OF_PIXEL)
    img2 = seq.get_cartesian_image(IDX2, N, SIZE_OF_PIXEL)
    raw1 = seq.get_raw_point_cloud(IDX1, INTENSITY_THRESHOLD)
    raw2 = seq.get_raw_point_cloud(IDX2, INTENSITY_THRESHOLD)
    gt = seq.get_gt_transform(IDX1, IDX2)
    gt_affine = get_affine_matrix(gt)
    gt_yaw = np.arctan2(gt[1, 0], gt[0, 0])
    print(f"Pair {IDX1}→{IDX2}: tx={gt[0,3]:.3f}m ty={gt[1,3]:.3f}m yaw={gt_yaw:.4f}rad ({np.degrees(gt_yaw):.2f}°)")
    print(f"\nRaw points: source={len(raw1)}, target={len(raw2)}\n")

    # -------------------------------------------------------------------
    # TEST 1: Resolution × z_scale sweep from identity
    # -------------------------------------------------------------------
    print("=" * 100)
    print("TEST 1: Resolution × z_scale sweep  (from IDENTITY initial guess)")
    print("=" * 100)
    print(f"{'z_scale':>8} | {'res=2m':>28} {'res=5m':>28} {'res=10m':>28} {'res=20m':>28} {'res=50m':>28}")
    print(f"{'─'*8}─┼─{'─'*28}─{'─'*28}─{'─'*28}─{'─'*28}─{'─'*28}")

    best = {"err": float("inf"), "res": None, "z": None, "step": 0.1}

    for zs in NDT_Z_SCALES:
        row = f"{zs:>8.1f} |"
        for res in NDT_RESOLUTIONS:
            cfg = build_ndt_config(res, zs)
            method = RegistrationFactory.create("ndt_p2d", cfg)
            t0 = time.time()
            result = method.register(img1, img2, pcd1=raw1, pcd2=raw2)
            elapsed = time.time() - t0
            rot_err, trans_err, conf, meta = evaluate(result, gt_affine)
            ok = trans_err < 1.5
            tag = "✓" if ok else " "
            if ok and trans_err < best["err"]:
                best = {"err": trans_err, "res": res, "z": zs, "step": 0.1}
            row += f" {tag}{trans_err:>6.3f}m {rot_err:>5.2f}°{elapsed*1000:>6.0f}ms |"
        print(row)

    print(f"\nBest (identity init): res={best['res']}m, z_scale={best['z']}, err={best['err']:.3f}m")

    # -------------------------------------------------------------------
    # TEST 2: Refine best params with step_size sweep
    # -------------------------------------------------------------------
    if best['res'] is not None:
        print("\n" + "=" * 100)
        print(f"TEST 2: Step size sweep at best params  (res={best['res']}m, z_scale={best['z']})")
        print("=" * 100)
        print(f"{'step_size':>10} | {'trans_err':>12} {'rot_err':>10} {'time(ms)':>10} {'iter':>6} {'fitness':>10} {'converged':>10}")
        for ss in NDT_STEP_SIZES:
            cfg = build_ndt_config(best['res'], best['z'], step_size=ss, max_iter=100)
            method = RegistrationFactory.create("ndt_p2d", cfg)
            t0 = time.time()
            result = method.register(img1, img2, pcd1=raw1, pcd2=raw2)
            elapsed = time.time() - t0
            rot_err, trans_err, conf, meta = evaluate(result, gt_affine)
            print(f"{ss:>10.1f} | {trans_err:>12.4f} {rot_err:>10.4f} {elapsed*1000:>10.0f} "
                  f"{meta.get('iterations','?'):>6} {meta.get('fitness',0):>10.2f} {meta.get('converged','?'):>10}")

        # -------------------------------------------------------------------
        # TEST 3: Best params, but with CARTESIAN IMAGE (no raw data)
        # -------------------------------------------------------------------
        print("\n" + "=" * 100)
        print(f"TEST 3: Cartestian image path (no raw data) at best params")
        print("=" * 100)
        cfg = build_ndt_config(best['res'], best['z'], step_size=best.get('step', 0.1))
        method = RegistrationFactory.create("ndt_p2d", cfg)
        t0 = time.time()
        result = method.register(img1, img2)  # No pcd1/pcd2 → cartesian image path
        elapsed = time.time() - t0
        rot_err, trans_err, conf, meta = evaluate(result, gt_affine)
        print(f"  Cartestian image: trans_err={trans_err:.4f}m, rot_err={rot_err:.4f}°, "
              f"time={elapsed*1000:.0f}ms, fitness={meta.get('fitness',0):.2f}, "
              f"iter={meta.get('iterations','?')}")

        # -------------------------------------------------------------------
        # TEST 4: Multi-resolution chain at best z_scale
        # -------------------------------------------------------------------
        print("\n" + "=" * 100)
        print(f"TEST 4: Multi-resolution chain  (z_scale={best['z']}, coarse→fine)")
        print("=" * 100)
        chain_res = [50.0, 20.0, 5.0]
        guess = np.eye(4)
        for cres in chain_res:
            cfg = build_ndt_config(cres, best['z'], step_size=0.5, max_iter=30)
            cfg["initial_guess"] = guess
            # Need to pre-transform points for coarse-to-fine
            # NDT uses identity internally, so transform source points manually
            src = np.zeros((len(raw1), 3), dtype=np.float64)
            src[:, :2] = raw1[:, :2]
            if best['z'] > 0:
                src[:, 2] = raw1[:, 2] * best['z']
            # Apply current guess
            tmp = np.ones((len(src), 4))
            tmp[:, :3] = src
            xformed = (guess @ tmp.T).T[:, :3]

            method = RegistrationFactory.create("ndt_p2d", cfg)
            t0 = time.time()
            # We can't pass pre-transformed points through the API directly,
            # so just run NDT and see if the chain helps
            result = method.register(img1, img2, pcd1=raw1, pcd2=raw2)
            elapsed = time.time() - t0
            guess = result.transform @ guess  # chaining
            rot_err, trans_err, conf, meta = evaluate(result, gt_affine)
            print(f"  res={cres:.0f}m: trans_err={trans_err:.4f}m rot_err={rot_err:.4f}° "
                  f"time={elapsed*1000:.0f}ms iter={meta.get('iterations','?')}")

    # -------------------------------------------------------------------
    # BASELINE: ICP and FS2D for comparison
    # -------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("BASELINE: Other methods from identity")
    print("=" * 100)
    for name in ["fs2d", "icp", "fourier_mellin"]:
        cfg = {"N": N, "size_of_pixel": SIZE_OF_PIXEL, "initial_guess": np.eye(4),
               "use_clahe": True, "use_hamming": True, "use_direct": True,
               "level_potential_rotation": 0.001, "potential_for_necessary_peak": 0.1,
               "normalization": 1, "use_phase_correlation": False,
               "icp_max_distance": 30.0, "icp_max_iteration": 200,
               "icp_scale": 1.0, "icp_threshold_pct": 20.0, "icp_voxel_size": 0.5,
               "fm_highpass": True}
        method = RegistrationFactory.create(name, cfg)
        t0 = time.time()
        sig = __import__('inspect').signature(method.register)
        if "pcd1" in sig.parameters:
            result = method.register(img1, img2, pcd1=raw1, pcd2=raw2)
        else:
            result = method.register(img1, img2)
        elapsed = time.time() - t0
        rot_err, trans_err, conf, _ = evaluate(result, gt_affine)
        print(f"  {name:<16} trans_err={trans_err:.4f}m rot_err={rot_err:.4f}° "
              f"conf={conf:.4f} time={elapsed*1000:.0f}ms")

    # -------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    if best['err'] < 1.5:
        print(f"  NDT WORKS from identity with intensity-as-z!")
        print(f"  Best: resolution={best['res']}m, z_scale={best['z']}, trans_err={best['err']:.3f}m")
    else:
        print(f"  NDT does NOT converge from identity even with intensity-as-z.")
        print(f"  Best: resolution={best['res']}m, z_scale={best['z']}, trans_err={best['err']:.3f}m")
        print(f"  Consecutive motion (~3m) exceeds NDT's convergence basin regardless of params.")
    print()


if __name__ == "__main__":
    main()
