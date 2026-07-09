import os
import sys
import time
import copy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from boreasDatasetLoader import load_single_sequence, BoreasSequence

DATA_DIR = "/home/tim-external/dataFolder/radar_boreas"
SEQUENCE_NAME = "boreas-2020-11-26-13-58"
START_FRAME = 3685
MATCHING_STEP = 1
ICP_INTENSITY_THRESHOLD = 0.01
ICP_VOXEL_SIZE = 0.5
ICP_MAX_DISTANCE = 30.0

# --- Load sequence ---
seq = load_single_sequence(DATA_DIR, SEQUENCE_NAME)
print(f"Sequence loaded: {SEQUENCE_NAME}, {seq.length} frames")

idx1, idx2 = START_FRAME, START_FRAME + MATCHING_STEP
print(f"\nFrames: {idx1} -> {idx2}")

# =========================================================
# 1. Debug raw polar data range
# =========================================================
frame1 = seq.sequence.get_radar(idx1)
polar = np.asarray(frame1.polar, dtype=np.float32)

print(f"\n--- Polar data (frame {idx1}) ---")
print(f"  shape: {polar.shape}")
print(f"  dtype original: {np.asarray(frame1.polar).dtype}")
print(f"  as float32 -> min: {polar.min():.4f}, max: {polar.max():.4f}, mean: {polar.mean():.4f}")
print(f"  max > 1.0? {polar.max() > 1.0}")

if polar.max() > 1.0:
    normd = polar / polar.max()
    print(f"  after /max -> min: {normd.min():.6f}, max: {normd.max():.6f}")

azimuths = np.asarray(frame1.azimuths, dtype=np.float32).squeeze()
print(f"  azimuths shape: {azimuths.shape}, range: [{azimuths.min():.4f}, {azimuths.max():.4f}] rad")
resolution = float(frame1.resolution)
print(f"  resolution: {resolution:.6f} m/bin")

# =========================================================
# 2. Point count vs intensity threshold
# =========================================================
print(f"\n--- Point count vs intensity threshold ---")
for thr in [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
    pc1 = seq.get_raw_point_cloud(idx1, intensity_threshold=thr)
    pc2 = seq.get_raw_point_cloud(idx2, intensity_threshold=thr)
    print(f"  thr={thr:.3f}: {len(pc1):7d} pts (f{idx1}), {len(pc2):7d} pts (f{idx2})")

# =========================================================
# 3. Point cloud statistics
# =========================================================
pc1 = seq.get_raw_point_cloud(idx1, intensity_threshold=ICP_INTENSITY_THRESHOLD)
pc2 = seq.get_raw_point_cloud(idx2, intensity_threshold=ICP_INTENSITY_THRESHOLD)
print(f"\n--- Point cloud details (thr={ICP_INTENSITY_THRESHOLD}) ---")
print(f"  Frame {idx1}: {len(pc1)} points, Frame {idx2}: {len(pc2)} points")

for label, pc in [("frame1", pc1), ("frame2", pc2)]:
    print(f"  {label}:")
    print(f"    x: [{pc[:,0].min():7.2f}, {pc[:,0].max():7.2f}]  (span {pc[:,0].max()-pc[:,0].min():.1f} m)")
    print(f"    y: [{pc[:,1].min():7.2f}, {pc[:,1].max():7.2f}]  (span {pc[:,1].max()-pc[:,1].min():.1f} m)")
    print(f"    z: [{pc[:,2].min():.4f}, {pc[:,2].max():.4f}]")

# =========================================================
# 4. Range distribution analysis
# =========================================================
print(f"\n--- Range distribution (frame {idx1}) ---")
ranges = np.sqrt(pc1[:, 0]**2 + pc1[:, 1]**2)
bins = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 80), (80, 120), (120, 200), (200, 999)]
for lo, hi in bins:
    if hi == 999:
        cnt = np.sum(ranges >= lo)
        label = f">={lo}m"
    else:
        cnt = np.sum((ranges >= lo) & (ranges < hi))
        label = f"{lo}-{hi}m"
    pct = 100 * cnt / len(ranges)
    print(f"    {label:>8s}: {cnt:6d} pts ({pct:5.1f}%)")

print(f"\n  Points within 30m: {np.sum(ranges < 30):6d} ({100*np.sum(ranges < 30)/len(ranges):.1f}%)")
print(f"  Points within 50m: {np.sum(ranges < 50):6d} ({100*np.sum(ranges < 50)/len(ranges):.1f}%)")

# =========================================================
# 5. ICP diagnostics (if open3d available)
# =========================================================
try:
    import open3d as o3d
except ImportError:
    print("\nOpen3D not available, skipping ICP tests.")
    sys.exit(0)

print(f"\n{'='*60}")
print("ICP Diagnostics")
print(f"{'='*60}")

def run_icp_debug(pc_src, pc_tgt, voxel_size, max_distance, initial_guess=np.eye(4)):
    """Run ICP and return result with stats."""
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(pc_src[:, :3])
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(pc_tgt[:, :3])

    src_ds = src.voxel_down_sample(voxel_size) if voxel_size > 0 else src
    tgt_ds = tgt.voxel_down_sample(voxel_size) if voxel_size > 0 else tgt

    # --- Correspondence check before ICP ---
    tgt_kdtree = o3d.geometry.KDTreeFlann(tgt_ds)
    src_samples = np.asarray(src_ds.points)
    n_check = min(2000, len(src_samples))
    found = 0
    total_d = 0.0
    for i in range(n_check):
        _, idx, dist2 = tgt_kdtree.search_knn_vector_3d(src_samples[i], 1)
        d = np.sqrt(dist2[0])
        if d < max_distance:
            found += 1
            total_d += d
    avg_d = total_d / found if found > 0 else 0.0

    # --- Run ICP ---
    reg = o3d.pipelines.registration.registration_icp(
        src_ds, tgt_ds, max_distance, initial_guess,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )

    return reg, src_ds, tgt_ds, found, avg_d, n_check

# --- Test with current config ---
print(f"\nConfig: voxel_size={ICP_VOXEL_SIZE}, max_distance={ICP_MAX_DISTANCE}, thr={ICP_INTENSITY_THRESHOLD}")
reg, src_ds, tgt_ds, corr_found, corr_avg_d, n_checked = run_icp_debug(
    pc1, pc2, ICP_VOXEL_SIZE, ICP_MAX_DISTANCE
)

T_pc = reg.transformation
print(f"\n  Before ICP: {n_checked} source points checked")
print(f"    {corr_found} had a target within {ICP_MAX_DISTANCE}m (avg dist {corr_avg_d:.3f}m)")
print(f"  Downsampled: {len(src_ds.points)} src, {len(tgt_ds.points)} tgt")
print(f"  ICP result:")
print(f"    fitness: {reg.fitness:.6f}")
print(f"    inlier_rmse: {reg.inlier_rmse:.6f}")
print(f"    transformation (point-cloud frame):")
print(f"      [{T_pc[0,0]:.6f}, {T_pc[0,1]:.6f}, {T_pc[0,2]:.6f}, {T_pc[0,3]:.6f}]")
print(f"      [{T_pc[1,0]:.6f}, {T_pc[1,1]:.6f}, {T_pc[1,2]:.6f}, {T_pc[1,3]:.6f}]")
print(f"      [{T_pc[2,0]:.6f}, {T_pc[2,1]:.6f}, {T_pc[2,2]:.6f}, {T_pc[2,3]:.6f}]")
print(f"      [{T_pc[3,0]:.6f}, {T_pc[3,1]:.6f}, {T_pc[3,2]:.6f}, {T_pc[3,3]:.6f}]")

# Convert to vehicle frame
tv = np.eye(4)
tv[:3, :3] = T_pc[:3, :3]
tv[0, 3] = -T_pc[1, 3]
tv[1, 3] = -T_pc[0, 3]

gt = seq.get_gt_transform(idx1, idx2)
print(f"  Vehicle frame:")
print(f"    rotation  Est: {np.arctan2(tv[1,0], tv[0,0]):.4f}  GT: {np.arctan2(gt[1,0], gt[0,0]):.4f}")
print(f"    tx        Est: {tv[0,3]:.4f}  GT: {gt[0,3]:.4f}")
print(f"    ty        Est: {tv[1,3]:.4f}  GT: {gt[1,3]:.4f}")

# =========================================================
# 6. Sweep max_distance
# =========================================================
print(f"\n--- Sweep max_distance ---")
for md in [1, 2, 3, 5, 10, 20, 30, 50, 100]:
    reg_md, _, _, cf, cad, nc = run_icp_debug(pc1, pc2, ICP_VOXEL_SIZE, md)
    T = reg_md.transformation
    tx = -T[1, 3]
    ty = -T[0, 3]
    rot = np.arctan2(T[1,0], T[0,0])
    print(f"  md={md:3d}: fit={reg_md.fitness:.4f} rmse={reg_md.inlier_rmse:.4f} "
          f"rx={rot:.4f} tx={tx:.4f} ty={ty:.4f}  (init corr: {cf}/{nc})")

# =========================================================
# 7. Sweep intensity threshold
# =========================================================
print(f"\n--- Sweep intensity threshold (max_dist={ICP_MAX_DISTANCE}) ---")
for thr in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
    pa = seq.get_raw_point_cloud(idx1, intensity_threshold=thr)
    pb = seq.get_raw_point_cloud(idx2, intensity_threshold=thr)
    if len(pa) < 50 or len(pb) < 50:
        print(f"  thr={thr:.3f}: too few pts ({len(pa)}, {len(pb)}), skip")
        continue
    reg_thr, sds, tds, cf, cad, nc = run_icp_debug(pa, pb, ICP_VOXEL_SIZE, ICP_MAX_DISTANCE)
    T = reg_thr.transformation
    tx = -T[1, 3]
    ty = -T[0, 3]
    n_sds = len(np.asarray(sds.points))
    print(f"  thr={thr:.3f}: {len(pa):7d}->{n_sds:5d} pts  fit={reg_thr.fitness:.4f} "
          f"tx={tx:.4f} ty={ty:.4f}  (init corr: {cf}/{nc})")

# =========================================================
# 8. Test with voxel_size=0 (no downsampling)
# =========================================================
print(f"\n--- No voxel downsampling (voxel_size=0) ---")
reg_nv, sds_nv, tds_nv, cf_nv, cad_nv, nc_nv = run_icp_debug(pc1, pc2, 0, ICP_MAX_DISTANCE)
Tnv = reg_nv.transformation
tx_nv = -Tnv[1, 3]
print(f"  voxel_size=0: {len(pc1)}->{len(np.asarray(sds_nv.points))} pts, fit={reg_nv.fitness:.4f}, "
      f"tx={tx_nv:.4f} (init corr: {cf_nv}/{nc_nv})")

# =========================================================
# 9. Compare with cartesian image pipeline
# =========================================================
print(f"\n--- Cartesian image pipeline (ICP_USE_RAW=False) ---")
from boreasRegistrationMethods import _image_to_pointcloud, ICPRegistration
from boreasDatasetLoader import BoreasSequence

N = 128
SIZE_OF_PIXEL = 2.344
SCALE = 1.0
THRESHOLD_PCT = 20.0

img1 = seq.get_cartesian_image(idx1, N, SIZE_OF_PIXEL)
img2 = seq.get_cartesian_image(idx2, N, SIZE_OF_PIXEL)

pcd_cart_src = _image_to_pointcloud(img1, SIZE_OF_PIXEL, SCALE, THRESHOLD_PCT)
pcd_cart_tgt = _image_to_pointcloud(img2, SIZE_OF_PIXEL, SCALE, THRESHOLD_PCT)

print(f"  Cart img pts: {len(np.asarray(pcd_cart_src.points))} src, "
      f"{len(np.asarray(pcd_cart_tgt.points))} tgt")

# Run ICP on cartesian point cloud
for md in [0.5, 1, 2, 5, 10, 20, 30, 50]:
    reg_cart = o3d.pipelines.registration.registration_icp(
        pcd_cart_src, pcd_cart_tgt, md, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )
    Tc = reg_cart.transformation
    ctx = -Tc[1, 3]
    cty = -Tc[0, 3]
    print(f"  md={md:5.1f}: fit={reg_cart.fitness:.4f} rmse={reg_cart.inlier_rmse:.4f} "
          f"tx={ctx:.4f} ty={cty:.4f}")

# =========================================================
# 10. Test sign convention: try both conversion formulas
# =========================================================
print(f"\n--- Frame conversion sign test (thr=0.1) ---")
pa = seq.get_raw_point_cloud(idx1, intensity_threshold=0.1)
pb = seq.get_raw_point_cloud(idx2, intensity_threshold=0.1)
reg10, sds10, _, _, _, _ = run_icp_debug(pa, pb, ICP_VOXEL_SIZE, ICP_MAX_DISTANCE)
T10 = reg10.transformation

print(f"  T_pc[1,3] (pc_ty) = {T10[1,3]:.4f}")
# Test both conversion signs
for sx in [1, -1]:
    for sy in [1, -1]:
        veh_x = sx * T10[1, 3]
        veh_y = sy * T10[0, 3]
        err_x = abs(veh_x - gt[0,3])
        err_y = abs(veh_y - gt[1,3])
        print(f"  veh_x = {sx:+.0f}*pc_ty = {veh_x:+.4f} (err {err_x:.4f}), "
              f"veh_y = {sy:+.0f}*pc_tx = {veh_y:+.4f} (err {err_y:.4f})")

# =========================================================
# 11. Test: does the ICP result LOWER the alignment error?
# =========================================================
print(f"\n--- Verify ICP improves alignment (thr=0.1) ---")
src = o3d.geometry.PointCloud()
src.points = o3d.utility.Vector3dVector(pa[:, :3])
tgt = o3d.geometry.PointCloud()
tgt.points = o3d.utility.Vector3dVector(pb[:, :3])
src_ds = src.voxel_down_sample(ICP_VOXEL_SIZE)
tgt_ds = tgt.voxel_down_sample(ICP_VOXEL_SIZE)

# Error at identity
identity_err = 0.0
tgt_kdtree = o3d.geometry.KDTreeFlann(tgt_ds)
for pt in np.asarray(src_ds.points)[:500]:
    _, idx, d2 = tgt_kdtree.search_knn_vector_3d(pt, 1)
    identity_err += np.sqrt(d2[0])
identity_err /= 500

# Error at ICP result
src_t = copy.deepcopy(src_ds).transform(T10)
src_t_pts = np.asarray(src_t.points)
icp_err = 0.0
tgt_kdtree2 = o3d.geometry.KDTreeFlann(tgt_ds)
for pt in src_t_pts[:500]:
    _, idx, d2 = tgt_kdtree2.search_knn_vector_3d(pt, 1)
    icp_err += np.sqrt(d2[0])
icp_err /= 500

print(f"  Mean nearest-neighbor distance at identity: {identity_err:.4f}m")
print(f"  Mean nearest-neighbor distance after ICP transform: {icp_err:.4f}m")
print(f"  Improvement: {identity_err - icp_err:.4f}m ({(1-icp_err/identity_err)*100:.1f}%)")

# Also test with NEGATIVE pc_ty to see if it aligns better
T_flipped = T10.copy()
T_flipped[0:3, 3] = -T10[0:3, 3]
src_t2 = copy.deepcopy(src_ds).transform(T_flipped)
src_t2_pts = np.asarray(src_t2.points)
flip_err = 0.0
tgt_kdtree3 = o3d.geometry.KDTreeFlann(tgt_ds)
for pt in src_t2_pts[:500]:
    _, idx, d2 = tgt_kdtree3.search_knn_vector_3d(pt, 1)
    flip_err += np.sqrt(d2[0])
flip_err /= 500
print(f"  Mean nearest-neighbor distance with FLIPPED translation: {flip_err:.4f}m")

print(f"\n--- Done ---")
