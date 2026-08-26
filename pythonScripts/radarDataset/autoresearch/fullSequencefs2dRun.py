#!/usr/bin/env python3
################################################################################
# fullSequencefs2dRun.py - FS2D registration over a full Boreas sequence
#
# Runs the FS2D (SOFT) 2D registration over ALL pairs of one Boreas radar
# sequence (starting at frame 0) on NUM_WORKERS parallel processes and writes
# the per-pair results to a CSV in the results/ folder.
#
# Usage:
#     python fullSequencefs2dRun.py
#
# Settings are read from config_fs2d.py in this folder. The output CSV is
# named from the config values (plus an 8-char config hash), so changing a
# setting produces a new file and old results are kept.
################################################################################

import csv
import hashlib
import importlib.util
import os
import sys
import time

# --- Limit per-process BLAS/OpenMP threads BEFORE numpy is imported ---------
# Each pool worker should use exactly one core; the NUM_WORKERS processes
# then cover the machine without oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from multiprocessing import Pool

# --- Paths -------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RADAR_DIR = os.path.dirname(SCRIPT_DIR)  # .../pythonScripts/radarDataset
if RADAR_DIR not in sys.path:
    sys.path.insert(0, RADAR_DIR)
# boreasRegistrationMethods adds the colcon install lib (pybind_registration_2d)
# itself when imported.

from boreasDatasetLoader import load_single_sequence, get_affine_matrix, transform_diff
from boreasRegistrationMethods import RegistrationFactory

CONFIG_FILENAME = "config_fs2d.py"
RESULT_COLUMNS = [
    "prev_frame", "curr_frame",
    "gt_rot_deg", "gt_tx", "gt_ty",
    "est_rot_deg", "est_tx", "est_ty",
    "rot_error_deg", "trans_error_m",
    "confidence", "time_ms", "num_solutions",
]


# ============================================================================
# Config loading
# ============================================================================

def load_config():
    """Import the config file from this folder; values are read at runtime."""
    cfg_path = os.path.join(SCRIPT_DIR, CONFIG_FILENAME)
    if not os.path.isfile(cfg_path):
        sys.exit(f"ERROR: config file not found: {cfg_path}")
    spec = importlib.util.spec_from_file_location("fs2d_full_sequence_config", cfg_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def config_hash(cfg) -> str:
    """Short hash over all config values (any change -> new output file)."""
    keys = sorted(k for k in vars(cfg) if k.isupper() and not k.startswith("_"))
    payload = "\n".join(f"{k}={vars(cfg)[k]}" for k in keys)
    return hashlib.sha1(payload.encode()).hexdigest()[:8]


def apply_circular_mask(image: np.ndarray) -> np.ndarray:
    """Zero out pixels outside the inscribed circle of a square image."""
    N = image.shape[0]
    Y, X = np.ogrid[:N, :N]
    mask = (X - N // 2) ** 2 + (Y - N // 2) ** 2 <= (N // 2) ** 2
    return image * mask


# ============================================================================
# Summary statistics
# ============================================================================

def compute_summary(rows, rot_thresh_deg, trans_thresh_m):
    """Aggregate per-pair results into a run summary.

    Outlier definitions:
      - rotation outlier:    |rot_error_deg| > rot_thresh_deg
      - translation outlier: trans_error_m    > trans_thresh_m
    A pair can count in both. Stats (mean/std/median, std with ddof=1 as in
    aggregate_benchmark_results.py) are computed over inliers = pairs that
    fail NEITHER criterion.
    """
    inlier_rot, inlier_trans = [], []
    rot_outliers = 0
    trans_outliers = 0
    for r in rows:
        is_rot_out = abs(r["rot_error_deg"]) > rot_thresh_deg
        is_trans_out = r["trans_error_m"] > trans_thresh_m
        if is_rot_out:
            rot_outliers += 1
        if is_trans_out:
            trans_outliers += 1
        if not is_rot_out and not is_trans_out:
            inlier_rot.append(r["rot_error_deg"])
            inlier_trans.append(r["trans_error_m"])

    def _stats(vals):
        a = np.asarray(vals, dtype=np.float64)
        a = a[~np.isnan(a)]
        if len(a) == 0:
            return float("nan"), float("nan"), float("nan")
        return (float(np.mean(a)), float(np.std(a, ddof=1)), float(np.median(a)))

    rot_mean, rot_std, rot_med = _stats(inlier_rot)
    trans_mean, trans_std, trans_med = _stats(inlier_trans)
    return {
        "rot_mean_deg": rot_mean,
        "rot_std_deg": rot_std,
        "rot_median_deg": rot_med,
        "trans_mean_m": trans_mean,
        "trans_std_m": trans_std,
        "trans_median_m": trans_med,
        "rot_outlier_count": rot_outliers,
        "trans_outlier_count": trans_outliers,
        "num_pairs": len(rows),
        "num_inliers": len(inlier_rot),
    }


def summary_table_lines(summary):
    """Human-readable (header, data) row for the summary table."""
    header = ("Method | Rot. err. (deg) mean\u00b1std | Rot median | "
              "Trans. err. (m) mean\u00b1std | Trans median | "
              "Rot outliers | Trans outliers")
    data = (f"fs2d   | {summary['rot_mean_deg']:6.2f} \u00b1 {summary['rot_std_deg']:<5.2f} "
            f"| {summary['rot_median_deg']:6.2f} "
            f"| {summary['trans_mean_m']:6.2f} \u00b1 {summary['trans_std_m']:<5.2f} "
            f"| {summary['trans_median_m']:6.2f} "
            f"| {summary['rot_outlier_count']}/{summary['num_pairs']:<4d} "
            f"| {summary['trans_outlier_count']}/{summary['num_pairs']}")
    return header, data


def summary_numeric_lines(summary):
    """Machine-readable '# key: value' lines for the CSV metadata section."""
    return [
        ("summary_rot_err_mean_deg", summary["rot_mean_deg"]),
        ("summary_rot_err_std_deg", summary["rot_std_deg"]),
        ("summary_rot_err_median_deg", summary["rot_median_deg"]),
        ("summary_trans_err_mean_m", summary["trans_mean_m"]),
        ("summary_trans_err_std_m", summary["trans_std_m"]),
        ("summary_trans_err_median_m", summary["trans_median_m"]),
        ("summary_rot_outlier_count", summary["rot_outlier_count"]),
        ("summary_trans_outlier_count", summary["trans_outlier_count"]),
        ("summary_num_inliers", summary["num_inliers"]),
    ]


# ============================================================================
# Worker processes
# ============================================================================

_WORKER = {}


def worker_init(data_dir, sequence_name, method_config, round_images):
    """Called once per pool worker: load sequence + create FS2D method."""
    _WORKER["round_images"] = round_images
    _WORKER["seq"] = load_single_sequence(data_dir, sequence_name)
    _WORKER["method"] = RegistrationFactory.create("fs2d", method_config)
    print(f"[Worker {os.getpid()}] sequence '{sequence_name}' loaded "
          f"({_WORKER['seq'].length} frames), method ready")


def process_pair(pair):
    """Register one pair (prev_idx, curr_idx) and return the result row."""
    prev_idx, curr_idx = pair
    seq = _WORKER["seq"]
    method = _WORKER["method"]
    cfg = method.config
    N = cfg["N"]
    size_of_pixel = cfg["size_of_pixel"]

    row = {"prev_frame": prev_idx, "curr_frame": curr_idx, "status": "OK",
           "error": ""}
    try:
        img1 = seq.get_cartesian_image(prev_idx, N, size_of_pixel)
        img2 = seq.get_cartesian_image(curr_idx, N, size_of_pixel)
        if _WORKER["round_images"]:
            img1 = apply_circular_mask(img1)
            img2 = apply_circular_mask(img2)

        gt_transform = seq.get_gt_transform(prev_idx, curr_idx)
        result = method.register(img1, img2)
        est_transform = result.transform

        # Errors use the same pixel-frame convention as viewBoreasPairs /
        # boreasBenchmark (transform_diff on get_affine_matrix outputs).
        gt_affine = get_affine_matrix(gt_transform)
        est_affine = get_affine_matrix(est_transform)
        trans_error, rot_error = transform_diff(gt_affine, est_affine)

        row.update({
            "gt_rot_deg": float(np.degrees(np.arctan2(gt_transform[1, 0], gt_transform[0, 0]))),
            "gt_tx": float(gt_transform[0, 3]),
            "gt_ty": float(gt_transform[1, 3]),
            "est_rot_deg": float(np.degrees(np.arctan2(est_transform[1, 0], est_transform[0, 0]))),
            "est_tx": float(est_transform[0, 3]),
            "est_ty": float(est_transform[1, 3]),
            "rot_error_deg": float(rot_error),
            "trans_error_m": float(np.linalg.norm(trans_error)),
            "confidence": float(result.confidence),
            "time_ms": float(result.computation_time * 1000.0),
            "num_solutions": int(result.metadata.get("num_solutions", 0)),
        })
    except Exception as e:
        row["status"] = "FAIL"
        row["error"] = f"{type(e).__name__}: {e}"
        row.update({c: float("nan") for c in RESULT_COLUMNS if c not in ("prev_frame", "curr_frame")})
    finally:
        # Free cached polar data — pyboreas caches it and never frees otherwise.
        try:
            seq.sequence.radar_frames[prev_idx].unload_data()
            seq.sequence.radar_frames[curr_idx].unload_data()
        except Exception:
            pass
    return row


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()
    cfg = load_config()

    size_of_pixel = (2.0 * cfg.RADIUS) / cfg.N
    method_config = {
        "N": cfg.N,
        "radius": cfg.RADIUS,
        "size_of_pixel": size_of_pixel,
        "use_clahe": cfg.USE_CLAHE,
        "use_hamming": cfg.USE_HAMMING,
        "potential_for_necessary_peak": cfg.POTENTIAL_NECCESSARY_FOR_PEAK,
        "multiple_radii": cfg.MULTIPLE_RADII,
        "use_gauss": cfg.USE_GAUSS,
        "use_direct": cfg.USE_DIRECT,
        "num_angles": cfg.NUM_ANGLES,
        "r_min": cfg.R_MIN,
        "r_max": cfg.R_MAX,
        "level_potential_rotation": cfg.LEVEL_POTENTIAL_ROTATION,
        "normalization": cfg.NORMALIZATION,
        "use_weighted_peak_score": cfg.USE_WEIGHTED_PEAK_SCORE,
        "use_phase_correlation": cfg.USE_PHASE_CORRELATION,
        "debug": cfg.DEBUG_MODE,
    }

    # Build the pair list (start frame is always 0).
    seq = load_single_sequence(cfg.DATA_DIR, cfg.SEQUENCE_NAME)
    total_frames = seq.length
    del seq  # workers load their own copies

    end = total_frames if cfg.MAX_FRAMES is None else min(total_frames, cfg.MAX_FRAMES)
    if end < cfg.MATCHING_STEP:
        sys.exit(f"ERROR: no pairs possible (end={end}, matching_step={cfg.MATCHING_STEP})")
    pairs = [(i - cfg.MATCHING_STEP, i) for i in range(cfg.MATCHING_STEP, end, cfg.MATCHING_STEP)]

    # Output file — name derived from config values + config hash.
    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    px = int(size_of_pixel * 100)
    out_name = (f"fs2d_{cfg.SEQUENCE_NAME}_N{cfg.N}_r{cfg.RADIUS:g}_"
                f"s{cfg.MATCHING_STEP}_p{px}_h{config_hash(cfg)}.csv")
    out_path = os.path.join(results_dir, out_name)

    print("=" * 80)
    print("FS2D full-sequence Boreas registration")
    print("=" * 80)
    print(f"Sequence : {cfg.SEQUENCE_NAME} ({total_frames} frames, start=0)")
    print(f"Matching : every {cfg.MATCHING_STEP}th frame -> {len(pairs)} pairs"
          + (f" (capped at MAX_FRAMES={cfg.MAX_FRAMES})" if cfg.MAX_FRAMES else ""))
    print(f"Grid     : N={cfg.N}, radius={cfg.RADIUS} m, pixel_size={size_of_pixel:.3f} m")
    print(f"Workers  : {cfg.NUM_WORKERS}")
    print(f"Output   : {out_path}")
    print()

    # Run all pairs in parallel.
    t0 = time.time()
    with Pool(processes=cfg.NUM_WORKERS, initializer=worker_init,
              initargs=(cfg.DATA_DIR, cfg.SEQUENCE_NAME, method_config, cfg.ROUND)) as pool:
        rows_ok, failures = [], []
        done = 0
        for row in pool.imap_unordered(process_pair, pairs):
            done += 1
            if row["status"] == "FAIL":
                failures.append(row)
                print(f"[{done:4d}/{len(pairs)}] pair {row['prev_frame']:5d}->{row['curr_frame']:5d} "
                      f"FAILED: {row['error']}")
            else:
                rows_ok.append(row)
                print(f"[{done:4d}/{len(pairs)}] pair {row['prev_frame']:5d}->{row['curr_frame']:5d} "
                      f"rot_err={row['rot_error_deg']:7.3f} deg "
                      f"trans_err={row['trans_error_m']:8.3f} m "
                      f"conf={row['confidence']:.3f} time={row['time_ms']:.0f} ms")
    run_elapsed = time.time() - t0

    rows_ok.sort(key=lambda r: r["prev_frame"])
    summary = compute_summary(rows_ok, cfg.OUTLIER_ROT_THRESH_DEG, cfg.OUTLIER_TRANS_THRESH_M)

    # Write CSV: summary table + metadata header (# key: value) + data rows.
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        # --- Summary at the top ---
        if rows_ok:
            header, data = summary_table_lines(summary)
            writer.writerow(["# " + "-" * 78])
            writer.writerow(["# " + header])
            writer.writerow(["# " + data])
            writer.writerow(["# " + "-" * 78])
            for key, val in summary_numeric_lines(summary):
                writer.writerow([f"# {key}: {val:.6f}" if isinstance(val, float) else f"# {key}: {val}"])
        # --- Run metadata ---
        writer.writerow([f"# sequence_name: {cfg.SEQUENCE_NAME}"])
        writer.writerow([f"# total_frames: {total_frames}"])
        writer.writerow([f"# num_pairs_total: {len(pairs)}"])
        writer.writerow([f"# num_pairs_ok: {len(rows_ok)}"])
        writer.writerow([f"# num_pairs_failed: {len(failures)}"])
        writer.writerow([f"# run_start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t_start))}"])
        writer.writerow([f"# wall_time_s: {run_elapsed:.1f}"])
        writer.writerow([f"# config_file: {CONFIG_FILENAME}"])
        for key in sorted(k for k in vars(cfg) if k.isupper() and not k.startswith("_")):
            writer.writerow([f"# config_{key}: {vars(cfg)[key]}"])
        writer.writerow([])
        writer.writerow(RESULT_COLUMNS)
        for row in rows_ok:
            writer.writerow([row[c] for c in RESULT_COLUMNS])

    # Failure log (only if any pairs failed).
    if failures:
        fail_path = os.path.join(results_dir, out_name.replace(".csv", "_failures.log"))
        with open(fail_path, "w") as f:
            for r in failures:
                f.write(f"pair {r['prev_frame']}->{r['curr_frame']}: {r['error']}\n")
        print(f"\nFailures logged to: {fail_path}")

    # Summary.
    print()
    print("-" * 80)
    print(f"Done in {run_elapsed:.1f}s: {len(rows_ok)}/{len(pairs)} pairs OK, "
          f"{len(failures)} failed "
          f"({run_elapsed / max(len(rows_ok), 1):.2f}s per pair wall)")
    if rows_ok:
        header, data = summary_table_lines(summary)
        print()
        print(header)
        print(data)
        print()
        print(f"Avg conf       : {np.mean([r['confidence'] for r in rows_ok]):.3f}")
        print(f"Avg time       : {np.mean([r['time_ms'] for r in rows_ok]):.1f} ms/pair")
    print(f"CSV: {out_path}")


if __name__ == "__main__":
    main()