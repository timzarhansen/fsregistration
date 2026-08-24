#!/usr/bin/env python3
"""
Aggregate benchmark results for the BremenMSS 2D radar dataset.

For every subdirectory with a results.csv:
  - Reconstructs the vehicle trajectory from frame-to-frame transforms
    and writes it as path.csv (estimated and ground truth paths)

For subdirectories with N >= 256 only:
  - Groups all sequences by registration method
  - Computes aggregate error statistics per method
  - Writes aggregated_bremen_results.csv

Usage:
    python aggregate_results_bremen.py
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np


# ============================================================================
# Configuration
# ============================================================================
INPUT_FOLDER = Path(__file__).resolve().parent / "ResultsRadar_DFKI" / "bremenmss2d"
OUTPUT_PATH = INPUT_FOLDER / "aggregated_bremen_results.csv"
OUTPUT_PATH_ALL = INPUT_FOLDER / "aggregated_bremen_results_all.csv"

MIN_N = 256
OUTLIER_ROT_THRESH_DEG = 10.0
OUTLIER_TRANS_THRESH_M = 4.0
MIN_GT_TRANS_M = 0.01
MIN_GT_MOTION = 1e-6       # skip runs where ALL pairs have zero GT

# Only process these sequences (matches seq number in dir names).
# Set to None to process all sequences.
SEQUENCES = [1, 3, 4, 9, 11]


def has_valid_gt(rows: list[dict]) -> bool:
    """Return True if at least one pair has measurable GT motion."""
    for r in rows:
        gt_tx = float(r.get("gt_tx_m", 0))
        gt_ty = float(r.get("gt_ty_m", 0))
        if abs(gt_tx) + abs(gt_ty) >= MIN_GT_MOTION:
            return True
    return False

# ── Directory name parser ──────────────────────────────────────────────────
DIR_PATTERN = re.compile(r"seq(\d+)_(.+)_N(\d+)_p(\d+)_s(\d+)")


def parse_dir_name(name: str) -> dict | None:
    """Extract seq, method, N, p, s from a directory name like
    'seq00_fourier_mellin_N256_p109_s3'."""
    m = DIR_PATTERN.match(name)
    if not m:
        return None
    return {
        "seq": int(m.group(1)),
        "method": m.group(2),
        "N": int(m.group(3)),
        "p": int(m.group(4)),
        "s": int(m.group(5)),
    }


# ── CSV helpers ────────────────────────────────────────────────────────────

def read_data_rows(filepath: Path) -> list[dict]:
    """Read results.csv, skipping comment lines."""
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            break
        reader = csv.DictReader(f, fieldnames=line.strip().split(","))
        return list(reader)


def numeric_cols(row: dict) -> dict:
    """Extract and compute numeric error values from a single CSV row."""
    keys = ["rot_error_deg", "trans_error_m",
            "best_rot_error_deg", "best_trans_error_m",
            "gt_tx_m", "gt_ty_m"]
    out = {}
    for k in keys:
        try:
            out[k] = float(row[k])
        except (ValueError, KeyError):
            out[k] = np.nan

    # Use absolute rotation error for statistics
    out["rot_error_deg"] = abs(out["rot_error_deg"])
    out["best_rot_error_deg"] = abs(out["best_rot_error_deg"])

    # Normalised odometry metrics (avoid div-by-near-zero)
    gt_trans_norm = np.sqrt(out["gt_tx_m"] ** 2 + out["gt_ty_m"] ** 2)
    if gt_trans_norm >= MIN_GT_TRANS_M:
        out["trans_err_pct"] = out["trans_error_m"] / gt_trans_norm * 100.0
        out["rot_err_per_m"] = out["rot_error_deg"] / gt_trans_norm
        out["best_trans_err_pct"] = out["best_trans_error_m"] / gt_trans_norm * 100.0
        out["best_rot_err_per_m"] = out["best_rot_error_deg"] / gt_trans_norm
    else:
        out["trans_err_pct"] = np.nan
        out["rot_err_per_m"] = np.nan
        out["best_trans_err_pct"] = np.nan
        out["best_rot_err_per_m"] = np.nan

    return out


def compute_stats(values: list[float]) -> dict:
    """Mean / std / median, ignoring NaN."""
    a = np.array(values, dtype=np.float64)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan")}
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)),
        "median": float(np.median(a)),
    }


# ── Per-subdirectory stats (same schema as the original script) ────────────

def process_subdirectory(subdir: Path, info: dict | None, rows: list[dict]) -> dict:
    """Compute per-subdirectory aggregate statistics (one row per run)."""
    num_pairs_failed = 0
    csv_path = subdir / "results.csv"
    with open(csv_path, "r") as f:
        for line in f:
            if not line.startswith("#"):
                break
            if "num_pairs_failed" in line:
                try:
                    num_pairs_failed = int(line.split(":")[-1].strip())
                except (ValueError, IndexError):
                    pass

    rot, trans, best_rot, best_trans = [], [], [], []
    trans_pct, rot_per_m, best_trans_pct, best_rot_per_m = [], [], [], []
    for r in rows:
        nc = numeric_cols(r)
        rot.append(nc["rot_error_deg"])
        trans.append(nc["trans_error_m"])
        best_rot.append(nc["best_rot_error_deg"])
        best_trans.append(nc["best_trans_error_m"])
        trans_pct.append(nc["trans_err_pct"])
        rot_per_m.append(nc["rot_err_per_m"])
        best_trans_pct.append(nc["best_trans_err_pct"])
        best_rot_per_m.append(nc["best_rot_err_per_m"])

    rot_s = compute_stats(rot)
    trans_s = compute_stats(trans)
    best_rot_s = compute_stats(best_rot)
    best_trans_s = compute_stats(best_trans)
    trans_pct_s = compute_stats(trans_pct)
    rot_per_m_s = compute_stats(rot_per_m)
    best_trans_pct_s = compute_stats(best_trans_pct)
    best_rot_per_m_s = compute_stats(best_rot_per_m)

    outlier_count = sum(
        1 for r, t in zip(rot, trans)
        if abs(r) > OUTLIER_ROT_THRESH_DEG or t > OUTLIER_TRANS_THRESH_M
    )
    outlier_best_count = sum(
        1 for r, t in zip(best_rot, best_trans)
        if abs(r) > OUTLIER_ROT_THRESH_DEG or t > OUTLIER_TRANS_THRESH_M
    )

    row = {
        "dir_name": subdir.name,
        "total_pairs": len(rows),
        "num_pairs_failed": num_pairs_failed,
        "rot_mean_deg": rot_s["mean"],
        "rot_std_deg": rot_s["std"],
        "rot_median_deg": rot_s["median"],
        "trans_mean_m": trans_s["mean"],
        "trans_std_m": trans_s["std"],
        "trans_median_m": trans_s["median"],
        "best_rot_mean_deg": best_rot_s["mean"],
        "best_rot_std_deg": best_rot_s["std"],
        "best_rot_median_deg": best_rot_s["median"],
        "best_trans_mean_m": best_trans_s["mean"],
        "best_trans_std_m": best_trans_s["std"],
        "best_trans_median_m": best_trans_s["median"],
        "trans_err_pct_mean": trans_pct_s["mean"],
        "trans_err_pct_std": trans_pct_s["std"],
        "trans_err_pct_median": trans_pct_s["median"],
        "rot_err_deg_per_m_mean": rot_per_m_s["mean"],
        "rot_err_deg_per_m_std": rot_per_m_s["std"],
        "rot_err_deg_per_m_median": rot_per_m_s["median"],
        "best_trans_err_pct_mean": best_trans_pct_s["mean"],
        "best_trans_err_pct_std": best_trans_pct_s["std"],
        "best_trans_err_pct_median": best_trans_pct_s["median"],
        "best_rot_err_deg_per_m_mean": best_rot_per_m_s["mean"],
        "best_rot_err_deg_per_m_std": best_rot_per_m_s["std"],
        "best_rot_err_deg_per_m_median": best_rot_per_m_s["median"],
        "outlier_count": outlier_count,
        "outlier_best_count": outlier_best_count,
    }

    if info is not None:
        row["seq"] = info["seq"]
        row["method"] = info["method"]
        row["N"] = info["N"]
        row["p"] = info["p"]
        row["s"] = info["s"]

    return row


# ── Path reconstruction ────────────────────────────────────────────────────

def reconstruct_path(rows: list[dict]) -> list[dict]:
    """Reconstruct estimated and ground-truth trajectories.

    Each row is a frame-to-frame transform. The path is accumulated
    by applying (rot, tx, ty) sequentially:

        x_{i+1} = x_i + cos(θ_i)·tx - sin(θ_i)·ty
        y_{i+1} = y_i + sin(θ_i)·tx + cos(θ_i)·ty
        θ_{i+1} = θ_i + rot

    Returns a list of dicts, one per frame, starting from the first
    prev_frame at (0, 0, 0°).
    """
    if not rows:
        return []

    sorted_rows = sorted(rows, key=lambda r: int(r["pair_idx"]))
    first_prev = int(sorted_rows[0]["prev_frame"])

    est_x, est_y = 0.0, 0.0
    gt_x, gt_y = 0.0, 0.0
    est_heading_deg = 0.0
    gt_heading_deg = 0.0

    path = [{
        "frame": first_prev,
        "est_x": est_x,
        "est_y": est_y,
        "est_heading_deg": est_heading_deg,
        "gt_x": gt_x,
        "gt_y": gt_y,
        "gt_heading_deg": gt_heading_deg,
    }]

    for row in sorted_rows:
        est_rot = float(row["est_rot_deg"])
        est_tx = float(row["est_tx_m"])
        est_ty = float(row["est_ty_m"])
        gt_rot = float(row["gt_rot_deg"])
        gt_tx = float(row["gt_tx_m"])
        gt_ty = float(row["gt_ty_m"])
        curr_frame = int(row["curr_frame"])

        # Estimated pose update
        est_rad = np.radians(est_heading_deg)
        est_x += np.cos(est_rad) * est_tx - np.sin(est_rad) * est_ty
        est_y += np.sin(est_rad) * est_tx + np.cos(est_rad) * est_ty
        est_heading_deg += est_rot

        # Ground truth pose update
        gt_rad = np.radians(gt_heading_deg)
        gt_x += np.cos(gt_rad) * gt_tx - np.sin(gt_rad) * gt_ty
        gt_y += np.sin(gt_rad) * gt_tx + np.cos(gt_rad) * gt_ty
        gt_heading_deg += gt_rot

        path.append({
            "frame": curr_frame,
            "est_x": est_x,
            "est_y": est_y,
            "est_heading_deg": est_heading_deg,
            "gt_x": gt_x,
            "gt_y": gt_y,
            "gt_heading_deg": gt_heading_deg,
        })

    return path


def write_path_csv(subdir: Path, rows: list[dict]) -> None:
    """Write reconstructed trajectory to path.csv inside *subdir*."""
    path_rows = reconstruct_path(rows)
    if not path_rows:
        return
    columns = [
        "frame", "est_x", "est_y", "est_heading_deg",
        "gt_x", "gt_y", "gt_heading_deg",
    ]
    csv_path = subdir / "path.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(path_rows)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    if not INPUT_FOLDER.is_dir():
        print(f"ERROR: {INPUT_FOLDER} is not a directory")
        sys.exit(1)

    all_subdirs = sorted(d for d in INPUT_FOLDER.iterdir() if d.is_dir())
    if not all_subdirs:
        print(f"No subdirectories found in {INPUT_FOLDER}")
        sys.exit(1)

    # ── Phase 1: Generate path.csv for every valid subdirectory ──
    path_count = 0
    path_skipped = 0
    for sd in all_subdirs:
        csv_path = sd / "results.csv"
        if not csv_path.is_file():
            path_skipped += 1
            continue
        info = parse_dir_name(sd.name)
        if info is not None and SEQUENCES is not None and info["seq"] not in SEQUENCES:
            path_skipped += 1
            continue
        rows = read_data_rows(csv_path)
        if not rows:
            path_skipped += 1
            continue
        write_path_csv(sd, rows)
        path_count += 1

    print(f"path.csv: {path_count} written, {path_skipped} skipped (no results.csv)")

    # ── Phase 2: Per-sequence & per-method aggregation (N >= MIN_N only) ──
    all_results = []   # one row per subdirectory (→ aggregated_bremen_results_all.csv)
    method_data = {}   # method_name -> pooled error values (→ aggregated_bremen_results.csv)

    for sd in all_subdirs:
        info = parse_dir_name(sd.name)
        if info is None:
            continue
        if info["N"] < MIN_N:
            continue
        if SEQUENCES is not None and info["seq"] not in SEQUENCES:
            continue

        csv_path = sd / "results.csv"
        if not csv_path.is_file():
            continue
        rows = read_data_rows(csv_path)
        if not rows:
            continue

        # Skip runs with no ground truth
        if not has_valid_gt(rows):
            continue

        # Per-subdirectory row
        row = process_subdirectory(sd, info, rows)
        all_results.append(row)

        # Accumulate into method pool
        method = info["method"]
        if method not in method_data:
            method_data[method] = {
                "rot": [], "trans": [], "best_rot": [], "best_trans": [],
                "trans_pct": [], "rot_per_m": [], "best_trans_pct": [], "best_rot_per_m": [],
                "sequences": set(),
            }

        d = method_data[method]
        d["sequences"].add(info["seq"])
        for r in rows:
            nc = numeric_cols(r)
            d["rot"].append(nc["rot_error_deg"])
            d["trans"].append(nc["trans_error_m"])
            d["best_rot"].append(nc["best_rot_error_deg"])
            d["best_trans"].append(nc["best_trans_error_m"])
            d["trans_pct"].append(nc["trans_err_pct"])
            d["rot_per_m"].append(nc["rot_err_per_m"])
            d["best_trans_pct"].append(nc["best_trans_err_pct"])
            d["best_rot_per_m"].append(nc["best_rot_err_per_m"])

    if not method_data:
        print("No N>=256 results found — nothing to aggregate")
        sys.exit(1)

    # ── Write per-subdirectory file (aggregated_bremen_results_all.csv) ──
    all_columns = ["dir_name", "seq", "method", "N", "p", "s",
                   "total_pairs", "num_pairs_failed",
                   "rot_mean_deg", "rot_std_deg", "rot_median_deg",
                   "trans_mean_m", "trans_std_m", "trans_median_m",
                   "best_rot_mean_deg", "best_rot_std_deg", "best_rot_median_deg",
                   "best_trans_mean_m", "best_trans_std_m", "best_trans_median_m",
                   "trans_err_pct_mean", "trans_err_pct_std", "trans_err_pct_median",
                   "rot_err_deg_per_m_mean", "rot_err_deg_per_m_std", "rot_err_deg_per_m_median",
                   "best_trans_err_pct_mean", "best_trans_err_pct_std", "best_trans_err_pct_median",
                   "best_rot_err_deg_per_m_mean", "best_rot_err_deg_per_m_std", "best_rot_err_deg_per_m_median",
                   "outlier_count", "outlier_best_count"]
    with open(OUTPUT_PATH_ALL, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"All per-sequence results ({len(all_results)} runs) -> {OUTPUT_PATH_ALL}")

    # ── Write per-method file (aggregated_bremen_results.csv) ──
    results = []
    for method in sorted(method_data.keys()):
        d = method_data[method]
        rot, trans = d["rot"], d["trans"]
        best_rot, best_trans = d["best_rot"], d["best_trans"]
        trans_pct, rot_per_m = d["trans_pct"], d["rot_per_m"]
        best_trans_pct, best_rot_per_m = d["best_trans_pct"], d["best_rot_per_m"]

        rot_s = compute_stats(rot)
        trans_s = compute_stats(trans)
        best_rot_s = compute_stats(best_rot)
        best_trans_s = compute_stats(best_trans)
        trans_pct_s = compute_stats(trans_pct)
        rot_per_m_s = compute_stats(rot_per_m)
        best_trans_pct_s = compute_stats(best_trans_pct)
        best_rot_per_m_s = compute_stats(best_rot_per_m)

        outlier_count = sum(
            1 for r, t in zip(rot, trans)
            if abs(r) > OUTLIER_ROT_THRESH_DEG or t > OUTLIER_TRANS_THRESH_M
        )
        outlier_best_count = sum(
            1 for r, t in zip(best_rot, best_trans)
            if abs(r) > OUTLIER_ROT_THRESH_DEG or t > OUTLIER_TRANS_THRESH_M
        )

        results.append({
            "method": method,
            "num_sequences": len(d["sequences"]),
            "total_pairs": len(rot),
            "rot_mean_deg": rot_s["mean"],
            "rot_std_deg": rot_s["std"],
            "rot_median_deg": rot_s["median"],
            "trans_mean_m": trans_s["mean"],
            "trans_std_m": trans_s["std"],
            "trans_median_m": trans_s["median"],
            "best_rot_mean_deg": best_rot_s["mean"],
            "best_rot_std_deg": best_rot_s["std"],
            "best_rot_median_deg": best_rot_s["median"],
            "best_trans_mean_m": best_trans_s["mean"],
            "best_trans_std_m": best_trans_s["std"],
            "best_trans_median_m": best_trans_s["median"],
            "trans_err_pct_mean": trans_pct_s["mean"],
            "trans_err_pct_std": trans_pct_s["std"],
            "trans_err_pct_median": trans_pct_s["median"],
            "rot_err_deg_per_m_mean": rot_per_m_s["mean"],
            "rot_err_deg_per_m_std": rot_per_m_s["std"],
            "rot_err_deg_per_m_median": rot_per_m_s["median"],
            "best_trans_err_pct_mean": best_trans_pct_s["mean"],
            "best_trans_err_pct_std": best_trans_pct_s["std"],
            "best_trans_err_pct_median": best_trans_pct_s["median"],
            "best_rot_err_deg_per_m_mean": best_rot_per_m_s["mean"],
            "best_rot_err_deg_per_m_std": best_rot_per_m_s["std"],
            "best_rot_err_deg_per_m_median": best_rot_per_m_s["median"],
            "outlier_count": outlier_count,
            "outlier_best_count": outlier_best_count,
        })

    columns = list(results[0].keys())
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    print(f"Aggregated {len(results)} methods (N >= {MIN_N}) -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
