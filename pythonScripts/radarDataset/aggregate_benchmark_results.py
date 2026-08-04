import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ============================================================================
# Configuration — edit as needed
# ============================================================================
INPUT_FOLDER = Path("/home/tim-external/ros_ws/src/fsregistration/pythonScripts/radarDataset/2D_registration_results/ResultsRadar_DFKI/boreas2d")
# INPUT_FOLDER = Path("/home/tim-external/ros_ws/src/fsregistration/pythonScripts/radarDataset/paramBenchMethods/benchmark_sweep")
OUTPUT_PATH = INPUT_FOLDER / "aggregated_results.csv"
OUTPUT_SUMMARY_PAPER = INPUT_FOLDER / "aggregated_summary_paper.csv"
OUTPUT_OUTLIER_PAPER = INPUT_FOLDER / "aggregated_outlier_matrix_paper.csv"

OUTLIER_ROT_THRESH_DEG = 10.0
OUTLIER_TRANS_THRESH_M = 4.0
MIN_GT_TRANS_M = 0.01


def read_data_rows(filepath: Path) -> list[dict]:
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            break

        reader = csv.DictReader(f, fieldnames=line.strip().split(","))
        for row in reader:
            rows.append(row)

    return rows


def numeric_cols(row: dict) -> dict:
    keys = ["rot_error_deg", "trans_error_m",
            "best_rot_error_deg", "best_trans_error_m",
            "gt_tx_m", "gt_ty_m"]
    out = {}
    for k in keys:
        try:
            out[k] = float(row[k])
        except (ValueError, KeyError):
            out[k] = np.nan
    # Rotation error is signed — use absolute for statistics
    out["rot_error_deg"] = abs(out["rot_error_deg"])
    out["best_rot_error_deg"] = abs(out["best_rot_error_deg"])
    # Translation error is already the L2 norm (always >= 0)

    # Normalised odometry metrics (avoid div-by-near-zero)
    gt_trans_norm = np.sqrt(out["gt_tx_m"]**2 + out["gt_ty_m"]**2)
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
    a = np.array(values, dtype=np.float64)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan")}
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)),
        "median": float(np.median(a)),
    }


def process_subdirectory(subdir: Path) -> dict | None:
    csv_path = subdir / "results.csv"
    if not csv_path.is_file():
        return None

    rows = read_data_rows(csv_path)
    if not rows:
        return None

    num_pairs_failed = 0
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

    return {
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


def collect_raw_data(subdir: Path) -> dict | None:
    """Extract method, sequence, and raw (rot, trans) pairs from a results.csv."""
    csv_path = subdir / "results.csv"
    if not csv_path.is_file():
        return None

    rows = read_data_rows(csv_path)
    if not rows:
        return None

    # Parse metadata from header comments
    method = None
    sequence = None
    with open(csv_path, "r") as f:
        for line in f:
            if not line.startswith("#"):
                break
            if line.startswith("# method:"):
                method = line.split(":")[-1].strip()
            if line.startswith("# sequence:"):
                sequence = line.split(":")[-1].strip()

    pairs = []
    for r in rows:
        nc = numeric_cols(r)
        gt_trans_norm = np.sqrt(nc["gt_tx_m"]**2 + nc["gt_ty_m"]**2)
        if gt_trans_norm < MIN_GT_TRANS_M:
            continue
        pairs.append((nc["rot_error_deg"], nc["trans_error_m"]))

    if method is None or sequence is None:
        return None

    return {"method": method, "sequence": sequence, "pairs": pairs}


def build_paper_tabulars(raw_data_list: list[dict]) -> None:
    """Produce two CSV files for the paper:
    1. aggregated_summary_paper.csv — per-method stats with outlier rejection
    2. aggregated_outlier_matrix_paper.csv — per-sequence outlier count matrix
    """
    # Group pairs by method and count outliers per (sequence, method)
    # Also track per-sequence total pair counts (use max across methods)
    method_pairs: dict[str, list] = defaultdict(list)
    seq_method_outliers: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    seq_total_pairs: dict[str, int] = defaultdict(int)

    for entry in raw_data_list:
        method = entry["method"]
        sequence = entry["sequence"]
        n_pairs = len(entry["pairs"])
        if n_pairs > seq_total_pairs[sequence]:
            seq_total_pairs[sequence] = n_pairs
        for rot, trans in entry["pairs"]:
            method_pairs[method].append((rot, trans))
            if abs(rot) > OUTLIER_ROT_THRESH_DEG or trans > OUTLIER_TRANS_THRESH_M:
                seq_method_outliers[sequence][method] += 1

    # --- Tabular 1: summary with outlier rejection ---
    methods = sorted(method_pairs.keys())
    summary_rows = []
    for method in methods:
        pairs = method_pairs[method]
        inlier_rots = []
        inlier_trans = []
        outlier_count = 0
        for rot, trans in pairs:
            if abs(rot) > OUTLIER_ROT_THRESH_DEG or trans > OUTLIER_TRANS_THRESH_M:
                outlier_count += 1
            else:
                inlier_rots.append(rot)
                inlier_trans.append(trans)

        rot_s = compute_stats(inlier_rots)
        trans_s = compute_stats(inlier_trans)

        summary_rows.append({
            "method": method,
            "total_pairs": len(pairs),
            "rot_mean_deg": rot_s["mean"],
            "rot_std_deg": rot_s["std"],
            "rot_median_deg": rot_s["median"],
            "trans_mean_m": trans_s["mean"],
            "trans_std_m": trans_s["std"],
            "trans_median_m": trans_s["median"],
            "outlier_count": outlier_count,
        })

    summary_fields = [
        "method", "total_pairs",
        "rot_mean_deg", "rot_std_deg", "rot_median_deg",
        "trans_mean_m", "trans_std_m", "trans_median_m", "outlier_count",
    ]
    with open(OUTPUT_SUMMARY_PAPER, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Paper summary -> {OUTPUT_SUMMARY_PAPER}")

    # --- Tabular 2: per-sequence outlier matrix ---
    sequences = sorted(seq_method_outliers.keys(), key=int)
    outlier_fields = ["sequence", "total_pairs"] + methods
    with open(OUTPUT_OUTLIER_PAPER, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=outlier_fields)
        writer.writeheader()
        for seq in sequences:
            row = {"sequence": seq, "total_pairs": seq_total_pairs.get(seq, 0)}
            for method in methods:
                row[method] = seq_method_outliers[seq].get(method, 0)
            writer.writerow(row)
    print(f"Paper outlier matrix -> {OUTPUT_OUTLIER_PAPER}")


def main():
    if not INPUT_FOLDER.is_dir():
        print(f"ERROR: {INPUT_FOLDER} is not a directory")
        sys.exit(1)

    subdirs = sorted(d for d in INPUT_FOLDER.iterdir() if d.is_dir())
    if not subdirs:
        print(f"No subdirectories found in {INPUT_FOLDER}")
        sys.exit(1)

    results = []
    skipped = 0
    for sd in subdirs:
        row = process_subdirectory(sd)
        if row is not None:
            results.append(row)
        else:
            skipped += 1

    if not results:
        print("No valid results.csv found in any subdirectory")
        sys.exit(1)

    columns = list(results[0].keys())
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    print(f"Aggregated {len(results)} runs -> {OUTPUT_PATH}")
    if skipped:
        print(f"Skipped {skipped} subdirectories (no results.csv)")

    # --- Paper tabulars (additional output, same data) ---
    raw_data_list = []
    for sd in subdirs:
        rd = collect_raw_data(sd)
        if rd is not None:
            raw_data_list.append(rd)

    build_paper_tabulars(raw_data_list)


if __name__ == "__main__":
    main()
