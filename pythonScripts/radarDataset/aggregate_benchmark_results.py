import csv
import sys
from pathlib import Path

import numpy as np


# ============================================================================
# Configuration — edit as needed
# ============================================================================
INPUT_FOLDER = Path("/home/tim-external/ros_ws/src/fsregistration/pythonScripts/radarDataset/benchmark_sweep")
OUTPUT_PATH = INPUT_FOLDER / "aggregated_results.csv"

OUTLIER_ROT_THRESH_DEG = 10.0
OUTLIER_TRANS_THRESH_M = 4.0


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
            "best_rot_error_deg", "best_trans_error_m"]
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
    for r in rows:
        nc = numeric_cols(r)
        rot.append(nc["rot_error_deg"])
        trans.append(nc["trans_error_m"])
        best_rot.append(nc["best_rot_error_deg"])
        best_trans.append(nc["best_trans_error_m"])

    rot_s = compute_stats(rot)
    trans_s = compute_stats(trans)
    best_rot_s = compute_stats(best_rot)
    best_trans_s = compute_stats(best_trans)

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
        "outlier_count": outlier_count,
        "outlier_best_count": outlier_best_count,
    }


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


if __name__ == "__main__":
    main()
