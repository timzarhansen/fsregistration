#!/usr/bin/env python3
"""Combined pipeline: aggregate benchmark results -> LaTeX tables (boreas only).

Adapted from 2D_registration_results/aggregate_and_generate_latex.py to run on
a single boreas results folder (allDatasets layout) pointed at by BOREAS_DIR.

In addition to the original outputs (aggregated_results.csv, the paper summary
CSVs/TeX), this version computes OUTLIER STATISTICS PER |GT-ROTATION| ANGLE
GROUP:

  * percent of outliers per method per angle group (ranges 15-20, 20-25,
    25-30, >30 deg and cumulative >15, >20, >25, >30 deg),
  * absolute outlier counts per method per angle group,
  * absolute number of pairs per angle group (whole dataset),

written to {prefix}_angle_group_outliers_paper.csv and appended as additional
LaTeX tables in {prefix}_angle_group_outliers_paper.tex.

Usage:
    python3 aggregate_and_generate_latex.py
"""

import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


# ============================================================================
# Configuration — edit as needed
# ============================================================================
# Path to the boreas results dataset. Must contain subdirectories
# seqNN_<method>_N<p>_p<..>_s<..>/results.csv (as in
# allDatasets/3stepNonRotation/boreas2d).
BOREAS_DIR = Path(
    "/home/tim-external/ros_ws/src/fsregistration/pythonScripts/radarDataset/"
    "2D_registration_results/allDatasets/3stepNonRotation/boreas2d"
)

DATASETS = {
    "boreas": BOREAS_DIR,
}

# Outlier thresholds.
# A rotation outlier is a pair with |rot_error| > OUTLIER_ROT_THRESH_DEG.
# A translation outlier is a pair with trans_error > OUTLIER_TRANS_THRESH_M.
# A pair can be flagged as both.
OUTLIER_ROT_THRESH_DEG = 5.0
OUTLIER_TRANS_THRESH_M = 4.0
MIN_GT_TRANS_M = 0.01

# Angle groups for the outlier-per-rotation statistics. Groups are defined on
# the absolute GT rotation of a pair (|gt_rot_deg|):
#   ANGLE_RANGES      -> mutually exclusive ranges [lo, hi) degrees
#   CUM_THRESHOLDS    -> cumulative "greater than" thresholds (degrees)
ANGLE_RANGES = [(15, 20), (20, 25), (25, 30), (30, None)]
CUM_THRESHOLDS = [15, 20, 25, 30]

# LaTeX options
BOLD_BEST = True

# Decimal places per column (summary table)
PRECISION: dict[str, int] = {
    "rot_mean_deg": 2,
    "rot_std_deg": 2,
    "rot_median_deg": 2,
    "trans_mean_m": 3,
    "trans_std_m": 3,
    "trans_median_m": 3,
}


# ============================================================================
# Directory name parser (for simulation noise variants)
# ============================================================================
DIR_PATTERN = re.compile(r"seq(\d+)_(.+)_N(\d+)_p(\d+)_s(\d+)(?:_(.*))?")


def parse_dir_name(name: str) -> dict | None:
    """Extract seq, method, N, p, s and noise level from a directory name
    like 'seq01_fourier_mellin_N256_p23_s1_high_gauss'.
    Runs without a noise suffix get noise = 'base'."""
    m = DIR_PATTERN.match(name)
    if not m:
        return None
    return {
        "seq": int(m.group(1)),
        "method": m.group(2),
        "N": int(m.group(3)),
        "p": int(m.group(4)),
        "s": int(m.group(5)),
        "noise": m.group(6) if m.group(6) else "base",
    }


# ============================================================================
# CSV helpers (aggregation stage)
# ============================================================================

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


def classify_outlier(rot: float, trans: float) -> tuple[bool, bool]:
    """Classify one (rot_error, trans_error) pair.

    Returns (is_rot_outlier, is_trans_outlier); a pair can be both.
    """
    is_rot = abs(rot) > OUTLIER_ROT_THRESH_DEG
    is_trans = trans > OUTLIER_TRANS_THRESH_M
    return is_rot, is_trans


def is_outlier_pair(p: dict) -> bool:
    """Union outlier flag for one collected pair dict."""
    is_rot, is_trans = classify_outlier(p["rot"], p["trans"])
    return is_rot or is_trans


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

    # Outlier counts, split into rotation / translation / both
    rot_outlier_count = sum(1 for r in rot if abs(r) > OUTLIER_ROT_THRESH_DEG)
    trans_outlier_count = sum(1 for t in trans if t > OUTLIER_TRANS_THRESH_M)
    both_outlier_count = sum(
        1 for r, t in zip(rot, trans)
        if abs(r) > OUTLIER_ROT_THRESH_DEG and t > OUTLIER_TRANS_THRESH_M
    )
    outlier_count = rot_outlier_count + trans_outlier_count - both_outlier_count

    rot_outlier_best_count = sum(1 for r in best_rot if abs(r) > OUTLIER_ROT_THRESH_DEG)
    trans_outlier_best_count = sum(1 for t in best_trans if t > OUTLIER_TRANS_THRESH_M)
    both_outlier_best_count = sum(
        1 for r, t in zip(best_rot, best_trans)
        if abs(r) > OUTLIER_ROT_THRESH_DEG and t > OUTLIER_TRANS_THRESH_M
    )
    outlier_best_count = (rot_outlier_best_count + trans_outlier_best_count
                         - both_outlier_best_count)

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
        "rot_outlier_count": rot_outlier_count,
        "trans_outlier_count": trans_outlier_count,
        "both_outlier_count": both_outlier_count,
        "outlier_best_count": outlier_best_count,
        "rot_outlier_best_count": rot_outlier_best_count,
        "trans_outlier_best_count": trans_outlier_best_count,
        "both_outlier_best_count": both_outlier_best_count,
    }


def collect_raw_data(subdir: Path) -> dict | None:
    """Extract method, sequence, noise, and raw (rot, trans) pairs from a results.csv.

    Each pair also carries its absolute GT rotation (|gt_rot_deg|) and a
    (sequence, prev_frame, curr_frame) key for deduplication across methods."""
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

    # Parse noise level from directory name
    dir_info = parse_dir_name(subdir.name)
    noise = dir_info["noise"] if dir_info else "base"

    pairs = []
    gt_all = []  # (key, |gt_rot|) for EVERY row — unfiltered, for dataset-level counts
    for r in rows:
        nc = numeric_cols(r)
        gt_trans_norm = np.sqrt(nc["gt_tx_m"]**2 + nc["gt_ty_m"]**2)
        try:
            gt_rot_abs = abs(float(r["gt_rot_deg"]))
            prev = str(r.get("prev_frame", ""))
            curr = str(r.get("curr_frame", ""))
        except (ValueError, KeyError, TypeError):
            continue
        key = (sequence, prev, curr)
        gt_all.append((key, gt_rot_abs))
        if gt_trans_norm < MIN_GT_TRANS_M:
            continue
        pairs.append({
            "rot": nc["rot_error_deg"],
            "trans": nc["trans_error_m"],
            "gt_rot_abs": gt_rot_abs,
            "key": key,
        })

    if method is None or sequence is None:
        return None

    return {"method": method, "sequence": sequence, "noise": noise,
            "pairs": pairs, "gt_all": gt_all}


# ============================================================================
# Paper tabulars (CSV stage)
# ============================================================================

def build_paper_tabulars(raw_data_list: list[dict], output_dir: Path, prefix: str) -> tuple[list, list]:
    """Write the paper CSVs and return the row dicts for the LaTeX stage.

    1. {prefix}_aggregated_summary_paper.csv — per-method stats with outlier
       rejection; outlier counts split into rotation / translation.
    2. {prefix}_aggregated_outlier_counts_paper.csv — exactly two rows
       (Rotation, Translation): outlier counts per method.
    """
    summary_path = output_dir / f"{prefix}_aggregated_summary_paper.csv"
    outlier_path = output_dir / f"{prefix}_aggregated_outlier_counts_paper.csv"

    # Group pairs by method
    method_pairs: dict[str, list] = defaultdict(list)
    for entry in raw_data_list:
        method = entry["method"]
        for p in entry["pairs"]:
            method_pairs[method].append(p)

    methods = sorted(method_pairs.keys())

    # --- Tabular 1: summary with outlier rejection ---
    summary_rows = []
    for method in methods:
        pairs = method_pairs[method]
        inlier_rots = []
        inlier_trans = []
        rot_out = 0
        trans_out = 0
        both_out = 0
        for p in pairs:
            is_rot, is_trans = classify_outlier(p["rot"], p["trans"])
            if is_rot or is_trans:
                if is_rot:
                    rot_out += 1
                if is_trans:
                    trans_out += 1
                if is_rot and is_trans:
                    both_out += 1
            else:
                inlier_rots.append(p["rot"])
                inlier_trans.append(p["trans"])

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
            "outlier_count": rot_out + trans_out - both_out,
            "rot_outlier_count": rot_out,
            "trans_outlier_count": trans_out,
        })

    summary_fields = [
        "method", "total_pairs",
        "rot_mean_deg", "rot_std_deg", "rot_median_deg",
        "trans_mean_m", "trans_std_m", "trans_median_m",
        "outlier_count", "rot_outlier_count", "trans_outlier_count",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Paper summary -> {summary_path}")

    # --- Tabular 2: outlier counts, two rows: rotation / translation ---
    rot_row = {"metric": "Rotation"}
    trans_row = {"metric": "Translation"}
    for method in methods:
        rot_row[method] = 0
        trans_row[method] = 0
    for method in methods:
        for p in method_pairs[method]:
            is_rot, is_trans = classify_outlier(p["rot"], p["trans"])
            if is_rot:
                rot_row[method] += 1
            if is_trans:
                trans_row[method] += 1
    outlier_rows = [rot_row, trans_row]

    outlier_fields = ["metric"] + methods
    with open(outlier_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=outlier_fields)
        writer.writeheader()
        writer.writerows(outlier_rows)
    print(f"Paper outlier counts -> {outlier_path}")

    return summary_rows, outlier_rows


# ============================================================================
# Angle-group outlier statistics (per |GT rotation| group)
# ============================================================================

def angle_groups() -> list[tuple]:
    """Build the (kind, lo, hi, label) group list: ranges then cumulatives."""
    groups = []
    for lo, hi in ANGLE_RANGES:
        label = f"{lo}-{hi}" if hi is not None else f">{lo}"
        groups.append(("range", lo, hi, label))
    for th in CUM_THRESHOLDS:
        groups.append(("cum", th, None, f">{th}"))
    return groups


def build_angle_group_tabular(raw_data_list: list[dict], output_dir: Path,
                              prefix: str) -> tuple[list, dict]:
    """Outlier statistics per method and |GT rotation| angle group.

    Writes {prefix}_angle_group_outliers_paper.csv and returns
    (method_rows, group_counts):

    - group_counts: absolute number of pairs per angle group over the whole
      dataset (GT-based; all pairs, deduplicated across methods via their
      (sequence, prev_frame, curr_frame) key — not subject to MIN_GT_TRANS_M)
    - method_rows: per-method, per-group pair counts, absolute outlier counts
      and outlier rates (percent; same filtered pair pool as the paper tables).
    """
    method_pairs: dict[str, list] = defaultdict(list)
    seen_keys: set = set()
    all_gt_rot = []
    for entry in raw_data_list:
        method = entry["method"]
        for p in entry["pairs"]:
            method_pairs[method].append(p)
        for key, gt_rot_abs in entry["gt_all"]:
            if key not in seen_keys:
                seen_keys.add(key)
                all_gt_rot.append(gt_rot_abs)

    methods = sorted(method_pairs.keys())
    groups = angle_groups()

    # Dataset-level counts per group
    group_counts: dict[str, int] = {}
    for kind, lo, hi, label in groups:
        if kind == "range":
            selected = [v for v in all_gt_rot
                        if lo <= v and (hi is None or v < hi)]
        else:
            selected = [v for v in all_gt_rot if v > lo]
        group_counts[label] = len(selected)
    group_counts["total"] = len(all_gt_rot)

    # Per-method rows
    method_rows = []
    for method in methods:
        pairs = method_pairs[method]
        row: dict = {"method": method, "total_pairs": len(pairs)}
        for kind, lo, hi, label in groups:
            if kind == "range":
                selected = [p for p in pairs
                            if lo <= p["gt_rot_abs"] and (hi is None or p["gt_rot_abs"] < hi)]
            else:
                selected = [p for p in pairs if p["gt_rot_abs"] > lo]
            n = len(selected)
            out = sum(1 for p in selected if is_outlier_pair(p))
            row[f"n_{label}"] = n
            row[f"out_{label}"] = out
            row[f"pct_{label}"] = f"{100.0 * out / n:.1f}" if n else ""
        method_rows.append(row)

    # CSV (one row per method + final ALL row with the dataset-level counts)
    fields = ["method", "total_pairs"]
    for _, _, _, label in groups:
        fields += [f"n_{label}", f"out_{label}", f"pct_{label}"]

    all_row = {"method": "ALL", "total_pairs": group_counts["total"]}
    for _, _, _, label in groups:
        n = group_counts[label]
        all_row[f"n_{label}"] = n
        all_row[f"out_{label}"] = ""
        all_row[f"pct_{label}"] = (
            f"{100.0 * n / group_counts['total']:.1f}" if group_counts["total"] else "")

    csv_path = output_dir / f"{prefix}_angle_group_outliers_paper.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(method_rows + [all_row])
    print(f"Angle-group outlier stats -> {csv_path}")

    # Console summary
    print(f"  pairs per |GT rot| group: " +
          ", ".join(f"{label}={group_counts[label]}" for _, _, _, label in groups))
    for row in method_rows:
        pcts = "  ".join(
            f"{label}: {row[f'pct_{label}']}%" if row[f"pct_{label}"] else f"{label}: ---"
            for _, _, _, label in groups)
        print(f"  {row['method']:<16} {pcts}")

    return method_rows, group_counts


# ============================================================================
# LaTeX helpers
# ============================================================================

def latex_escape(text: str) -> str:
    """Escape special LaTeX characters (underscores, &, %, etc.)."""
    result = text.replace("\\", r"\textbackslash ")
    result = result.replace("_", r"\_")
    result = result.replace("&", r"\&")
    result = result.replace("%", r"\%")
    result = result.replace("$", r"\$")
    result = result.replace("#", r"\#")
    result = result.replace("{", r"\{")
    result = result.replace("}", r"\}")
    result = result.replace("~", r"\textasciitilde ")
    result = result.replace("^", r"\textasciicircum ")
    return result


def format_value(val, decimals: int) -> str:
    """Format a numeric value for LaTeX, handling NaN, inf, and None."""
    if val is None or val == "":
        return r"\multicolumn{1}{c}{---}"
    if isinstance(val, str):
        low = val.strip().lower()
        if low in ("nan", ""):
            return r"\multicolumn{1}{c}{---}"
        if low in ("inf", "+inf", "-inf", "infinity"):
            return r"\multicolumn{1}{c}{$\infty$}"
        try:
            val = float(val)
        except ValueError:
            return latex_escape(val.strip())
    if isinstance(val, float):
        if math.isnan(val):
            return r"\multicolumn{1}{c}{---}"
        if math.isinf(val):
            return r"\multicolumn{1}{c}{$\infty$}"
    return f"{float(val):.{decimals}f}"


def format_threshold(val) -> str:
    """Format a threshold value for a caption, e.g. 5.0 -> '5', 0.5 -> '0.5'."""
    fv = float(val)
    return str(int(fv)) if fv.is_integer() else f"{fv:g}"


# ============================================================================
# Table generators (LaTeX stage)
# ============================================================================

def generate_summary_table(rows: list[dict], bold_best: bool = False) -> str:
    """Generate LaTeX for the summary table (one row per method).

    Rot. err. and Trans. err. columns merge mean+std into \"mean $\\pm$ std\".
    The outlier counts get three columns like the error stats: total (union),
    rotation, translation.
    """
    if not rows:
        return ""

    # Column groups: (key_prefix, mean_col, std_col, median_col, label)
    groups = [
        ("rot", "rot_mean_deg", "rot_std_deg", "rot_median_deg",
         "Rot. err. (\\textdegree )"),
        ("trans", "trans_mean_m", "trans_std_m", "trans_median_m",
         "Trans. err. (m)"),
    ]

    def _valid_values(list_rows, col) -> list[float]:
        valid = []
        for row in list_rows:
            v = row.get(col)
            if v is not None and v != "":
                try:
                    fv = float(v)
                    if not math.isnan(fv) and not math.isinf(fv):
                        valid.append(fv)
                except (ValueError, TypeError):
                    pass
        return valid

    # Find best values for bolding
    best_vals: dict[tuple[str, str], float] = {}
    if bold_best:
        for gkey, mean_col, _, median_col, _ in groups:
            for what, col in (("mean", mean_col), ("median", median_col)):
                valid = _valid_values(rows, col)
                if valid:
                    best_vals[(gkey, what)] = min(valid)
        for what, col in (("total", "outlier_count"),
                          ("rot", "rot_outlier_count"),
                          ("trans", "trans_outlier_count")):
            valid = _valid_values(rows, col)
            if valid:
                best_vals[("out", what)] = min(valid)

    # Build column spec: method | (mean\pmstd, median) x2 | total, rot., trans.
    spec = "l" + "cr" * len(groups) + "rrr"

    # Build header: two-level
    header_top = ["{Method}"]
    header_bot = [""]
    for _, _, _, _, label in groups:
        header_top.append(f"\\multicolumn{{2}}{{c}}{{{label}}}")
        header_bot.append("{mean $\\pm$ std}")
        header_bot.append("{median}")
    header_top.append("\\multicolumn{3}{c}{Outliers}")
    header_bot.append("{total}")
    header_bot.append("{rot.}")
    header_bot.append("{trans.}")

    # Compute total registrations (from first method's total_pairs)
    total_regs = ""
    for row in rows:
        if "total_pairs" in row and row["total_pairs"]:
            try:
                total_regs = f"{int(float(row['total_pairs'])):,}"
            except (ValueError, TypeError):
                pass
            break  # all methods have same total_pairs

    caption = (
        "Aggregated registration performance with outlier rejection "
        f"(N = {total_regs} total pairs). "
        "Outlier thresholds: rotation "
        f"$>{format_threshold(OUTLIER_ROT_THRESH_DEG)}^\\circ$ or translation "
        f"$>{format_threshold(OUTLIER_TRANS_THRESH_M)}$\\,m."
    )
    fs2d_outliers = ""
    for row in rows:
        if row.get("method", "").strip().lower() == "fs2d":
            try:
                fs2d_outliers = f"{int(float(row['outlier_count'])):,}"
            except (ValueError, TypeError, KeyError):
                pass
            break
    if fs2d_outliers:
        caption += f" FS2D produces the fewest outliers ({fs2d_outliers})."

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\label{tab:reg_summary}")
    lines.append("\\small")
    lines.append(f"\\begin{{tabular}}{{{spec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(header_top) + " \\\\")
    lines.append("\\cmidrule(r){2-3}\\cmidrule(r){4-5}\\cmidrule(lr){6-8}")
    lines.append(" & ".join(header_bot) + " \\\\")
    lines.append("\\midrule")

    def _is_best(key: tuple[str, str], raw_val) -> bool:
        if not bold_best or key not in best_vals:
            return False
        try:
            fv = float(raw_val) if raw_val not in (None, "") else float("nan")
            return (not math.isnan(fv) and not math.isinf(fv)
                    and abs(fv - best_vals[key]) < 1e-12)
        except (ValueError, TypeError):
            return False

    def _maybe_bold(key: tuple[str, str], raw_val, formatted: str) -> str:
        if _is_best(key, raw_val):
            return f"\\bf{{{formatted}}}"
        return formatted

    # Data rows
    for row in rows:
        method = row.get("method", "")
        cells = [latex_escape(method)]
        for gkey, mean_col, std_col, median_col, _ in groups:
            mean_raw = row.get(mean_col)
            std_raw = row.get(std_col)
            d = PRECISION.get(mean_col, 2)
            mean_str = format_value(mean_raw, d)
            std_str = format_value(std_raw, d)
            combined = f"{mean_str} $\\pm$ {std_str}"
            if _is_best((gkey, "mean"), mean_raw):
                combined = f"\\bf{{{combined}}}"
            cells.append(combined)

            med_raw = row.get(median_col)
            med_str = format_value(med_raw, PRECISION.get(median_col, 2))
            cells.append(_maybe_bold((gkey, "median"), med_raw, med_str))

        # Outlier columns: total / rotation / translation
        for what, col in (("total", "outlier_count"),
                          ("rot", "rot_outlier_count"),
                          ("trans", "trans_outlier_count")):
            raw = row.get(col)
            cells.append(_maybe_bold(("out", what), raw, format_value(raw, 0)))

        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    lines.append("")

    return "\n".join(lines)


def generate_angle_group_tables(prefix: str, group_counts: dict,
                                method_rows: list) -> str:
    """LaTeX for the angle-group outlier statistics: three tables.

    Tab. 1: absolute number of pairs per |GT rotation| group (whole dataset).
    Tab. 2: outlier rate (percent) per method per group.
    Tab. 3: absolute outlier counts per method per group.
    """
    groups = angle_groups()
    total = int(group_counts.get("total", 0) or 0)

    # --- Header cells (LaTeX) --------------------------------------------
    def tex_header_label(group) -> str:
        kind, lo, hi, label = group
        if kind == "range":
            return f"{lo}--{hi}" if hi is not None else f"$>{lo}$"
        return f"$>{lo}$"

    headers = [tex_header_label(g) for g in groups]
    spec = "l" + "r" * len(groups)

    outlier_desc = (
        f"outlier $\\equiv$ |rot. error| $>{format_threshold(OUTLIER_ROT_THRESH_DEG)}^\\circ$ "
        f"or trans. error $>{format_threshold(OUTLIER_TRANS_THRESH_M)}$\\,m (union)"
    )
    group_desc = ("angle groups over the absolute GT rotation "
                  "of a pair; ranges are $[\\mathrm{lo}, \\mathrm{hi})$ degrees, "
                  "cumulative columns count all pairs $>$ threshold")

    lines = []

    # --- Table 1: pair counts per group -----------------------------------
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    if total:
        lines.append(f"\\caption{{Number of pairs per |GT-rotation| group "
                     f"(boreas2d, N = {total:,} pairs). "
                     f"{group_desc}.}}")
    else:
        lines.append("\\caption{Number of pairs per |GT-rotation| group "
                     "(boreas2d). " + group_desc + ".}")
    lines.append("\\label{tab:angle_group_counts}")
    lines.append("\\small")
    lines.append(f"\\begin{{tabular}}{{{spec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(["{Group}"] + headers) + " \\\\")
    lines.append("\\midrule")
    n_cells = [f"{int(group_counts.get(g[3], 0)):,}" for g in groups]
    pct_cells = [
        (f"{100.0 * int(group_counts.get(g[3], 0)) / total:.1f}\\%" if total else "---")
        for g in groups
    ]
    lines.append(" & ".join(["pairs (n)"] + n_cells) + " \\\\")
    lines.append(" & ".join(["share of dataset"] + pct_cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    lines.append("")

    # --- Table 2: outlier rate (percent) per method ----------------------
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Outlier rate per method and |GT-rotation| group "
                 f"(percent of the method's pairs in the group; "
                 f"{outlier_desc}).}}")
    lines.append("\\label{tab:angle_group_pct}")
    lines.append("\\small")
    lines.append(f"\\begin{{tabular}}{{{spec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(["{Method}"] + headers) + " \\\\")
    lines.append("\\midrule")
    for row in method_rows:
        cells = [latex_escape(row["method"])]
        for _, _, _, label in groups:
            pct = row.get(f"pct_{label}", "")
            cells.append(f"{pct}\\%" if pct != "" else r"\multicolumn{1}{c}{---}")
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    lines.append("")

    # --- Table 3: absolute outlier counts per method ---------------------
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Absolute outlier counts per method and "
                 f"|GT-rotation| group ({outlier_desc}).}}")
    lines.append("\\label{tab:angle_group_out}")
    lines.append("\\small")
    lines.append(f"\\begin{{tabular}}{{{spec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(["{Method}"] + headers) + " \\\\")
    lines.append("\\midrule")
    for row in method_rows:
        cells = [latex_escape(row["method"])]
        for _, _, _, label in groups:
            cells.append(f"{int(row.get(f'out_{label}', 0)):,}")
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    for dataset_name, input_folder in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} ({input_folder})")
        print(f"{'='*60}")

        if not input_folder.is_dir():
            print(f"  WARNING: directory not found, skipping")
            continue

        subdirs = sorted(d for d in input_folder.iterdir() if d.is_dir())
        if not subdirs:
            print(f"  WARNING: no subdirectories found, skipping")
            continue

        # --- General aggregated results (all runs, one row per subdir) ---
        results = []
        skipped = 0
        for sd in subdirs:
            row = process_subdirectory(sd)
            if row is not None:
                results.append(row)
            else:
                skipped += 1

        if not results:
            print(f"  WARNING: no valid results.csv found, skipping")
            continue

        agg_path = input_folder / "aggregated_results.csv"
        columns = list(results[0].keys())
        with open(agg_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)
        print(f"  Aggregated {len(results)} runs -> {agg_path}")
        if skipped:
            print(f"  Skipped {skipped} subdirectories (no results.csv)")

        # --- Paper tabulars ---
        raw_data_list = []
        for sd in subdirs:
            rd = collect_raw_data(sd)
            if rd is not None:
                raw_data_list.append(rd)

        if not raw_data_list:
            print(f"  WARNING: no raw data for paper tabulars")
            continue

        # Detect noise levels present in this dataset
        noise_levels = sorted({e["noise"] for e in raw_data_list})

        if noise_levels == ["base"]:
            # No noise variants — single paper tabular set
            summary_rows, outlier_rows = build_paper_tabulars(
                raw_data_list, input_folder, dataset_name)
            latex_jobs = [(dataset_name, summary_rows, outlier_rows, raw_data_list)]
        else:
            # Multiple noise levels — one tabular set per level
            latex_jobs = []
            for noise in noise_levels:
                filtered = [e for e in raw_data_list if e["noise"] == noise]
                prefix = f"{dataset_name}_{noise}"
                print(f"\n  Noise level: {noise} ({len(filtered)} entries)")
                summary_rows, outlier_rows = build_paper_tabulars(
                    filtered, input_folder, prefix)
                latex_jobs.append((prefix, summary_rows, outlier_rows, filtered))

        # --- LaTeX stage (from the same in-memory data) ---
        for prefix, summary_rows, outlier_rows, raw_entries in latex_jobs:
            print(f"\n  LaTeX: {prefix}")

            # Angle-group outlier statistics (additional tables)
            angle_rows, group_counts = build_angle_group_tabular(
                raw_entries, input_folder, prefix)
            tex_angle = generate_angle_group_tables(
                prefix, group_counts, angle_rows)
            out_angle = input_folder / f"{prefix}_angle_group_outliers_paper.tex"
            out_angle.write_text(tex_angle)
            print(f"    Written: {out_angle.name}")

            if not summary_rows:
                print(f"    WARNING: summary rows empty, no summary .tex generated")
                continue

            tex_summary = generate_summary_table(summary_rows, bold_best=BOLD_BEST)
            out_summary = input_folder / f"{prefix}_aggregated_summary_paper.tex"
            out_summary.write_text(tex_summary)
            print(f"    Written: {out_summary.name}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()