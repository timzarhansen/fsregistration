#!/usr/bin/env python3
"""Convert paper CSV outputs to LaTeX tabulars.

Reads the _paper.csv files produced by aggregate_benchmark_results.py and
generates corresponding .tex table files with booktabs + siunitx formatting.

Usage:
    python3 csv_to_latex.py
"""

import csv
import math
import sys
from pathlib import Path


# ============================================================================
# Configuration — edit as needed
# ============================================================================
RESULTS_BASE = Path(__file__).resolve().parent / "ResultsRadar_DFKI"

# Map dataset name → list of (output_prefix, summary_csv, outlier_csv)
# For datasets without noise variants: one entry.
# For simulation: one entry per noise level.
SIMULATION_NOISE_LEVELS = [
    "base", "high", "low",
    "high_gauss", "high_salt_pepper",
    "low_gauss", "low_salt_pepper",
]

DATASETS: dict[str, list[tuple[str, Path, Path]]] = {
    "boreas": [
        (
            "boreas",
            RESULTS_BASE / "boreas2d" / "boreas_aggregated_summary_paper.csv",
            RESULTS_BASE / "boreas2d" / "boreas_aggregated_outlier_matrix_paper.csv",
        ),
    ],
    "bremen": [
        (
            "bremen",
            RESULTS_BASE / "bremenmss2d" / "bremen_aggregated_summary_paper.csv",
            RESULTS_BASE / "bremenmss2d" / "bremen_aggregated_outlier_matrix_paper.csv",
        ),
    ],
    "simulation": [
        (
            f"simulation_{noise}",
            RESULTS_BASE / "simulation_gazebo_scans" / f"simulation_{noise}_aggregated_summary_paper.csv",
            RESULTS_BASE / "simulation_gazebo_scans" / f"simulation_{noise}_aggregated_outlier_matrix_paper.csv",
        )
        for noise in SIMULATION_NOISE_LEVELS
    ],
}

# Bold the best value in each column by default
BOLD_BEST = True

# Decimal places per column (summary table)
PRECISION: dict[str, int] = {
    "rot_mean_deg": 2,
    "rot_std_deg": 2,
    "rot_median_deg": 2,
    "trans_mean_m": 3,
    "trans_std_m": 3,
    "trans_median_m": 3,
    "outlier_count": 0,
}

# siunitx format for outlier matrix columns (all integer counts)
OUTLIER_CELL_FORMAT = "S[table-format=4.0]"


# ============================================================================
# Helpers
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


def read_csv(path: Path) -> list[dict]:
    """Read a CSV file and return a list of row dicts (all values as strings)."""
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ============================================================================
# Table generators
# ============================================================================

def generate_summary_table(rows: list[dict], bold_best: bool = False) -> str:
    """Generate LaTeX for the summary table (one row per method).

    Merges mean+std into \"mean $\\pm$ std\" cells.
    Caption includes total registrations and fs2d outlier count.
    """
    if not rows:
        return ""

    # Column groups: (mean_col, std_col, median_col, label)
    groups = [
        ("rot_mean_deg", "rot_std_deg", "rot_median_deg",
         "Rot. err. (\textdegree )"),
        ("trans_mean_m", "trans_std_m", "trans_median_m",
         "Trans. err. (m)"),
    ]
    outlier_col = "outlier_count"

    # Find best values for bolding
    best_vals: dict[str, float] = {}
    if bold_best:
        for mean_col, _, median_col, _ in groups:
            for col in (mean_col, median_col):
                valid = []
                for row in rows:
                    v = row.get(col)
                    if v is not None and v != "":
                        try:
                            fv = float(v)
                            if not math.isnan(fv) and not math.isinf(fv):
                                valid.append(fv)
                        except (ValueError, TypeError):
                            pass
                if valid:
                    best_vals[col] = min(valid)
        valid_oc = []
        for row in rows:
            v = row.get(outlier_col)
            if v is not None and v != "":
                try:
                    fv = float(v)
                    if not math.isnan(fv) and not math.isinf(fv):
                        valid_oc.append(fv)
                except (ValueError, TypeError):
                    pass
        if valid_oc:
            best_vals[outlier_col] = min(valid_oc)

    # Build column spec: method | (mean\pmstd, median) x2 | outlier
    spec = "l" + "cr" * len(groups) + "r"

    # Build header: two-level
    header_top = ["{Method}"]
    header_bot = [""]
    for _, _, _, label in groups:
        header_top.append(f"\\multicolumn{{2}}{{c}}{{{label}}}")
        header_bot.append("{mean $\\pm$ std}")
        header_bot.append("{median}")
    header_top.append(f"{{{latex_escape(outlier_col)}}}")
    header_bot.append("")

    # Compute total registrations (from first method's total_pairs)
    total_regs = ""
    fs2d_outliers = ""
    for row in rows:
        if "total_pairs" in row and row["total_pairs"]:
            try:
                total_regs = f"{int(float(row['total_pairs'])):,}"
            except (ValueError, TypeError):
                pass
            break  # all methods have same total_pairs
    for row in rows:
        if row.get("method", "").strip().lower() == "fs2d":
            try:
                fs2d_outliers = f"{int(float(row[outlier_col])):,}"
            except (ValueError, TypeError):
                pass
            break

    caption = (
        "Aggregated registration performance with outlier rejection "
        f"(N = {total_regs} total pairs). "
        "Outlier thresholds: rotation $>10^\\circ$ or translation $>4$\\,m. "
    )
    if fs2d_outliers:
        caption += f"FS2D produces the fewest outliers ({fs2d_outliers})."

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\label{tab:reg_summary}")
    lines.append("\\small")
    lines.append(f"\\begin{{tabular}}{{{spec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(header_top) + " \\\\")
    lines.append("\\cmidrule(r){2-3}\\cmidrule(r){4-5}")
    lines.append(" & ".join(header_bot) + " \\\\")
    lines.append("\\midrule")

    def _is_best(col: str, raw_val) -> bool:
        if not bold_best or col not in best_vals:
            return False
        try:
            fv = float(raw_val) if raw_val not in (None, "") else float("nan")
            return not math.isnan(fv) and not math.isinf(fv) and abs(fv - best_vals[col]) < 1e-12
        except (ValueError, TypeError):
            return False

    def _maybe_bold(col: str, raw_val, formatted: str) -> str:
        if _is_best(col, raw_val):
            return f"\\bf{{{formatted}}}"
        return formatted

    # Data rows
    for row in rows:
        method = row.get("method", "")
        cells = [latex_escape(method)]
        for mean_col, std_col, median_col, _ in groups:
            mean_raw = row.get(mean_col)
            std_raw = row.get(std_col)
            d = PRECISION.get(mean_col, 2)
            mean_str = format_value(mean_raw, d)
            std_str = format_value(std_raw, d)
            combined = f"{mean_str} $\\pm$ {std_str}"
            if _is_best(mean_col, mean_raw):
                combined = f"\\bf{{{combined}}}"
            cells.append(combined)

            med_raw = row.get(median_col)
            d_m = PRECISION.get(median_col, 2)
            med_str = format_value(med_raw, d_m)
            med_str = _maybe_bold(median_col, med_raw, med_str)
            cells.append(med_str)

        oc_raw = row.get(outlier_col)
        oc_str = format_value(oc_raw, 0)
        oc_str = _maybe_bold(outlier_col, oc_raw, oc_str)
        cells.append(oc_str)

        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    return "\n".join(lines)


def generate_outlier_matrix(rows: list[dict]) -> str:
    """Generate LaTeX for the per-sequence outlier matrix (wide pivot table).

    Expects columns: sequence, total_pairs, method1, method2, ...
    Adds a \"Total\" column on the right showing per-sequence pair count.
    """
    if not rows:
        return ""

    all_keys = list(rows[0].keys())
    seq_col = all_keys[0]  # 'sequence'
    total_col = all_keys[1]  # 'total_pairs'
    methods = sorted(all_keys[2:])
    n_methods = len(methods)

    spec = "l" + OUTLIER_CELL_FORMAT * n_methods + OUTLIER_CELL_FORMAT

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Outlier counts per sequence and method. The ``Total'' shows "
                 "the number of registration pairs in each sequence.}")
    lines.append("\\label{tab:outlier_matrix}")
    lines.append("\\small")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{spec}}}")
    lines.append("\\toprule")

    header_cells = ["{Seq}"]
    for m in methods:
        header_cells.append(f"{{{latex_escape(m)}}}")
    header_cells.append("{Total}")
    lines.append(" & ".join(header_cells) + " \\\\")
    lines.append("\\midrule")

    for row in rows:
        seq = row.get(seq_col, "")
        cells = [latex_escape(seq)]
        for m in methods:
            val = row.get(m, "0")
            formatted = format_value(val, 0)
            cells.append(formatted)
        total_val = row.get(total_col, "0")
        total_formatted = format_value(total_val, 0)
        cells.append(total_formatted)
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    for dataset_name, entries in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        for prefix, summary_csv, outlier_csv in entries:
            print(f"\n  {prefix}:")

            if not summary_csv.is_file():
                print(f"    SKIP: summary CSV not found: {summary_csv.name}")
                continue
            if not outlier_csv.is_file():
                print(f"    SKIP: outlier CSV not found: {outlier_csv.name}")
                continue

            outdir = summary_csv.parent
            summary_rows = read_csv(summary_csv)
            outlier_rows = read_csv(outlier_csv)

            if summary_rows:
                tex_summary = generate_summary_table(summary_rows, bold_best=BOLD_BEST)
                out_summary = outdir / f"{prefix}_aggregated_summary_paper.tex"
                out_summary.write_text(tex_summary)
                print(f"    Written: {out_summary.name}")
            else:
                print(f"    WARNING: summary CSV empty, no .tex generated")

            if outlier_rows:
                tex_outlier = generate_outlier_matrix(outlier_rows)
                out_outlier = outdir / f"{prefix}_aggregated_outlier_matrix_paper.tex"
                out_outlier.write_text(tex_outlier)
                print(f"    Written: {out_outlier.name}")
            else:
                print(f"    WARNING: outlier CSV empty, no .tex generated")


if __name__ == "__main__":
    main()
