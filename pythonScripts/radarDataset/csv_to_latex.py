#!/usr/bin/env python3
"""Convert paper CSV outputs to LaTeX tabulars.

Reads the two _paper.csv files from aggregate_benchmark_results.py and
generates corresponding .tex table files with booktabs + siunitx formatting.

Usage:
    python3 csv_to_latex.py \\
        --summary path/to/aggregated_summary_paper.csv \\
        --outlier path/to/aggregated_outlier_matrix_paper.csv
"""

import argparse
import csv
import math
from pathlib import Path


# ============================================================================
# Configuration — edit as needed
# ============================================================================

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

    # Column groups: (mean_col, std_col, median_col, label, median_spec)
    groups = [
        ("rot_mean_deg", "rot_std_deg", "rot_median_deg",
         "Rot. err. (\textdegree )", "S[table-format=3.2]"),
        ("trans_mean_m", "trans_std_m", "trans_median_m",
         "Trans. err. (m)", "S[table-format=5.3]"),
    ]
    outlier_col = "outlier_count"

    # Find best values for bolding
    best_vals: dict[str, float] = {}
    if bold_best:
        for mean_col, _, median_col, _, _ in groups:
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
    spec = "l"
    for _, _, _, _, median_spec in groups:
        spec += "c" + median_spec
    spec += "S[table-format=5.0]"

    # Build header: two-level
    header_top = ["{Method}"]
    header_bot = [""]
    for _, _, _, label, _ in groups:
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

    # Helper to check if a value is the best
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
            return f"{{\\bfseries {formatted}}}"
        return formatted

    # Data rows
    for row in rows:
        method = row.get("method", "")
        cells = [latex_escape(method)]
        for mean_col, std_col, median_col, _, median_spec in groups:
            # Build mean $\\pm$ std cell
            mean_raw = row.get(mean_col)
            std_raw = row.get(std_col)
            d = PRECISION.get(mean_col, 2)
            mean_str = format_value(mean_raw, d)
            std_str = format_value(std_raw, d)
            combined = f"{mean_str} $\\pm$ {std_str}"
            # Bold entire combined cell if mean is best
            if _is_best(mean_col, mean_raw):
                combined = f"{{\\bfseries {combined}}}"
            cells.append(combined)

            # Median cell
            med_raw = row.get(median_col)
            d_m = PRECISION.get(median_col, 2)
            med_str = format_value(med_raw, d_m)
            med_str = _maybe_bold(median_col, med_raw, med_str)
            cells.append(med_str)

        # Outlier count
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

    # Detect columns: sequence is first, total_pairs is second, then method columns
    all_keys = list(rows[0].keys())
    seq_col = all_keys[0]  # 'sequence'
    total_col = all_keys[1]  # 'total_pairs'
    methods = sorted(all_keys[2:])
    n_methods = len(methods)

    # Column spec: sequence + one S per method + total_pairs
    spec = "l" + OUTLIER_CELL_FORMAT * n_methods + OUTLIER_CELL_FORMAT

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Outlier counts per sequence and method. The ``Total'' column shows "
                 "the number of registration pairs in each sequence.}")
    lines.append("\\label{tab:outlier_matrix}")
    lines.append("\\small")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{spec}}}")
    lines.append("\\toprule")

    # Header
    header_cells = ["{Seq}"]
    for m in methods:
        header_cells.append(f"{{{latex_escape(m)}}}")
    header_cells.append("{Total}")
    lines.append(" & ".join(header_cells) + " \\\\")
    lines.append("\\midrule")

    # Data rows
    for row in rows:
        seq = row.get(seq_col, "")
        cells = [latex_escape(seq)]
        for m in methods:
            val = row.get(m, "0")
            formatted = format_value(val, 0)
            cells.append(formatted)
        # Total column
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
    parser = argparse.ArgumentParser(
        description="Convert aggregate_benchmark_results _paper CSVs to LaTeX tabulars."
    )
    parser.add_argument(
        "--summary",
        required=True,
        help="Path to aggregated_summary_paper.csv",
    )
    parser.add_argument(
        "--outlier",
        required=True,
        help="Path to aggregated_outlier_matrix_paper.csv",
    )
    parser.add_argument(
        "--bold-best",
        action="store_true",
        help="Bold the best (lowest) value in each numeric column of the summary table",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory for .tex files (default: same directory as input CSVs)",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    outlier_path = Path(args.outlier)

    if not summary_path.is_file():
        print(f"ERROR: --summary file not found: {summary_path}")
        sys.exit(1)
    if not outlier_path.is_file():
        print(f"ERROR: --outlier file not found: {outlier_path}")
        sys.exit(1)

    outdir = Path(args.outdir) if args.outdir else summary_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    # Read CSVs
    summary_rows = read_csv(summary_path)
    outlier_rows = read_csv(outlier_path)

    if not summary_rows:
        print("WARNING: summary CSV is empty — no .tex generated")
    else:
        tex_summary = generate_summary_table(summary_rows, bold_best=args.bold_best)
        out_summary = outdir / "aggregated_summary_paper.tex"
        out_summary.write_text(tex_summary)
        print(f"Written: {out_summary}")

    if not outlier_rows:
        print("WARNING: outlier CSV is empty — no .tex generated")
    else:
        tex_outlier = generate_outlier_matrix(outlier_rows)
        out_outlier = outdir / "aggregated_outlier_matrix_paper.tex"
        out_outlier.write_text(tex_outlier)
        print(f"Written: {out_outlier}")


def read_csv(path: Path) -> list[dict]:
    """Read a CSV file and return a list of row dicts (all values as strings)."""
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


if __name__ == "__main__":
    import sys
    main()
