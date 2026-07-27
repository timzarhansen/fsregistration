#!/usr/bin/env python3
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Boreas 2D Trajectory Viewer
#
# Interactive viewer for benchmark results on the Boreas 2D radar dataset (N ≥ 256).
#
# - **Single view** — pick a sequence and a method → estimated path + ground truth
# - **All methods** — pick a sequence → overlay all 10 method paths + ground truth
# - **Stats table** — per-sequence summary metrics for every method

# %% [markdown]
# ## 1. Imports

# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ipywidgets import interact, IntSlider, Dropdown
from pathlib import Path
import re

print("Imports OK")

# %% [markdown]
# ## 2. Configuration

# %%
DATA_DIR = Path.home() / "ros_ws" / "src" / "fsregistration" / "pythonScripts" / "radarDataset" / "2D_registration_results" / "ResultsRadar_DFKI" / "boreas2d"
STATS_FILE = DATA_DIR / "aggregated_boreas_results_all.csv"
AGGREGATED_FILE = DATA_DIR / "aggregated_boreas_results.csv"
MIN_N = 256

print(f"Data directory: {DATA_DIR}")

# %% [markdown]
# ## 3. Load data index

# %%
DIR_PATTERN = re.compile(r"seq(\d+)_(.+)_N(\d+)_p(\d+)_s(\d+)")

def parse_dir_name(name: str):
    m = DIR_PATTERN.match(name)
    if not m:
        return None
    return {"seq": int(m.group(1)), "method": m.group(2), "N": int(m.group(3)),
            "p": int(m.group(4)), "s": int(m.group(5))}

# Build index: (seq, method) -> Path to path.csv, filtered to N >= MIN_N
path_files = {}
seq_methods = {}

all_subdirs = sorted(DATA_DIR.iterdir())
for sd in all_subdirs:
    if not sd.is_dir():
        continue
    info = parse_dir_name(sd.name)
    if info is None or info["N"] < MIN_N:
        continue
    path_csv = sd / "path.csv"
    if not path_csv.is_file():
        continue
    # Skip runs with no ground truth
    df_check = pd.read_csv(path_csv)
    gt_max = np.sqrt(df_check["gt_x"]**2 + df_check["gt_y"]**2).max()
    if gt_max < 1e-6:
        continue
    key = (info["seq"], info["method"])
    path_files[key] = path_csv
    seq_methods.setdefault(info["seq"], []).append(info["method"])

all_seqs = sorted(seq_methods.keys())
all_methods = sorted({m for methods in seq_methods.values() for m in methods})

print(f"Indexed {len(path_files)} runs ({len(all_seqs)} sequences × {len(all_methods)} methods)")

# Load aggregated stats
stats_df = pd.read_csv(STATS_FILE)
stats_df["seq"] = stats_df["seq"].astype(int)
print(f"Stats loaded: {len(stats_df)} rows")

# Helper to load path data
def load_traj(seq, method):
    p = path_files.get((seq, method))
    if p is None or not p.is_file():
        return None
    df = pd.read_csv(p)
    return {
        "est_x": df["est_x"].values,
        "est_y": df["est_y"].values,
        "gt_x": df["gt_x"].values,
        "gt_y": df["gt_y"].values,
        "frames": df["frame"].values,
    }

# colours for the 10 methods
METHOD_COLORS = {
    "akaze": "#1f77b4",
    "eloftr": "#ff7f0e",
    "fourier_mellin": "#2ca02c",
    "fs2d": "#d62728",
    "icp": "#9467bd",
    "kaze": "#8c564b",
    "lightglue": "#e377c2",
    "loftr": "#7f7f7f",
    "ndt_p2d": "#bcbd22",
    "sift": "#17becf",
}
GT_COLOR = "black"

print(f"Sequences: {all_seqs[0]}–{all_seqs[-1]}  ({len(all_seqs)} total)")
print(f"Methods: {', '.join(all_methods)}")


def fmt_outlier_best(row):
    """Return outlier_best_count as int, or '—' if best_* metrics are not meaningful."""
    if not np.isfinite(row["best_rot_mean_deg"]):
        return "—"
    return int(row["outlier_best_count"])

# %% [markdown]
# ## 4. Single-view — pick a sequence and a method

# %%
if len(all_seqs) > 0:

    @interact(
        seq_idx=IntSlider(min=0, max=len(all_seqs) - 1, value=0, step=1,
                          description="Sequence", continuous_update=False,
                          readout=True, readout_format="d"),
        method=Dropdown(options=all_methods, value=all_methods[0],
                        description="Method"),
    )
    def plot_single(seq_idx=0, method=all_methods[0]):
        seq = all_seqs[seq_idx]
        data = load_traj(seq, method)
        if data is None:
            return go.Figure().add_annotation(
                text=f"No path data for seq{seq:02d} / {method}",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )

        gt_x, gt_y = data["gt_x"], data["gt_y"]
        est_x, est_y = data["est_x"], data["est_y"]
        final_error = np.sqrt((est_x[-1] - gt_x[-1])**2 + (est_y[-1] - gt_y[-1])**2)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gt_x, y=gt_y, mode="lines",
                      name="Ground Truth", line=dict(color=GT_COLOR, width=3)))
        fig.add_trace(go.Scatter(x=est_x, y=est_y, mode="lines",
                      name=f"Estimated ({method})",
                      line=dict(color=METHOD_COLORS.get(method, "blue"), width=2)))
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers",
                      name="Start", marker=dict(color="green", size=12, symbol="circle")))
        fig.add_trace(go.Scatter(x=[gt_x[-1]], y=[gt_y[-1]], mode="markers",
                      name="End (GT)", marker=dict(color=GT_COLOR, size=12, symbol="x",
                                                    line=dict(width=3))))
        fig.add_trace(go.Scatter(x=[est_x[-1]], y=[est_y[-1]], mode="markers",
                      name="End (est)", marker=dict(color=METHOD_COLORS.get(method, "blue"),
                                                    size=12, symbol="circle")))
        fig.update_layout(
            title=f"seq{seq:02d} – {method} (N={MIN_N})  |  Final disp.: {final_error:.2f} m",
            height=650, width=850,
            xaxis_title="x (m)", yaxis_title="y (m)",
            legend=dict(x=0.01, y=0.99),
            margin=dict(l=60, r=60, t=50, b=60),
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        return fig

# %% [markdown]
# ## 5. All methods for one sequence

# %%
if len(all_seqs) > 0:

    @interact(
        seq_idx=IntSlider(min=0, max=len(all_seqs) - 1, value=0, step=1,
                          description="Sequence", continuous_update=False,
                          readout=True, readout_format="d"),
    )
    def plot_all(seq_idx=0):
        seq = all_seqs[seq_idx]
        fig = go.Figure()

        # Ground truth (from first available method)
        for m in seq_methods[seq]:
            data = load_traj(seq, m)
            if data is not None:
                fig.add_trace(go.Scatter(
                    x=data["gt_x"], y=data["gt_y"], mode="lines",
                    name="Ground Truth", line=dict(color=GT_COLOR, width=3),
                ))
                break

        # All methods
        for method in sorted(seq_methods[seq]):
            data = load_traj(seq, method)
            if data is None:
                continue
            fig.add_trace(go.Scatter(
                x=data["est_x"], y=data["est_y"], mode="lines",
                name=method,
                line=dict(color=METHOD_COLORS.get(method, None), width=2),
            ))

        fig.update_layout(
            title=f"seq{seq:02d} – All methods (N={MIN_N})",
            height=650, width=850,
            xaxis_title="x (m)", yaxis_title="y (m)",
            legend=dict(x=0.01, y=0.99),
            margin=dict(l=60, r=60, t=40, b=60),
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        return fig

# %% [markdown]
# ## 6. Per-sequence statistics table

# %%
if len(stats_df) > 0:

    @interact(
        seq_idx=IntSlider(min=0, max=len(all_seqs) - 1, value=0, step=1,
                          description="Sequence", continuous_update=False,
                          readout=True, readout_format="d"),
    )
    def show_stats(seq_idx=0):
        seq = all_seqs[seq_idx]
        sub = stats_df[stats_df["seq"] == seq].copy()
        if sub.empty:
            return go.Figure().add_annotation(
                text=f"No stats for seq{seq:02d}",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )

        display_df = pd.DataFrame({
            "method": sub["method"].values,
            "rot_deg": sub.apply(
                lambda r: f"{r['rot_mean_deg']:.3f} ± {r['rot_std_deg']:.3f}", axis=1),
            "rot_median_deg": sub["rot_median_deg"].round(3),
            "trans_m": sub.apply(
                lambda r: f"{r['trans_mean_m']:.3f} ± {r['trans_std_m']:.3f}", axis=1),
            "trans_median_m": sub["trans_median_m"].round(3),
            "outlier_count": sub["outlier_count"],
            "outlier_best_count": sub.apply(fmt_outlier_best, axis=1),
        })

        import plotly.figure_factory as ff
        fig = ff.create_table(display_df, height_constant=30)
        fig.update_layout(title=f"seq{seq:02d} – Per-method statistics")
        return fig

# %% [markdown]
# ## 7. Aggregated per-method statistics (all sequences pooled)

# %%
if AGGREGATED_FILE.is_file():
    agg_df = pd.read_csv(AGGREGATED_FILE).round(3)
    display_df = pd.DataFrame({
        "method": agg_df["method"].values,
        "rot_deg": agg_df.apply(
            lambda r: f"{r['rot_mean_deg']:.3f} ± {r['rot_std_deg']:.3f}", axis=1),
        "rot_median_deg": agg_df["rot_median_deg"].round(3),
        "trans_m": agg_df.apply(
            lambda r: f"{r['trans_mean_m']:.3f} ± {r['trans_std_m']:.3f}", axis=1),
        "trans_median_m": agg_df["trans_median_m"].round(3),
        "outlier_count": agg_df["outlier_count"],
        "outlier_best_count": agg_df.apply(fmt_outlier_best, axis=1),
    })

    import plotly.figure_factory as ff
    fig = ff.create_table(display_df, height_constant=30)
    fig.update_layout(title=f"Aggregated across all {len(all_seqs)} sequences (N≥{MIN_N})")
    fig.show()
else:
    print("Aggregated stats file not found")

# %% [markdown]
# ## Done
#
# Use the interactive widgets above to explore the Boreas 2D benchmark results.
