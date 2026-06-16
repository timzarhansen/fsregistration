#!/usr/bin/env python3
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "2d")
FIGS_DIR = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(FIGS_DIR, exist_ok=True)


def load_csv(filename):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    return np.loadtxt(filepath)


def save_fig(fig, basename):
    path = os.path.join(FIGS_DIR, basename)
    fig.savefig(path + ".png", dpi=150)
    fig.savefig(path + ".pdf")
    plt.close(fig)
    print(f"  Saved: {path}.png, {path}.pdf")


def plot_1d_correlation(filename, title=None):
    data = load_csv(filename)
    if data is None:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data)
    ax.set_title(title or filename)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Correlation")
    ax.grid(True)
    fig.tight_layout()
    save_fig(fig, filename.replace(".csv", "_1d_corr"))


def plot_2d_grid(filename, shape, title=None):
    data = load_csv(filename)
    if data is None:
        return
    if data.ndim == 1:
        data = data.reshape(shape)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title(title or filename)
    fig.tight_layout()
    save_fig(fig, filename.replace(".csv", "_2d_grid"))


def compare_1d_correlations(files, titles=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, filename in enumerate(files):
        data = load_csv(filename)
        if data is not None:
            label = titles[i] if titles else filename
            ax.plot(data, label=label)
    ax.legend()
    ax.set_xlabel("Sample")
    ax.set_ylabel("Correlation")
    ax.grid(True)
    fig.tight_layout()
    save_fig(fig, "comparison_1d_corr")


def list_available_files():
    if not os.path.exists(DATA_DIR):
        print(f"Directory not found: {DATA_DIR}")
        return []
    csv_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    print(f"Available CSV files in {DATA_DIR}:")
    for f in csv_files:
        size = os.path.getsize(os.path.join(DATA_DIR, f))
        print(f"  {f} ({size / 1024:.1f} KB)")
    return csv_files


def plot_all_1d():
    csv_files = list_available_files()
    corr_files = [f for f in csv_files if "Correlation" in f or "correlation" in f or "corr" in f]
    if corr_files:
        print(f"\nPlotting {len(corr_files)} 1D correlation file(s) ...")
        for f in corr_files:
            plot_1d_correlation(f)
        if len(corr_files) > 1:
            print("\nCreating comparison plot ...")
            compare_1d_correlations(corr_files)


def plot_all_2d(N):
    csv_files = os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else []
    grid_candidates = [f for f in csv_files if any(k in f for k in ["magnitude", "resampled", "voxel"])]
    print(f"\nPlotting {len(grid_candidates)} 2D grid file(s) with shape ({N},{N}) ...")
    for f in grid_candidates:
        plot_2d_grid(f, (N, N))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate 2D registration debug results")
    parser.add_argument("--all-1d", action="store_true", help="Plot all 1D correlation CSVs")
    parser.add_argument("--all-2d", type=int, metavar="N",
                        help="Plot all 2D grid CSVs with shape NxN")
    parser.add_argument("--list", action="store_true", help="List available CSV files")
    args = parser.parse_args()

    if args.list or (not args.all_1d and not args.all_2d):
        list_available_files()
    if args.all_1d:
        plot_all_1d()
    if args.all_2d:
        plot_all_2d(args.all_2d)

    print(f"\nFigures saved to: {FIGS_DIR}")
