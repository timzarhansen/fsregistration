#!/usr/bin/env python3
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "3d")
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


def inspect_3d_data(filename, N):
    data = load_csv(filename)
    if data is None:
        return None
    data = data.reshape((N, N, N))
    print(f"\n{filename}:")
    print(f"  Shape: {data.shape}")
    print(f"  Min: {data.min():.6f}")
    print(f"  Max: {data.max():.6f}")
    print(f"  Mean: {data.mean():.6f}")
    print(f"  Std: {data.std():.6f}")
    return data


def plot_3d_slices(filename, N, title=None):
    data = load_csv(filename)
    if data is None:
        return
    data = data.reshape((N, N, N))
    basename = filename.replace(".csv", "")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    mid = N // 2
    slices = [
        (data[:, :, mid], "Z-mid (XY plane)"),
        (data[:, mid, :], "Y-mid (XZ plane)"),
        (data[mid, :, :], "X-mid (YZ plane)"),
    ]
    for ax, (sl, label) in zip(axes, slices):
        im = ax.imshow(sl.T, cmap="viridis", aspect="auto", origin="lower")
        ax.set_title(label)
        fig.colorbar(im, ax=ax)

    fig.suptitle(title or filename, fontsize=12)
    fig.tight_layout()
    save_fig(fig, f"{basename}_slices")


def plot_3d_max_projection(filename, N, title=None):
    data = load_csv(filename)
    if data is None:
        return
    data = data.reshape((N, N, N))
    basename = filename.replace(".csv", "")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    projections = [
        (np.max(data, axis=2), "Max projection (XY)"),
        (np.max(data, axis=1), "Max projection (XZ)"),
        (np.max(data, axis=0), "Max projection (YZ)"),
    ]
    for ax, (proj, label) in zip(axes, projections):
        im = ax.imshow(proj.T, cmap="viridis", aspect="auto", origin="lower")
        ax.set_title(label)
        fig.colorbar(im, ax=ax)

    fig.suptitle(title or filename, fontsize=12)
    fig.tight_layout()
    save_fig(fig, f"{basename}_max_proj")


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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate 3D registration debug results")
    parser.add_argument("--list", action="store_true", help="List available CSV files")
    parser.add_argument("--inspect", type=int, metavar="N",
                        help="Inspect all 3D CSVs with shape NxNxN")
    parser.add_argument("--slices", type=int, metavar="N",
                        help="Plot mid-slices for all 3D CSVs with shape NxNxN")
    parser.add_argument("--max-proj", type=int, metavar="N",
                        help="Plot max projections for all 3D CSVs with shape NxNxN")
    args = parser.parse_args()

    if args.list or (not args.inspect and not args.slices and not args.max_proj):
        list_available_files()

    if args.inspect or args.slices or args.max_proj:
        N = args.inspect or args.slices or args.max_proj
        csv_files = os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else []
        big_files = [f for f in csv_files if f.endswith(".csv")
                     and os.path.getsize(os.path.join(DATA_DIR, f)) > 1024]

        if args.inspect:
            print(f"\nInspecting {len(big_files)} file(s) with shape ({N},{N},{N}) ...")
            for f in big_files:
                inspect_3d_data(f, N)

        if args.slices:
            print(f"\nPlotting slices for {len(big_files)} file(s) ...")
            for f in big_files:
                plot_3d_slices(f, N)

        if args.max_proj:
            print(f"\nPlotting max projections for {len(big_files)} file(s) ...")
            for f in big_files:
                plot_3d_max_projection(f, N)

    print(f"\nFigures saved to: {FIGS_DIR}")
