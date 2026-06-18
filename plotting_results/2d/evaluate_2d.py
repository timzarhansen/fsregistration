#!/usr/bin/env python3
"""Evaluate 2D registration debug results.

Usage:
    python evaluate_2d.py --list                          # List available CSVs
    python evaluate_2d.py --inspect-2d 128                # Inspect 2D grid stats
    python evaluate_2d.py --correlation-surface           # Plot all correlation2D_*.csv
    python evaluate_2d.py --peaks                         # Plot peak detection results
    python evaluate_2d.py --rotation-peaks                # Plot rotation correlation curve
    python evaluate_2d.py --all                           # Run all of the above
"""
import os
import sys
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(FIGS_DIR, exist_ok=True)


def load_csv(filename):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
    
    # Check if tab-separated
    is_tab = '\t' in first_line
    
    if is_tab or not _is_numeric(first_line):
        # Use numpy to read, handling headers and mixed delimiters
        try:
            if is_tab:
                # Tab-separated with header - skip header
                data = np.genfromtxt(filepath, delimiter='\t', skip_header=1)
                # Filter out empty lines
                if data.ndim == 1:
                    return data
                # Remove rows with all NaN (empty lines)
                mask = ~np.all(np.isnan(data), axis=1)
                return data[mask]
            else:
                # Has header but not tab-separated
                return np.genfromtxt(filepath, skip_header=1)
        except ValueError:
            # Mixed format file (e.g., dataForReadIn.csv with two sections)
            # Read all lines, skip headers and empty lines
            lines = []
            with open(filepath, 'r') as f2:
                for line in f2:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('N\t') and not line.startswith('angleIndex'):
                        lines.append(line)
            if lines:
                delim = '\t' if '\t' in lines[0] else None
                return np.loadtxt(lines, delimiter=delim)
            return None
        except Exception:
            return None
    
    return np.loadtxt(filepath)


def _is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def save_fig(fig, basename):
    path = os.path.join(FIGS_DIR, basename)
    fig.savefig(path + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}.png, {path}.pdf")


def list_available_files():
    if not os.path.exists(DATA_DIR):
        print(f"Directory not found: {DATA_DIR}")
        return []
    csv_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    print(f"Available CSV files in {DATA_DIR} ({len(csv_files)} files):")
    for f in csv_files:
        size = os.path.getsize(os.path.join(DATA_DIR, f))
        size_str = f"{size / 1024:.1f} KB" if size >= 1024 else f"{size} B"
        print(f"  {f} ({size_str})")
    return csv_files


def inspect_2d_data(filename, N, title=""):
    data = load_csv(filename)
    if data is None:
        return None
    if data.ndim == 1:
        data = data.reshape((N, N))
    print(f"\n{filename} {title}:")
    print(f"  Shape: {data.shape}")
    print(f"  Min: {data.min():.6f}")
    print(f"  Max: {data.max():.6f}")
    print(f"  Mean: {data.mean():.6f}")
    print(f"  Std: {data.std():.6f}")
    return data


def plot_input_data(N):
    """Plot input voxel data and spectra."""
    print("\n=== Plotting Input Data ===")
    files = [
        ("voxelDataFFTW1.csv", "Voxel Data 1 (FFT input)", "voxel1"),
        ("voxelDataFFTW2.csv", "Voxel Data 2 (FFT input)", "voxel2"),
        ("magnitudeFFTW1.csv", "Magnitude FFTW 1", "mag1"),
        ("phaseFFTW1.csv", "Phase FFTW 1", "phase1"),
        ("magnitudeFFTW2.csv", "Magnitude FFTW 2", "mag2"),
        ("phaseFFTW2.csv", "Phase FFTW 2", "phase2"),
        ("resampledVoxel1.csv", "Resampled Voxel 1", "resamp1"),
        ("resampledVoxel2.csv", "Resampled Voxel 2", "resamp2"),
    ]
    for fname, title, prefix in files:
        data = load_csv(fname)
        if data is None:
            continue
        data = data.reshape((N, N))
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(data, cmap="viridis", aspect="equal", origin="lower")
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.colorbar(im, ax=ax, label="Value")
        fig.tight_layout()
        save_fig(fig, f"input_{prefix}")


def plot_rotation_peaks():
    """Plot rotation correlation curve and detected peaks."""
    print("\n=== Plotting Rotation Peaks ===")

    # Plot correlation curve
    data = load_csv("rotationCorrelation1D.csv")
    if data is not None:
        if data.ndim == 2:
            # Header row + data
            angles = data[:, 0]
            correlations = data[:, 2]
        else:
            angles = np.arange(len(data))
            correlations = data
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(angles, correlations, "b-", linewidth=0.5, label="Correlation")
        ax.set_title("Rotation Correlation Curve")
        ax.set_xlabel("Angle Index")
        ax.set_ylabel("Normalized Correlation")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        save_fig(fig, "rotation_correlation_curve")

    # Plot detected peaks
    data = load_csv("rotationPeaks.csv")
    if data is not None:
        if data.ndim == 2 and data.shape[1] >= 4:
            indices = data[:, 3].astype(int)
            angles = data[:, 0]
            peak_heights = data[:, 1]
        else:
            print("  rotationPeaks.csv has unexpected format, skipping")
            return
        fig, ax = plt.subplots(figsize=(12, 4))
        corr = load_csv("rotationCorrelation1D.csv")
        if corr is not None:
            if corr.ndim == 2:
                ax.plot(corr[:, 0], corr[:, 2], "b-", linewidth=0.5, alpha=0.5, label="Correlation")
        ax.scatter(indices, peak_heights, c="red", s=50, zorder=5, label="Detected Peaks")
        ax.set_title("Detected Rotation Peaks")
        ax.set_xlabel("Angle Index")
        ax.set_ylabel("Peak Correlation")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        save_fig(fig, "rotation_peaks")


def plot_correlation_surfaces():
    """Plot 2D correlation surfaces for each rotation angle."""
    print("\n=== Plotting Correlation Surfaces ===")
    pattern = re.compile(r"correlation2D_(\d+)(?:_maxNorm)?\.csv$")
    corr_files = []
    for f in os.listdir(DATA_DIR):
        m = pattern.match(f)
        if m:
            angle_idx = int(m.group(1))
            is_norm = "_maxNorm" in f
            corr_files.append((angle_idx, f, is_norm))
    corr_files.sort()

    # Group by angle index
    angles = {}
    for angle_idx, fname, is_norm in corr_files:
        if angle_idx not in angles:
            angles[angle_idx] = {}
        angles[angle_idx]["norm" if is_norm else "raw"] = fname

    max_angle = max(angles.keys()) if angles else 0
    print(f"  Found correlation surfaces for {len(angles)} angles (0-{max_angle})")

    for angle_idx in sorted(angles.keys()):
        fname = angles[angle_idx].get("raw", angles[angle_idx].get("norm"))
        if fname is None:
            continue
        data = load_csv(fname)
        if data is None:
            continue
        correlationN = int(round(data.size ** 0.5))
        data = data.reshape((correlationN, correlationN))
        title = f"Correlation Surface (Angle {angle_idx})"
        if "_maxNorm" in fname:
            title += " [Normalized]"

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(data, cmap="viridis", aspect="equal", origin="lower")
        ax.set_title(title)
        ax.set_xlabel("X Voxel")
        ax.set_ylabel("Y Voxel")
        fig.colorbar(im, ax=ax, label="Correlation" if "Norm" not in fname else "Normalized Correlation")
        fig.tight_layout()
        save_fig(fig, f"correlation_surface_{angle_idx:04d}")


def plot_peak_detection():
    """Plot raw peaks and filtered peaks for each rotation angle."""
    print("\n=== Plotting Peak Detection ===")
    pattern = re.compile(r"(peaks|peaksFiltered|peakFilterParams)_(\d+)\.csv$")
    peak_files = []
    for f in os.listdir(DATA_DIR):
        m = pattern.match(f)
        if m:
            peak_type = m.group(1)
            angle_idx = int(m.group(2))
            peak_files.append((angle_idx, peak_type, f))
    peak_files.sort()

    # Group by angle index
    angles = {}
    for angle_idx, peak_type, fname in peak_files:
        if angle_idx not in angles:
            angles[angle_idx] = {}
        angles[angle_idx][peak_type] = fname

    for angle_idx in sorted(angles.keys()):
        raw_data = load_csv(angles[angle_idx].get("peaks", "")) if "peaks" in angles[angle_idx] else None
        filtered_data = load_csv(angles[angle_idx].get("peaksFiltered", "")) if "peaksFiltered" in angles[angle_idx] else None

        if raw_data is None and filtered_data is None:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Raw peaks
        ax1 = axes[0]
        if raw_data is not None and raw_data.ndim == 2 and raw_data.shape[1] >= 7:
            ax1.scatter(raw_data[:, 0], raw_data[:, 1], c=raw_data[:, 2], s=10,
                       cmap="viridis", alpha=0.6, label="Raw peaks")
            ax1.set_title(f"Raw Peaks (Angle {angle_idx}): {len(raw_data)} peaks")
        else:
            ax1.text(0.5, 0.5, "No raw peak data", ha="center", va="center", transform=ax1.transAxes)
            ax1.set_title(f"Raw Peaks (Angle {angle_idx})")

        # Filtered peaks
        ax2 = axes[1]
        if filtered_data is not None and filtered_data.ndim == 2 and filtered_data.shape[1] >= 6:
            ax2.scatter(filtered_data[:, 2], filtered_data[:, 3], c=filtered_data[:, 4], s=50,
                       cmap="viridis", edgecolors="red", linewidths=0.5, label="Filtered peaks")
            ax2.set_title(f"Filtered Peaks (Angle {angle_idx}): {len(filtered_data)} peaks")
        else:
            ax2.text(0.5, 0.5, "No filtered peak data", ha="center", va="center", transform=ax2.transAxes)
            ax2.set_title(f"Filtered Peaks (Angle {angle_idx})")

        for ax in axes:
            ax.set_xlabel("X Voxel")
            ax.set_ylabel("Y Voxel")
            ax.set_aspect("equal")
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Peak Detection Analysis (Angle {angle_idx})")
        fig.tight_layout()
        save_fig(fig, f"peak_detection_{angle_idx:04d}")


def plot_transform_summary():
    """Plot summary of all detected transforms."""
    print("\n=== Plotting Transform Summary ===")
    
    # Read dataForReadIn.csv manually (special format with two sections)
    filepath = os.path.join(DATA_DIR, "dataForReadIn.csv")
    if not os.path.exists(filepath):
        print("  dataForReadIn.csv not found")
        return
    
    num_angles = 0
    num_total_solutions = 0
    angle_data = []  # list of (angleIndex, angle, numTranslations)
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # First line: header
    # Second line: N, correlationN, cellSize, potentialNecessaryForPeak, numAngles, numTotalSolutions
    if len(lines) > 1:
        meta = lines[1].strip().split()
        if len(meta) >= 5:
            num_angles = int(meta[4])
            if len(meta) >= 6:
                num_total_solutions = int(meta[5])
    
    # Find the per-angle section (starts with "angleIndex")
    for i, line in enumerate(lines):
        if line.strip().startswith('angleIndex'):
            for j in range(i + 1, len(lines)):
                parts = lines[j].strip().split('\t')
                if len(parts) >= 3:
                    angle_data.append((int(parts[0]), float(parts[1]), int(parts[2])))
            break
    
    print(f"  Total angles: {num_angles}")
    print(f"  Total solutions: {num_total_solutions}")

    # Load individual transform files
    transforms = []
    for i in range(100):  # Max 100 transforms
        fname = f"potentialTransformation{i}.csv"
        data_t = load_csv(fname)
        if data_t is None:
            break
        transform = data_t[:4, :].flatten() if data_t.ndim > 1 else data_t
        peak_height = data_t[4] if data_t.ndim > 1 else 0
        transforms.append((i, transform, peak_height))

    if not transforms:
        print("  No transform files found")
        return

    print(f"  Found {len(transforms)} transforms")

    # Plot peak heights vs transform index
    indices = [t[0] for t in transforms]
    heights = [t[2] for t in transforms]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(indices, heights, c=heights, cmap="viridis", s=30, edgecolors="black", linewidths=0.3)
    ax.set_title("Transform Peak Heights")
    ax.set_xlabel("Transform Index")
    ax.set_ylabel("Peak Height")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "transform_summary")


def run_all(N):
    """Run all visualization steps."""
    plot_input_data(N)
    plot_rotation_peaks()
    plot_correlation_surfaces()
    plot_peak_detection()
    plot_transform_summary()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate 2D registration debug results")
    parser.add_argument("--list", action="store_true", help="List available CSV files")
    parser.add_argument("--inspect-2d", type=int, metavar="N",
                        help="Inspect all 2D grid CSVs with shape NxN")
    parser.add_argument("--correlation-surface", action="store_true",
                        help="Plot all correlation2D_*.csv as heatmaps")
    parser.add_argument("--peaks", action="store_true",
                        help="Plot peak detection results")
    parser.add_argument("--rotation-peaks", action="store_true",
                        help="Plot rotation correlation curve and detected peaks")
    parser.add_argument("--transform-summary", action="store_true",
                        help="Plot summary of all detected transforms")
    parser.add_argument("--all", action="store_true", dest="run_all",
                        help="Run all visualization steps")
    args = parser.parse_args()

    if args.list or (not args.list and not args.inspect_2d and not args.run_all and
                     not args.correlation_surface and not args.peaks and not args.rotation_peaks and
                     not args.transform_summary):
        list_available_files()

    # Infer N from available files
    N = None
    for fname in ["voxelDataFFTW1.csv", "magnitudeFFTW1.csv", "resampledVoxel1.csv"]:
        data = load_csv(fname)
        if data is not None and data.ndim == 1:
            N = int(round(data.size ** 0.5))
            print(f"Inferred N = {N} from {fname}")
            break

    if N is None:
        print("Could not infer N. Use --inspect-2d N to specify.")
        sys.exit(1)

    if args.inspect_2d:
        N = args.inspect_2d
        for fname in ["voxelDataFFTW1.csv", "voxelDataFFTW2.csv", "magnitudeFFTW1.csv",
                      "phaseFFTW1.csv", "magnitudeFFTW2.csv", "phaseFFTW2.csv",
                      "resampledVoxel1.csv", "resampledVoxel2.csv"]:
            inspect_2d_data(fname, N)

    if args.correlation_surface or args.run_all:
        plot_correlation_surfaces()

    if args.peaks or args.run_all:
        plot_peak_detection()

    if args.rotation_peaks or args.run_all:
        plot_rotation_peaks()

    if args.transform_summary or args.run_all:
        plot_transform_summary()

    if args.run_all:
        plot_input_data(N)

    print(f"\nFigures saved to: {FIGS_DIR}")
