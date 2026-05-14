#!/usr/bin/env python3
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- CONFIG ---
which_rotation = 0              # which rotation candidate (for all_solutions mode)
use_rotation_suffix = True      # True = "Rotated0.csv" (all_solutions), False = "Rotated.csv" (one_solution)
save_figs = True                # save figures to disk
figs_subdir = "figs"            # subdirectory for saved figures

# --- DATA DIR ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FIGS_DIR = os.path.join(DATA_DIR, figs_subdir)
os.makedirs(FIGS_DIR, exist_ok=True)


def load_csv(filename):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    return np.loadtxt(filepath)


def volume_viewer(volume_3d, fig_name=None, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mid_slice = volume_3d.shape[0] // 2
    x, y = np.meshgrid(np.arange(volume_3d.shape[1]), np.arange(volume_3d.shape[2]))
    ax.plot_surface(x, y, volume_3d[mid_slice, :, :], cmap='viridis')
    if title:
        ax.set_title(title)
    if save_figs and fig_name:
        path = os.path.join(FIGS_DIR, fig_name)
        fig.savefig(path + ".png", dpi=150)
        fig.savefig(path + ".pdf")
        plt.close(fig)
        print(f"  Saved: {path}.png, {path}.pdf")
    else:
        plt.close(fig)


# --- Build rotated filenames from config ---
if use_rotation_suffix:
    rot_suffix = str(which_rotation)
else:
    rot_suffix = ""

# --- Load voxel data (N x N x N) ---
magnitude_fftw1 = load_csv("magnitudeFFTW1.csv")
phase_fftw1 = load_csv("phaseFFTW1.csv")
voxel_data_used1 = load_csv("voxelDataFFTW1.csv")

magnitude_fftw2 = load_csv("magnitudeFFTW2.csv")
phase_fftw2 = load_csv("phaseFFTW2.csv")
voxel_data_used2 = load_csv("voxelDataFFTW2.csv")

spectrum_real_fftw2_rotated = load_csv(f"spectrumRealFFTW2Rotated{rot_suffix}.csv")
spectrum_imag_fftw2_rotated = load_csv(f"spectrumImagFFTW2Rotated{rot_suffix}.csv")
voxel_data_used2_rotated = load_csv(f"voxelDataFFTW2Rotated{rot_suffix}.csv")

# --- Infer N and correlationN ---
if voxel_data_used1 is not None:
    N = int(round(len(voxel_data_used1) ** (1.0 / 3.0)))
    print(f"Inferred N = {N}")
else:
    raise ValueError("Cannot determine N: voxelDataFFTW1.csv not found")

magnitude_fftw2_rotated = np.sqrt(spectrum_real_fftw2_rotated ** 2 + spectrum_imag_fftw2_rotated ** 2)
phase_fftw2_rotated = np.arctan2(spectrum_imag_fftw2_rotated, spectrum_real_fftw2_rotated)
correlationN = int(round(len(magnitude_fftw2_rotated) ** (1.0 / 3.0)))
print(f"Inferred correlationN = {correlationN}  (expected N*2-1 = {N*2-1})")

# --- Reshape 3D data (C++ writes i,j,k order = row-major [i][j][k]) ---
magnitude1 = magnitude_fftw1.reshape((N, N, N))
phase1 = phase_fftw1.reshape((N, N, N))
voxel_data1 = voxel_data_used1.reshape((N, N, N))
magnitude2 = magnitude_fftw2.reshape((N, N, N))
phase2 = phase_fftw2.reshape((N, N, N))
voxel_data2 = voxel_data_used2.reshape((N, N, N))
magnitude2_rotated = magnitude_fftw2_rotated.reshape((correlationN, correlationN, correlationN))
phase2_rotated = phase_fftw2_rotated.reshape((correlationN, correlationN, correlationN))
voxel_data2_rotated = voxel_data_used2_rotated.reshape((N, N, N))

# --- Resampled magnitude (N x N) ---
resampled_magnitude1 = load_csv("resampledMagnitudeSO3_1.csv")
resampled_magnitude2 = load_csv("resampledMagnitudeSO3_2.csv")

resampled_magnitude1_matrix = resampled_magnitude1.reshape((N, N))
resampled_magnitude2_matrix = resampled_magnitude2.reshape((N, N))

# --- Plot resampled magnitudes ---
fig1, ax1 = plt.subplots(figsize=(6, 5))
im1 = ax1.imshow(resampled_magnitude1_matrix, aspect='equal', cmap='viridis')
ax1.set_title("Resampled Magnitude 1")
fig1.colorbar(im1, ax=ax1)
fig1.tight_layout()
if save_figs:
    p = os.path.join(FIGS_DIR, "resampled_magnitude_1")
    fig1.savefig(p + ".png", dpi=150)
    fig1.savefig(p + ".pdf")
    print(f"  Saved: {p}.png, {p}.pdf")
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(6, 5))
im2 = ax2.imshow(resampled_magnitude2_matrix, aspect='equal', cmap='viridis')
ax2.set_title("Resampled Magnitude 2")
fig2.colorbar(im2, ax=ax2)
fig2.tight_layout()
if save_figs:
    p = os.path.join(FIGS_DIR, "resampled_magnitude_2")
    fig2.savefig(p + ".png", dpi=150)
    fig2.savefig(p + ".pdf")
    print(f"  Saved: {p}.png, {p}.pdf")
plt.close(fig2)

# --- Sphere surface plot (signal 1) ---
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
u = np.linspace(0, 2 * np.pi, N)
v = np.linspace(0, np.pi, N)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
norm1 = resampled_magnitude1_matrix[:-1, :-1] / resampled_magnitude1_matrix.max()
ax3.plot_surface(x_sphere, y_sphere, z_sphere,
                 facecolors=plt.cm.viridis(norm1),
                 edgecolor='none')
ax3.set_title("Sphere: Resampled Magnitude 1")
ax3.set_aspect('equal')
if save_figs:
    p = os.path.join(FIGS_DIR, "sphere_magnitude_1")
    fig3.savefig(p + ".png", dpi=150)
    fig3.savefig(p + ".pdf")
    print(f"  Saved: {p}.png, {p}.pdf")
plt.close(fig3)

# --- Sphere surface plot (signal 2) ---
fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')
norm2 = resampled_magnitude2_matrix[:-1, :-1] / resampled_magnitude2_matrix.max()
ax4.plot_surface(x_sphere, y_sphere, z_sphere,
                 facecolors=plt.cm.viridis(norm2),
                 edgecolor='none')
ax4.set_title("Sphere: Resampled Magnitude 2")
ax4.set_aspect('equal')
if save_figs:
    p = os.path.join(FIGS_DIR, "sphere_magnitude_2")
    fig4.savefig(p + ".png", dpi=150)
    fig4.savefig(p + ".pdf")
    print(f"  Saved: {p}.png, {p}.pdf")
plt.close(fig4)

# --- SO(3) correlation (infer bwOut from data size) ---
correlation_values_real = load_csv("resultingCorrelationReal.csv")
correlation_values_complex = load_csv("resultingCorrelationComplex.csv")

corr_side = int(round(len(correlation_values_real) ** (1.0 / 3.0)))
bw_out = corr_side // 2
print(f"Inferred correlation grid: {corr_side}x{corr_side}x{corr_side}  (bwOut = {bw_out})")

correlation_values_real_matrix = correlation_values_real.reshape((corr_side, corr_side, corr_side))
correlation_values_complex_matrix = correlation_values_complex.reshape((corr_side, corr_side, corr_side))
correlation_values_magnitude = np.sqrt(
    correlation_values_real_matrix ** 2 + correlation_values_complex_matrix ** 2
)

print(f"Correlation magnitude - max: {correlation_values_magnitude.max():.6f}, min: {correlation_values_magnitude.min():.6f}")
volume_viewer(correlation_values_magnitude, fig_name=f"correlation_so3_mag_rot{rot_suffix}",
              title=f"SO(3) Correlation Magnitude (rot {which_rotation})")

# --- Translation correlation ---
translation_correlation = load_csv(f"resultingCorrelationShift{rot_suffix}.csv")
corrN = int(round(len(translation_correlation) ** (1.0 / 3.0)))
print(f"Translation correlation grid: {corrN}x{corrN}x{corrN}")

translation_correlation_3d = translation_correlation.reshape((corrN, corrN, corrN))

volume_viewer(translation_correlation_3d, fig_name=f"translation_corr_rot{rot_suffix}",
              title=f"Translation Correlation (rot {which_rotation})")

# --- Peak detection ---
c = np.max(translation_correlation_3d.flatten())
i = np.argmax(translation_correlation_3d.flatten())
i1, i2, i3 = np.unravel_index(i, translation_correlation_3d.shape)
print(f"\n=== Translation Peak (rotation {which_rotation}) ===")
print(f"  Max correlation: {c:.6f}")
print(f"  Peak index: ({i1}, {i2}, {i3})")
print(f"  Correlation grid size: {corrN} (voxel size: {N})")

# Estimate translation in voxel units (shift from center)
center = corrN // 2
trans_x = i1 - center
trans_y = i2 - center
trans_z = i3 - center
print(f"  Translation (voxel units from center): ({trans_x}, {trans_y}, {trans_z})")

print(f"\nFigures saved to: {FIGS_DIR}")
