#!/usr/bin/env python3
"""
PyVista-based 3D visualization for SO(3) correlation and translation data.

Opens interactive VTK windows with:
  - Interactive isosurface with slider (drag to adjust threshold)
  - Interactive orthogonal slice planes (drag to slice through volume)
  - Peak marker (red sphere at max-correlation location)
  - Optional volume rendering window

Run: python3 plot_correlation_pyvista.py
"""
import os
import numpy as np
import vtk
import pyvista as pv

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
which_rotation = 0              # which rotation candidate (for all_solutions mode)
use_rotation_suffix = True      # True = "Rotated0.csv" (all_solutions), False = "Rotated.csv"
isosurface_opacity = 0.6        # isosurface opacity (0.0–1.0)
slice_opacity = 0.5             # slice plane opacity (0.0–1.0)
show_slices = True              # add interactive orthogonal slice planes
show_volume_rendering = False   # add separate volume-rendering window (slow for large grids)

# ---------------------------------------------------------------------------
# DATA DIR
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_csv(filename):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    return np.loadtxt(filepath)


# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
if use_rotation_suffix:
    rot_suffix = str(which_rotation)
else:
    rot_suffix = ""

print(f"Loading data from: {DATA_DIR}")
print(f"Rotation candidate: {which_rotation}")

# --- SO(3) correlation ---
correlation_values_real = load_csv("resultingCorrelationReal.csv")
correlation_values_complex = load_csv("resultingCorrelationComplex.csv")

corr_side = int(round(len(correlation_values_real) ** (1.0 / 3.0)))
print(f"Correlation grid: {corr_side}x{corr_side}x{corr_side}")

correlation_values_real_matrix = correlation_values_real.reshape(
    (corr_side, corr_side, corr_side)
)
correlation_values_complex_matrix = correlation_values_complex.reshape(
    (corr_side, corr_side, corr_side)
)
correlation_values_magnitude = np.sqrt(
    correlation_values_real_matrix ** 2 + correlation_values_complex_matrix ** 2
)

corr_max = correlation_values_magnitude.max()
corr_mean = correlation_values_magnitude.mean()
corr_std = correlation_values_magnitude.std()
corr_threshold = corr_mean + corr_std
print(
    f"  Magnitude - max: {corr_max:.4f}, mean+1std: {corr_threshold:.4f}"
)

peak_idx = np.argmax(correlation_values_magnitude)
peak_z, peak_y, peak_x = np.unravel_index(
    peak_idx, correlation_values_magnitude.shape
)
print(
    f"  Peak at ({peak_x}, {peak_y}, {peak_z}) = "
    f"{correlation_values_magnitude[peak_z, peak_y, peak_x]:.4f}"
)

# --- Translation correlation ---
translation_correlation = load_csv(f"resultingCorrelationShift{rot_suffix}.csv")
corrN = int(round(len(translation_correlation) ** (1.0 / 3.0)))
print(f"Translation grid: {corrN}x{corrN}x{corrN}")

translation_correlation_3d = translation_correlation.reshape(
    (corrN, corrN, corrN)
)

trans_max = translation_correlation_3d.max()
trans_idx = np.argmax(translation_correlation_3d)
trans_z, trans_y, trans_x = np.unravel_index(
    trans_idx, translation_correlation_3d.shape
)
center = corrN // 2
print(f"  Peak at ({trans_x}, {trans_y}, {trans_z}) = {trans_max:.4f}")
print(
    f"  Translation from center: "
    f"({trans_x - center}, {trans_y - center}, {trans_z - center})"
)

# ---------------------------------------------------------------------------
# HELPER: build a PyVista ImageData from a 3D numpy array
# ---------------------------------------------------------------------------
def array_to_grid(data_3d):
    """Convert (z, y, x) numpy array to PyVista ImageData."""
    grid = pv.ImageData()
    nz, ny, nx = data_3d.shape
    grid.dimensions = (nx, ny, nz)
    grid.origin = (0, 0, 0)
    grid.spacing = (1, 1, 1)
    grid.point_data["values"] = data_3d.flatten(order="F")
    return grid


# ---------------------------------------------------------------------------
# HELPER: interactive isosurface + slice viewer
# ---------------------------------------------------------------------------
def plot_interactive_isosurface(data_3d, title, peak_position, threshold,
                                do_slices=True):
    """Open an interactive PyVista window.

    Features:
      - vtkContourFilter + slider: single isosurface, drag slider to adjust
      - add_mesh_slice_orthogonal: draggable slice planes (x, y, z)
      - red sphere at the peak location
    """
    grid = array_to_grid(data_3d)

    data_min = float(data_3d.min())
    data_max = float(data_3d.max())

    print(f"\n{title}")
    print(f"  Isosurface initial value: {threshold:.4f}")
    print(f"  Data range: [{data_min:.4f}, {data_max:.4f}]")

    # --- Contour filter (single isosurface) ---
    contour_alg = vtk.vtkContourFilter()
    contour_alg.SetInputData(grid)
    contour_alg.SetInputArrayToProcess(
        0, 0, 0,
        vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
        "values",
    )
    contour_alg.SetNumberOfContours(1)
    contour_alg.SetValue(0, threshold)

    plotter = pv.Plotter()

    # Add contour algorithm directly so slider updates it in place
    plotter.add_mesh(
        contour_alg,
        color="white",
        opacity=isosurface_opacity,
        show_scalar_bar=False,
    )

    # --- Slider to control isovalue ---
    def update_iso(value):
        contour_alg.SetValue(0, value)
        contour_alg.Update()

    plotter.add_slider_widget(
        callback=update_iso,
        rng=[data_min, data_max],
        value=threshold,
        title="Isosurface Value",
    )

    # --- Peak marker ---
    peak_sphere = pv.Sphere(
        center=peak_position, radius=1.5,
        theta_resolution=16, phi_resolution=16,
    )
    plotter.add_mesh(peak_sphere, color="red", opacity=0.9)

    # --- Interactive slice planes ---
    if do_slices:
        plotter.add_mesh_slice_orthogonal(
            grid,
            scalars="values",
            opacity=slice_opacity,
            cmap="viridis",
        )

    # --- Axes & grid ---
    plotter.add_box_axes()
    plotter.show_grid()
    plotter.add_title(title, font_size=14)

    plotter.show()


# ---------------------------------------------------------------------------
# HELPER: volume rendering viewer
# ---------------------------------------------------------------------------
def plot_volume_rendering(data_3d, title, threshold):
    """Open a PyVista window with volume rendering (opacity transfer function)."""
    grid = array_to_grid(data_3d)
    data_max = data_3d.max()

    print(f"\n{title} (volume rendering)")

    plotter = pv.Plotter()

    # Build opacity transfer function: transparent below threshold, opaque above
    tf = pv.PiecewiseFunction()
    tf.add_point(threshold, 0.0)
    tf.add_point(threshold + (data_max - threshold) * 0.1, 0.3)
    tf.add_point(data_max, 0.7)

    plotter.add_volume(
        grid,
        scalars="values",
        cmap="hot",
        opacity=tf,
        clipping_planes=False,
    )

    plotter.add_box_axes()
    plotter.add_title(title + " — Volume Rendering", font_size=14)

    plotter.show()


# ---------------------------------------------------------------------------
# VIZ 1 — SO(3) Correlation Magnitude
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Opening SO(3) Correlation viewer ...")
print("=" * 60)
plot_interactive_isosurface(
    data_3d=correlation_values_magnitude,
    title=f"SO(3) Correlation Magnitude (rot {which_rotation})",
    peak_position=(peak_x, peak_y, peak_z),
    threshold=corr_threshold,
    do_slices=show_slices,
)

if show_volume_rendering:
    plot_volume_rendering(
        data_3d=correlation_values_magnitude,
        title=f"SO(3) Correlation (rot {which_rotation})",
        threshold=corr_threshold,
    )

# ---------------------------------------------------------------------------
# VIZ 2 — Translation Correlation
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Opening Translation Correlation viewer ...")
print("=" * 60)
plot_interactive_isosurface(
    data_3d=translation_correlation_3d,
    title=f"Translation Correlation (rot {which_rotation})",
    peak_position=(trans_x, trans_y, trans_z),
    threshold=0.5,
    do_slices=show_slices,
)

if show_volume_rendering:
    plot_volume_rendering(
        data_3d=translation_correlation_3d,
        title=f"Translation Correlation (rot {which_rotation})",
        threshold=0.5,
    )

print("\nDone. All viewers closed.")
