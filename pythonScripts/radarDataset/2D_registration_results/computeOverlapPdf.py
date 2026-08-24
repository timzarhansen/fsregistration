################################################################################
#
# Overlap PDF generator for pairComparison results
#
# Reads the stored results of one setup folder
# (2D_registration_results/pairComparison/{PAIR_INDEX}/{SEQUENCE_NAME}/)
# and writes one single-page PDF per overlap into that folder:
#     overlap_gt.pdf                         = ground truth alignment
#     overlap_{method}.pdf                   = one per registration method
#
# Each page is a dense radar-image overlap (not sparse point clouds):
#     - scan 1 rendered in red   (from input1.png, the cartesian image)
#     - scan 2 rendered in blue  (input2.png warped into scan 1's frame by the
#                                 respective transformation)  -> overlap = purple
#     - axes in meters, forward up (+y = forward), equal aspect, grid
#     - the canvas is extended so the FULL (even rotated) scan is always visible
#
# The per-method transformation is recovered by fitting a rigid transform from
# the stored aligned point clouds (green part of gt.ply / {method}.ply vs
# pointcloud2.ply). This recovers the transform used to create the results to
# well under 0.001 deg / 0.001 m, so no re-computation is needed.
#
# Usage:
#     python computeOverlapPdf.py [pair_index(s)] [sequence_name]
#
# Without arguments the config values at the top of the file are used. The
# PAIR_INDICES config is a list of pair indices to process; put -1 (or leave it
# empty) to process every pair under BASE_DIR. The list is overridable on the
# CLI (comma-separated, or "all" / "-" for everything). Pairs are processed in
# parallel using NUM_THREADS threads.
#
################################################################################

import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")   # force headless backend (thread-safe for parallel PDF generation)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================================
# CONFIGURATION - Edit these to select the setup and adjust the plots
# ============================================================================

# possible examples:
# 150,135,2010 where all work
# Only FS2D: 1020 1005 3440
# only FS2D fails: 2725 2730 2135
# List of pair indices to process. Put -1 (or leave the list empty) to process
# ALL pairs under BASE_DIR. You can restrict to a subset, e.g. [150, 135, 2010].
# The list is also overridable on the CLI (see Usage/Help below).
PAIR_INDICES = [150, 135, 2010, -1]          # e.g. [150], [-1], or [] for all
SEQUENCE_NAME = "boreas-2020-11-26-13-58"     # e.g. 'boreas-2020-11-26-13-58'
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pairComparison")

NUM_THREADS = max(1, (os.cpu_count() or 1) // 2)  # pairs computed simultaneously

# Image geometry - MUST match how the results were produced. The cartesian
# images are N x N pixels covering +/-RADIUS metres, so pixel_size = 2*RADIUS/N.
N = 256
RADIUS = 150.0

# Plot settings
POINT_PIXEL_SIZE = 2 * RADIUS / N   # metres per image pixel (computed)
MARGIN_M = 0.2                      # metres of padding between scan and plot border (lower = tighter)
FIG_W, FIG_H = 9.0, 8.0             # figure size in inches
TITLE_FONT_SIZE = 10
FIG_DPI = 200                       # resolution of rasterized parts (higher = crisper)
SHOW_TITLE = True                   # set False to hide the "Pair ... / method" title on each page

# Colors / appearance
SCAN1_COLOR = (255, 0, 0)       # scan 1 (reference), RGB tuple 0-255
SCAN2_COLOR = (0, 0, 255)       # scan 2 (aligned),   RGB tuple 0-255
BACKGROUND_COLOR = (255, 255, 255)  # background where there is no radar data
SCAN1_ALPHA = 1.0               # opacity of scan 1 (0..1)
SCAN2_ALPHA = 1.0               # opacity of scan 2 (0..1)
GAMMA = 1.0                    # contrast boost on scan intensities before coloring (lower = brighter faint returns)
INTERPOLATION = "nearest"      # "nearest" | "bilinear" (reduces pixelation)

# Blend mode
BLEND_MODE = "points"          # "points" = red/blue point clouds with transparency (pcshowpair style)
                                # "average" = grayscale mosaic of the true radar intensities
                                # "color"   = red (scan1) / blue (scan2) / purple (overlap)
SHOW_SCAN_ORIGINS = True        # green dots at each scan's world origin
GRAY_GAMMA = 0.35               # tone-map for the grayscale average mode (lower = brighter faint returns)

# Heading arrows (make the rotation visible next to the origin dots)
SHOW_HEADING_ARROWS = True      # small arrows at each scan's origin pointing along its heading
ARROW_LENGTH_FRAC = 0.1        # arrow length as a fraction of the axis half-extent (scales with zoom)
ARROW_LW = 1.9                  # arrow line width in points
ARROW_HEAD_MUTATION = 17        # arrow head size in points
SHOW_YAW_ANGLE = False          # print the scan-2 yaw offset in deg next to its arrow
HEADING_ARROW_COLOR_SCAN1 = "forestgreen"  # scan 1 origin dot + heading arrow (was lime, too light)
HEADING_ARROW_COLOR_SCAN2 = "magenta"      # scan 2 origin dot + heading arrow

# 'points' mode settings
POINT_THRESHOLD = 0.05          # cartesian-image intensity above which a pixel becomes a point
POINT_SIZE = 8                   # scatter marker size
POINT_ALPHA = 0.4               # transparency (overlap shows as red+blue -> magenta)

# Per-point intensity styling (how each point's radar return strength changes it)
INTENSITY_STYLE = "alpha"       # "alpha" = strong returns more opaque | "size" = strong = bigger
                                # "color" = strong = saturated, weak fade to white | "off" = flat color
INTENSITY_GAMMA = 0.6           # gamma on normalized intensity (0..1); strength of the effect
INTENSITY_FLOOR = 0.15          # minimum factor, so weak returns stay slightly visible
# ============================================================================


def read_ply_binary(path: str):
    """Read a binary little-endian PLY file as written by comparisonBoreasPais.py.

    Returns:
        pts: (N, 3) float64 coordinates (x, y, z)
        colors: (N, 3) float64 RGB in [0, 1]
    """
    with open(path, "rb") as f:
        data = f.read()
    header_end = data.find(b"end_header")
    if header_end < 0:
        raise ValueError(f"'{path}' is not a PLY file (no end_header found)")
    header = data[:header_end].decode("ascii")
    body = data[header_end + len(b"end_header\n"):]

    n = None
    for line in header.splitlines():
        if line.startswith("element vertex"):
            n = int(line.split()[-1])
    if n is None:
        raise ValueError(f"'{path}' has no 'element vertex' in its header")

    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                      ('intensity', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    arr = np.frombuffer(body, dtype=dtype, count=n)
    if len(arr) != n:
        raise ValueError(f"'{path}' is truncated (expected {n} vertices)")

    pts = np.column_stack([arr['x'], arr['y']]).astype(np.float64)
    colors = np.column_stack([arr['red'], arr['green'], arr['blue']]).astype(np.float64) / 255.0
    return pts, colors


def green_mask(colors: np.ndarray) -> np.ndarray:
    """Points whose color is predominantly green (the aligned scan 2)."""
    return (colors[:, 1] > 0.5) & (colors[:, 0] < 0.5) & (colors[:, 2] < 0.5)


def fit_transform_pc(orig: np.ndarray, aligned: np.ndarray):
    """Fit a 2D rigid transform R, t (pc frame) mapping orig -> aligned (Umeyama)."""
    c1, c2 = orig.mean(0), aligned.mean(0)
    H = (orig - c1).T @ (aligned - c2)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = c2 - R @ c1
    return R, t


def to_vehicle_transform(R2: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Convert a pc-frame rigid transform to a vehicle-frame 4x4 (z-rotation).

    pc->veh is P = [[0,-1],[-1,0]] (veh_x = -pc_y, veh_y = -pc_x), so
    T_veh = P @ T_pc @ P (P is its own inverse).
    """
    P = np.array([[0., -1.], [-1., 0.]])
    Rv = P @ R2 @ P
    tv = P @ t2
    T = np.eye(4)
    T[:2, :2] = Rv
    T[:2, 3] = tv
    return T


def recovery_transform(pair_dir: str, aligned_ply: str):
    """Recover the vehicle-frame transform that aligned scan 2 into scan 1.

    Fits the rigid transform from the stored aligned points (green part of
    aligned_ply) back to the original scan 2 points (pointcloud2.ply).
    """
    p2, _ = read_ply_binary(os.path.join(pair_dir, "pointcloud2.ply"))
    a_pts, a_col = read_ply_binary(os.path.join(pair_dir, aligned_ply))
    al = a_pts[green_mask(a_col)]
    if len(al) < 4 or len(p2) < 4:
        raise ValueError(f"Not enough points to recover transform from {aligned_ply}")
    R, t = fit_transform_pc(p2, al)
    return to_vehicle_transform(R, t)


def dense_warp(img: np.ndarray, T_veh, pixel_size: float, N: int, xs, ys):
    """Sample img (a cartesian image) onto a display pixel grid given by xs (cols,
    metres) and ys (rows, metres, forward = +).

    If T_veh is None the image is used as-is (identity -> its own frame).
    Otherwise each display pixel's vehicle coords are mapped back into the
    source frame via inv(T_veh) before sampling (inverse warp).
    """
    X, Y = np.meshgrid(xs, ys)  # X varies with column, Y varies with row
    xv = X.ravel()
    yv = Y.ravel()
    if T_veh is not None:
        hom = np.column_stack([xv, yv, np.zeros_like(xv), np.ones_like(xv)])
        q = (np.linalg.inv(T_veh) @ hom.T).T[:, :2]
        xv, yv = q[:, 0], q[:, 1]
    # cartesian image pixel coords: row = N/2 - x_veh/cs (forward up),
    # col = N/2 + y_veh/cs (left right)
    col = N / 2.0 + yv / pixel_size
    row = N / 2.0 - xv / pixel_size
    mapx = col.reshape(X.shape).astype(np.float32)
    mapy = row.reshape(X.shape).astype(np.float32)
    return cv2.remap(img.astype(np.float32), mapx, mapy, cv2.INTER_LINEAR, borderValue=0.0)


def read_meta(pair_dir) -> tuple:
    """Read registration_meta.csv -> ({method: row}, frame1, frame2)."""
    csv_path = os.path.join(pair_dir, "registration_meta.csv")
    meta = {}
    frame1 = frame2 = None
    if os.path.isfile(csv_path):
        with open(csv_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if lines:
            header = lines[0].split(",")
            for line in lines[1:]:
                row = dict(zip(header, line.split(",")))
                meth = row.get("method", "")
                frame1, frame2 = row.get("frame1", frame1), row.get("frame2", frame2)
                meta[meth] = row
    return meta, frame1, frame2


def build_pages(pair_dir):
    """Collect (label, title, T_veh, input1, input2) for GT and each method."""
    global N
    meta, frame1, frame2 = read_meta(pair_dir)
    pair_label = f"Pair {frame1} -> {frame2}" if frame1 is not None else f"Pair {os.path.basename(pair_dir)}"

    img1 = cv2.imread(os.path.join(pair_dir, "input1.png"), 0).astype(np.float64) / 255.0
    img2 = cv2.imread(os.path.join(pair_dir, "input2.png"), 0).astype(np.float64) / 255.0
    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Missing input1.png / input2.png in {pair_dir}")
    N = img1.shape[0]

    pages = []

    gt_path = os.path.join(pair_dir, "gt.ply")
    if os.path.isfile(gt_path):
        pages.append(("gt", f"{pair_label}\nGT", recovery_transform(pair_dir, "gt.ply"), img1, img2))
    else:
        print(f"  [WARN] no gt.ply in {pair_dir} - skipping GT page")

    methods = list(meta.keys())
    if not methods:
        fnames = sorted(os.listdir(pair_dir))
        methods = [f[:-4] for f in fnames if f.endswith(".ply") and f not in ("gt.ply", "pointcloud1.ply", "pointcloud2.ply")]
    for meth in methods:
        meth_path = os.path.join(pair_dir, f"{meth}.ply")
        if not os.path.isfile(meth_path):
            print(f"  [WARN] no {meth}.ply in {pair_dir} - skipping page")
            continue
        row = meta.get(meth, {})
        err = ""
        if row and row.get("rot_err_deg") and row.get("trans_err_m"):
            err = f"  (rot err {float(row['rot_err_deg']):.2f}°, trans err {float(row['trans_err_m']):.2f} m)"
        try:
            T = recovery_transform(pair_dir, f"{meth}.ply")
        except Exception as e:
            print(f"  [WARN] could not recover transform for '{meth}': {e} - skipping page")
            continue
        pages.append((meth, f"{pair_label}\n{meth}{err}", T, img1, img2))
    return pages


def scan_extent(pair_dir: str, T_veh) -> float:
    """Metre half-extent needed to show scan 1 and the full rotated scan 2."""
    p1, _ = read_ply_binary(os.path.join(pair_dir, "pointcloud1.ply"))
    p2, _ = read_ply_binary(os.path.join(pair_dir, "pointcloud2.ply"))
    r1 = np.max(np.hypot(p1[:, 0], p1[:, 1]))
    r2 = np.max(np.hypot(p2[:, 0], p2[:, 1]))
    th = np.arctan2(T_veh[1, 0], T_veh[0, 0])
    blue_extent = r2 * (abs(np.cos(th)) + abs(np.sin(th)))  # rotated square bbox
    return max(RADIUS, r1, blue_extent) + MARGIN_M


def composite_average(scan1, scan2):
    """Grayscale mosaic: average the true radar intensities in shared world space.

    Where both scans cover a pixel the average is shown (preserving detail),
    where only one covers it that scan is shown, and empty areas are white.
    Contrast auto-stretched to full range (like MATLAB's imshow([],)).
    Returns a uint8 RGB image.
    """
    bg = np.array(BACKGROUND_COLOR, np.uint8)
    p1 = scan1 > 0.0
    p2 = scan2 > 0.0
    cnt = p1.astype(np.float64) + p2.astype(np.float64)
    avg = np.where(cnt > 0, (scan1 + scan2) / np.maximum(cnt, 1.0), 0.0)
    mx = avg.max()
    if mx > 0:
        avg = avg / mx
    # Tone-map the grayscale so faint radar returns are visible (radar images are
    # heavily skewed toward low values; linear max-stretch leaves them near black).
    avg = np.power(np.clip(avg, 0.0, 1.0), GRAY_GAMMA)
    gray = (np.clip(avg, 0, 1) * 255.0).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], -1)
    rgb[cnt == 0] = bg
    return rgb


def composite(scan1, scan2):
    """Blend the two intensity images (0..1) into a colored RGB image.

    - no data in either  -> BACKGROUND_COLOR
    - scan 1 only        -> SCAN1_COLOR
    - scan 2 only        -> SCAN2_COLOR
    - overlap            -> additive mix of the two colors (e.g. magenta)
    - weak content       -> soft pastel of the scan color (never pure black)
    """
    c1 = np.array(SCAN1_COLOR, np.float64)
    c2 = np.array(SCAN2_COLOR, np.float64)
    bg = np.array(BACKGROUND_COLOR, np.float64)
    # Normalize each scan to its own max so the strongest return is full-intensity
    # (the cartesian images peak at ~0.5, so without this everything is faint).
    s1, s2 = scan1, scan2
    for a in (s1, s2):
        m = a.max()
        if m > 0:
            a /= m
    f1 = (s1 ** GAMMA) * SCAN1_ALPHA
    f2 = (s2 ** GAMMA) * SCAN2_ALPHA
    ink = c1 * f1[..., None] + c2 * f2[..., None]   # additive color contribution
    activity = np.clip(f1 + f2, 0.0, 1.0)[..., None]
    rgb = bg + (ink - bg) * activity
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def image_to_points(img, pixel_size, N, threshold=0.2):
    """Convert a cartesian image to a 2D point cloud in metres (x = forward, y = left).

    Returns (pts, intensity): pts is (M, 2) world coords, intensity is (M,) the
    image value at each point (before thresholding stays > threshold).
    """
    ys, xs = np.where(img > threshold)
    x_veh = (N / 2.0 - ys) * pixel_size
    y_veh = (xs - N / 2.0) * pixel_size
    inten = img[ys, xs].astype(np.float64)
    return np.column_stack([x_veh, y_veh]), inten


def _scatter_scan(ax, pts, inten, cbase, label):
    """Scatter one scan, styling each point by its intensity (see INTENSITY_STYLE)."""
    if INTENSITY_STYLE == "off":
        ax.scatter(pts[:, 0], pts[:, 1], s=POINT_SIZE, color=cbase / 255.0, marker=".",
                   alpha=POINT_ALPHA, rasterized=True, label=label)
        return
    m = inten.max()
    g = (inten / m if m > 0 else inten)
    g = np.clip(g, 0.0, 1.0) ** INTENSITY_GAMMA
    fac = INTENSITY_FLOOR + (1.0 - INTENSITY_FLOOR) * g
    if INTENSITY_STYLE == "alpha":
        ax.scatter(pts[:, 0], pts[:, 1], s=POINT_SIZE, color=cbase / 255.0, marker=".",
                   alpha=POINT_ALPHA * fac, rasterized=True, label=label)
    elif INTENSITY_STYLE == "size":
        ax.scatter(pts[:, 0], pts[:, 1], s=POINT_SIZE * (0.5 + g), color=cbase / 255.0, marker=".",
                   alpha=POINT_ALPHA, rasterized=True, label=label)
    elif INTENSITY_STYLE == "color":
        white = np.array([255.0, 255.0, 255.0])
        col = np.clip(white + (cbase - white) * g[:, None], 0.0, 255.0) / 255.0
        ax.scatter(pts[:, 0], pts[:, 1], s=POINT_SIZE, c=col, marker=".",
                   alpha=POINT_ALPHA, rasterized=True, label=label)


def _plot_points(ax, img1, img2, T_veh):
    """Red/blue point-cloud overlay with transparency (pcshowpair style).

    The axis extent is auto-fitted to the actual drawn points so the content
    hugs the plot border (instead of relying on the raw-cloud radius).
    """
    p1, i1 = image_to_points(img1, POINT_PIXEL_SIZE, N, POINT_THRESHOLD)
    p2, i2 = image_to_points(img2, POINT_PIXEL_SIZE, N, POINT_THRESHOLD)
    # Align scan 2 into scan 1's frame.
    hom = np.column_stack([p2, np.zeros(len(p2)), np.ones(len(p2))])
    p2a = (np.asarray(T_veh, np.float64) @ hom.T).T[:, :2]
    _scatter_scan(ax, p1, i1, np.array(SCAN1_COLOR, np.float64), "scan 1")
    _scatter_scan(ax, p2a, i2, np.array(SCAN2_COLOR, np.float64), "scan 2")

    # Tight auto-fit to the plotted content (center it and buffer by MARGIN_M).
    allp = np.vstack([p1, p2a])
    cx = (allp[:, 0].min() + allp[:, 0].max()) / 2.0
    cy = (allp[:, 1].min() + allp[:, 1].max()) / 2.0
    half = max(allp[:, 0].max() - allp[:, 0].min(), allp[:, 1].max() - allp[:, 1].min()) / 2.0 + MARGIN_M
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.margins(0.0)  # keep the scan close to the plot border (no matplotlib auto-margin)
    ax.legend(loc="upper right", fontsize=8)


def _scan2_heading(T_veh):
    """Direction (dx, dy) of scan 2's forward axis after alignment by T_veh.

    Vehicle +x is forward and the plot uses the same (x, y) convention as the
    data, so an arrow drawn along this direction shows the yaw offset directly
    (angle relative to the unrotated scan 1 heading).
    """
    return T_veh[:2, :2] @ np.array([1.0, 0.0])


def _draw_heading_arrow(ax, ox, oy, dx, dy, color):
    """Small vector arrow at (ox, oy) pointing along unit vector (dx, dy).

    Length is a fraction of the current axis half-extent so it scales
    automatically in both plot modes (auto-fitted points, fixed density).
    Drawn as an annotate arrow -> stays crisp vector graphics in the PDF.
    """
    half = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 2.0
    L = ARROW_LENGTH_FRAC * half
    ax.annotate("", xy=(ox + L * dx, oy + L * dy), xytext=(ox, oy),
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", lw=ARROW_LW, color=color,
                                mutation_scale=ARROW_HEAD_MUTATION))


def plot_overlap(ax, title, pair_dir, T_veh, img1, img2):
    """Draw one overlap page on axis ax."""
    if BLEND_MODE == "points":
        _plot_points(ax, img1, img2, T_veh)
    else:
        extent = scan_extent(pair_dir, T_veh)
        M = int(np.ceil(2 * extent / POINT_PIXEL_SIZE))
        eer = M * POINT_PIXEL_SIZE / 2.0  # exact half-extent of the pixel grid
        xs = np.linspace(-eer, eer, M)
        ys = np.linspace(-eer, eer, M)
        red = dense_warp(img1, None, POINT_PIXEL_SIZE, N, xs, ys)       # scan 1
        blue = dense_warp(img2, T_veh, POINT_PIXEL_SIZE, N, xs, ys)     # scan 2 aligned
        rgb = composite(red, blue) if BLEND_MODE == "color" else composite_average(red, blue)
        ax.imshow(rgb, extent=[-eer, eer, -eer, eer], origin="lower", interpolation=INTERPOLATION)
        ax.set_xlim(-eer, eer)
        ax.set_ylim(-eer, eer)

    if SHOW_SCAN_ORIGINS:
        ax.plot(0.0, 0.0, "o", color=HEADING_ARROW_COLOR_SCAN1, ms=6, mec="k", label="scan 1 origin")
        ax.plot(T_veh[0, 3], T_veh[1, 3], "o", color=HEADING_ARROW_COLOR_SCAN2, ms=6, mec="k", label="scan 2 origin")
    if SHOW_HEADING_ARROWS:
        # scan 1 heading is the reference (no rotation); scan 2's heading arrow
        # is rotated by T_veh, so the angle between the two shows the yaw.
        _draw_heading_arrow(ax, 0.0, 0.0, 1.0, 0.0, HEADING_ARROW_COLOR_SCAN1)
        dx, dy = _scan2_heading(T_veh)
        _draw_heading_arrow(ax, T_veh[0, 3], T_veh[1, 3], dx, dy, HEADING_ARROW_COLOR_SCAN2)
        if SHOW_YAW_ANGLE:
            yaw = np.degrees(np.arctan2(T_veh[1, 0], T_veh[0, 0]))
            h = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 2.0
            ax.text(T_veh[0, 3] + 0.05 * h, T_veh[1, 3] + 0.05 * h,
                    f"yaw {yaw:+.1f}°", color="magenta", fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.4)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if SHOW_TITLE:
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)


def process_setup(pair_dir):
    """Write the per-method PDFs and one combined PDF (GT + all methods) for a setup folder."""
    if not os.path.isdir(pair_dir):
        raise NotADirectoryError(f"Setup folder not found: {pair_dir}")
    pages = build_pages(pair_dir)
    if not pages:
        print(f"  [WARN] no pages for {pair_dir} - nothing to write")
        return

    combined_path = os.path.join(pair_dir, "overlap.pdf")
    with PdfPages(combined_path) as pdf:
        for label, title, T, img1, img2 in pages:
            fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=FIG_DPI)
            plot_overlap(ax, title, pair_dir, T, img1, img2)
            fig.tight_layout()
            pdf.savefig(fig)  # add as a page of the combined PDF
            pdf_path = os.path.join(pair_dir, f"overlap_{label}.pdf")
            fig.savefig(pdf_path, format="pdf")  # individual PDF
            plt.close(fig)
            print(f"  Saved {pdf_path}")
    print(f"  Saved {combined_path} ({len(pages)} pages)")


def _safe_process(pair_dir):
    """process_setup wrapped for parallel workers (prints progress per pair)."""
    print(f"Processing: {pair_dir}", flush=True)
    process_setup(pair_dir)


def main():
    global PAIR_INDICES, SEQUENCE_NAME

    # CLI overrides (backward compatible):
    #   python computeOverlapPdf.py                 -> use PAIR_INDICES config
    #   python computeOverlapPdf.py all / - / none  -> process everything
    #   python computeOverlapPdf.py 150 135         -> just those pairs
    #   python computeOverlapPdf.py 150 135 <seq>   -> pairs in a given sequence
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.lower() in ("none", "all", "-"):
            PAIR_INDICES = [-1]
        else:
            PAIR_INDICES = [int(x) for x in arg.split(",") if x.strip()]
    if len(sys.argv) > 2:
        SEQUENCE_NAME = sys.argv[2]

    # Resolve the effective list of pair indices.
    if not PAIR_INDICES or -1 in PAIR_INDICES:
        indices = sorted((d for d in os.listdir(BASE_DIR)
                          if os.path.isdir(os.path.join(BASE_DIR, d))), key=int)
    else:
        indices = sorted({int(i) for i in PAIR_INDICES})

    pair_dirs = [os.path.join(BASE_DIR, str(i), SEQUENCE_NAME) for i in indices]
    print(f"Processing {len(pair_dirs)} pair(s) with {NUM_THREADS} thread(s)")

    failures = []

    if NUM_THREADS <= 1:
        for d in pair_dirs:
            try:
                _safe_process(d)
            except Exception as e:
                print(f"  [WARN] {type(e).__name__}: {e}")
                failures.append((d, e))
    else:
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as ex:
            futs = {ex.submit(_safe_process, d): d for d in pair_dirs}
            for fut in as_completed(futs):
                d = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"  [WARN] failed {d}: {type(e).__name__}: {e}")
                    failures.append((d, e))

    if failures:
        print("\nFinished with failures:")
        for d, e in failures:
            print(f"  {d}: {type(e).__name__}: {e}")
    else:
        print("\nAll pairs processed successfully.")


if __name__ == "__main__":
    main()
