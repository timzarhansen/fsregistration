#!/usr/bin/env python3
################################################################################
#
# timeBoreasFs2dSteps.py - Per-step computation-time report for FS2D on Boreas
#
# Runs the FS2D registration (direct vs SO3 / non-direct, useDirect toggle) on
# consecutive Boreas radar pairs — same setup as the boreas benchmark (N,
# radius, matching_step, start_frame, level_potential_rotation, random azimuth
# rotation ...) — and times every step of the C++ pipeline via the benchmark
# flag (SoftRegistrationWrapper2D.register_all_solutions_timed, returns the
# BenchmarkTimings2D per-step ms values).
#
# Output (written to OUTPUT_DIR, overwritten on every run):
#   per_pair_timings.csv          per-pair per-method wide timing table
#   step_timing_aggregated.csv    mean/std/median ms per step per method
#   step_timing_summary_paper.tex LaTeX table (rows = steps)
#   fs2d_direct/results.csv       total-time results.csv per method, compatible
#   fs2d_so3/results.csv          with aggregate_and_generate_latex_timing.py
#
# Usage:
#     python timeBoreasFs2dSteps.py
#
# Edit the CONFIGURATION block at the top (MAX_FRAMES to cap the number of
# frames, e.g. 10 for a quick sanity run).
################################################################################

import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths / imports
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_script_dir))))
_install_lib = os.path.join(_root_dir, 'install', 'fsregistration', 'lib', 'fsregistration')
if os.path.isdir(_install_lib):
    sys.path.insert(0, _install_lib)

from boreasDatasetLoader import load_sequence, load_single_sequence
from pybind_registration_2d import SoftRegistrationWrapper2D

# ---------------------------------------------------------------------------
# CONFIGURATION - Edit these to change the run
# ---------------------------------------------------------------------------
DATA_DIR = "/home/tim-external/dataFolder/radar_boreas"
SEQUENCE_NUMBER = 0
SEQUENCE_NAME = None  # e.g. 'boreas-2020-11-26-13-58' (None = use SEQUENCE_NUMBER)

# Boreas benchmark setup (mirrors the timing runs that produced
# IcraPaperResults/timing_seq0, e.g. seq00_fs2d_N256_p109_s1: N=256, radius=140,
# matching_step=1, start_frame=0, level_potential_rotation=0.001, rand rot 0-20).
N = 256
RADIUS = 140.0
SIZE_OF_PIXEL = (2.0 * RADIUS) / N
MATCHING_STEP = 1
START_FRAME = 0
MAX_FRAMES = 10          # None = full sequence; use a small number to sanity check

# Which method(s) to run. Each toggles useDirect in the C++ call.
RUN_DIRECT = True        # useDirect=True  -> 1-angle direct correlation
RUN_SO3 = True           # useDirect=False -> SO(3) correlation (much slower)

# Random azimuth rotation of the CURRENT scan (bin level, before rendering),
# magnitude ~ U[RAND_ROT_MIN_DEG, RAND_ROT_MAX_DEG] with random sign, seed
# RAND_ROT_SEED + SEQUENCE_NUMBER (identical to boreasBenchmark.py). The
# rotation does not affect the timing, but keeps pairs identical to the
# benchmark. Set APPLY_RAND_ROT=False for plain consecutive pairs.
APPLY_RAND_ROT = True
RAND_ROT_MIN_DEG = 0.0
RAND_ROT_MAX_DEG = 20.0
RAND_ROT_SEED = 42

# FS2D parameters (boreas benchmark defaults)
LEVEL_POTENTIAL_ROTATION = 0.001
POTENTIAL_FOR_NECESSARY_PEAK = 0.01
USE_CLAHE = True
USE_HAMMING = True
MULTIPLE_RADII = True
USE_GAUSS = False
NORMALIZATION = 0
USE_PHASE_CORRELATION = False
R_MIN = 0.0               # 0.0 = auto N-dependent default
R_MAX = 0.0               # 0.0 = auto N-dependent default
USE_HIDDEN_COMPONENT_SCAN = False

# Report output (overwritten on every run)
OUTPUT_DIR = os.path.join(
    _script_dir, "2D_registration_results", "allDatasets", "IcraPaperResults",
    f"fs2d_step_timing_seq{SEQUENCE_NUMBER}",
)

# ---------------------------------------------------------------------------
# Steps reported (field name in the C++ BenchmarkTimings2D dict -> label)
# ---------------------------------------------------------------------------
STEPS = [
    ("spectrumTime", "FFT spectrum (both scans)"),
    ("softDescriptorTime", "SOFT descriptor / sphere projection"),
    ("rotationCorrelationTime", "Rotation correlation"),
    ("rotationExtractionTime", "Rotation curve extraction"),
    ("rotationPeakDetectionTime", "Rotation peak detection"),
    ("transPreprocessingTime", "Translation preprocessing (per angle)"),
    ("transFft1Time", "Translation FFT scan 1"),
    ("transFft2Time", "Translation FFT scan 2"),
    ("transCorrelationTime", "Translation correlation"),
    ("transIfftTime", "Translation IFFT"),
    ("transFftshiftTime", "Translation FFT shift"),
    ("transPeakDetectionTime", "Translation peak detection"),
    ("totalTranslationTime", "Translation total (all angles)"),
    ("totalTime", "Total (C++ registration)"),
]


def compute_stats(values):
    """mean / std (ddof=1) / median of finite values."""
    a = np.asarray([v for v in values
                    if v is not None and math.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "median": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)),
        "median": float(np.median(a)),
        "n": int(a.size),
    }


def main():
    if not (RUN_DIRECT or RUN_SO3):
        print("ERROR: at least one of RUN_DIRECT / RUN_SO3 must be True")
        sys.exit(1)

    print("=== FS2D per-step timing on Boreas ===")
    print(f"  N={N}, radius={RADIUS} m, pixel_size={SIZE_OF_PIXEL:.3f} m")
    print(f"  matching_step={MATCHING_STEP}, start_frame={START_FRAME}, max_frames={MAX_FRAMES}")
    print(f"  methods: {[m for m, on in (('direct', RUN_DIRECT), ('so3', RUN_SO3)) if on]}")
    print(f"  level_potential_rotation={LEVEL_POTENTIAL_ROTATION}")
    print(f"  apply_rand_rot={APPLY_RAND_ROT}"
          + (f" (U[{RAND_ROT_MIN_DEG:.1f},{RAND_ROT_MAX_DEG:.1f}] deg, seed {RAND_ROT_SEED})"
             if APPLY_RAND_ROT else ""))

    # Load sequence
    if SEQUENCE_NAME is not None:
        print(f"Loading sequence '{SEQUENCE_NAME}' from {DATA_DIR} ...")
        seq = load_single_sequence(DATA_DIR, SEQUENCE_NAME)
    else:
        print(f"Loading sequence {SEQUENCE_NUMBER} from {DATA_DIR} ...")
        seq = load_sequence(DATA_DIR, SEQUENCE_NUMBER)
    print(f"Sequence has {seq.length} radar scans")
    print()

    wrapper = SoftRegistrationWrapper2D(N)

    # Frame range (same as boreasBenchmark.run_benchmark)
    total_frames = seq.length
    if MAX_FRAMES is not None:
        end_frame = min(START_FRAME + MAX_FRAMES, total_frames)
    else:
        end_frame = total_frames
    num_pairs = 0
    if end_frame > START_FRAME + MATCHING_STEP:
        num_pairs = max(0, (end_frame - START_FRAME - 1) // MATCHING_STEP)
    print(f"Registering {num_pairs} pairs (frames {START_FRAME}..{end_frame})")
    print("=" * 80)

    rng = np.random.default_rng(RAND_ROT_SEED + SEQUENCE_NUMBER) if APPLY_RAND_ROT else None

    # per_pair_timings.csv rows
    pair_rows = []

    idx = START_FRAME + MATCHING_STEP
    pair_counter = 0
    while idx < end_frame:
        prev_idx = idx - MATCHING_STEP

        # Random azimuth rotation of the current scan (mirrors boreasBenchmark)
        if APPLY_RAND_ROT:
            mag_rad = rng.uniform(np.radians(RAND_ROT_MIN_DEG), np.radians(RAND_ROT_MAX_DEG))
            applied_rot_deg = np.degrees(mag_rad) * rng.choice([-1.0, 1.0])
        else:
            applied_rot_deg = 0.0
        applied_rot_rad = np.radians(applied_rot_deg)

        img1 = seq.get_cartesian_image(prev_idx, N, SIZE_OF_PIXEL)
        img2 = seq.get_cartesian_image(idx, N, SIZE_OF_PIXEL, azimuth_offset_rad=applied_rot_rad)

        image_1 = img1.astype(np.float64).reshape(-1)
        image_2 = img2.astype(np.float64).reshape(-1)

        for method_name, use_direct in (("direct", True), ("so3", False)):
            if (method_name == "direct" and not RUN_DIRECT) or \
               (method_name == "so3" and not RUN_SO3):
                continue

            t0 = time.time()
            peaks, timings = wrapper.register_all_solutions_timed(
                image_1, image_2,
                cellSize=SIZE_OF_PIXEL,
                useGauss=USE_GAUSS,
                debug=False,
                potentialNecessaryForPeak=POTENTIAL_FOR_NECESSARY_PEAK,
                multipleRadii=MULTIPLE_RADII,
                useClahe=USE_CLAHE,
                useHamming=USE_HAMMING,
                useDirect=use_direct,
                levelPotentialRotation=LEVEL_POTENTIAL_ROTATION,
                normalization=NORMALIZATION,
                usePhaseCorrelation=USE_PHASE_CORRELATION,
                numAngles=-1,
                r_min=R_MIN,
                r_max=R_MAX,
                useHiddenComponentScan=USE_HIDDEN_COMPONENT_SCAN,
            )
            wall_ms = (time.time() - t0) * 1000.0

            num_solutions = sum(len(p.potentialTranslations) for p in peaks)

            row = {
                "pair_idx": pair_counter,
                "prev_frame": prev_idx,
                "curr_frame": idx,
                "applied_rot_deg": applied_rot_deg,
                "method": method_name,
                "use_direct": int(use_direct),
                "wall_ms": wall_ms,
                "num_angles": timings.get("numAngles", 0),
                "total_trans_peaks": timings.get("totalTransPeaks", 0),
                "num_solutions": num_solutions,
            }
            for field, _label in STEPS:
                row[field] = timings.get(field, float("nan"))

            pair_rows.append(row)
            print(f"pair {pair_counter:4d} ({prev_idx}->{idx}) {method_name:6s}: "
                  f"total={timings.get('totalTime', float('nan')):8.1f} ms "
                  f"(wall {wall_ms:8.1f} ms), angles={row['num_angles']}, "
                  f"trans_peaks={row['total_trans_peaks']}, solutions={num_solutions}")

        pair_counter += 1
        idx += MATCHING_STEP

    if not pair_rows:
        print("No pairs processed - nothing to report.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Write outputs (overwrite the whole output dir on every run)
    # ------------------------------------------------------------------
    out_dir = Path(OUTPUT_DIR)
    if out_dir.exists():
        for f in out_dir.iterdir():
            if f.is_dir():
                for ff in f.iterdir():
                    ff.unlink()
                f.rmdir()
            else:
                f.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) per_pair_timings.csv (wide)
    header = ["pair_idx", "prev_frame", "curr_frame", "applied_rot_deg",
              "method", "use_direct", "wall_ms", "num_angles",
              "total_trans_peaks", "num_solutions"]
    header += [f for f, _ in STEPS]
    with open(out_dir / "per_pair_timings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in pair_rows:
            w.writerow(row)

    # 2) step_timing_aggregated.csv + LaTeX summary table
    methods = sorted({r["method"] for r in pair_rows})
    step_stats = {}   # (method, field) -> stats dict
    for method in methods:
        for field, _label in STEPS:
            values = [r[field] for r in pair_rows if r["method"] == method]
            step_stats[(method, field)] = compute_stats(values)

    with open(out_dir / "step_timing_aggregated.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "step", "n", "mean_ms", "std_ms", "median_ms"])
        for method in methods:
            for field, label in STEPS:
                s = step_stats[(method, field)]
                w.writerow([method, field, s["n"], f"{s['mean']:.3f}",
                            f"{s['std']:.3f}", f"{s['median']:.3f}"])

    _write_latex(out_dir, methods, step_stats)

    # 3) results.csv per method (compatible with aggregate_and_generate_latex_timing.py)
    for method in methods:
        method_dir = out_dir / f"fs2d_{method}"
        method_dir.mkdir(exist_ok=True)
        rows = [r for r in pair_rows if r["method"] == method]
        times = [r["wall_ms"] for r in rows]
        stats = compute_stats(times)
        with open(method_dir / "results.csv", "w", newline="") as f:
            f.write(f"# method: fs2d_{method}\n")
            f.write(f"# sequence: {SEQUENCE_NUMBER}\n")
            f.write(f"# N: {N}\n")
            f.write(f"# radius: {RADIUS}\n")
            f.write(f"# matching_step: {MATCHING_STEP}\n")
            f.write(f"# start_frame: {START_FRAME}\n")
            f.write(f"# total_frames: {end_frame}\n")
            f.write(f"# num_pairs_processed: {len(rows)}\n")
            f.write(f"# apply_rand_rot: {APPLY_RAND_ROT}\n")
            f.write(f"# rand_rot_seed: {RAND_ROT_SEED}\n")
            f.write(f"# rand_rot_min_deg: {RAND_ROT_MIN_DEG}\n")
            f.write(f"# rand_rot_max_deg: {RAND_ROT_MAX_DEG}\n")
            f.write(f"# level_potential_rotation: {LEVEL_POTENTIAL_ROTATION}\n")
            f.write(f"# avg_time_ms: {stats['mean']}\n")
            f.write(f"# median_time_ms: {stats['median']}\n")
            f.write("pair_idx,prev_frame,curr_frame,applied_rot_deg,"
                    "computation_time_ms,total_time_ms,num_angles,total_trans_peaks\n")
            for r in rows:
                f.write(f"{r['pair_idx']},{r['prev_frame']},{r['curr_frame']},"
                        f"{r['applied_rot_deg']:.6f},{r['wall_ms']:.6f},"
                        f"{r['totalTime']:.6f},{r['num_angles']},{r['total_trans_peaks']}\n")

    # ------------------------------------------------------------------
    # Console summary (ICRA-timing-table style)
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("Per-step computation times (ms), mean/std/median over pairs")
    print("=" * 80)
    for method in methods:
        print(f"\n--- method: {method} ---")
        print(f"{'step':<38} {'mean':>10} {'std':>10} {'median':>10}")
        for field, label in STEPS:
            s = step_stats[(method, field)]
            print(f"{label:<38} {s['mean']:10.3f} {s['std']:10.3f} {s['median']:10.3f}")

    print()
    print("=" * 80)
    print("Overall totals (per method)")
    print("=" * 80)
    print(f"{'method':<8} {'n_pairs':>8} {'mean_ms':>10} {'std_ms':>10} {'median_ms':>10}")
    for method in methods:
        s = compute_stats([r["wall_ms"] for r in pair_rows if r["method"] == method])
        print(f"{method:<8} {s['n']:>8} {s['mean']:>10.3f} {s['std']:>10.3f} {s['median']:>10.3f}")

    print()
    print(f"Written reports -> {out_dir}")
    print("  per_pair_timings.csv, step_timing_aggregated.csv, step_timing_summary_paper.tex")
    print("  fs2d_direct/results.csv, fs2d_so3/results.csv")
    print("Done.")


def _write_latex(out_dir: Path, methods, step_stats):
    """LaTeX tabular: rows = steps, cols = per-method Mean/Std/Median."""
    with open(out_dir / "step_timing_summary_paper.tex", "w") as f:
        ncols = 1 + 3 * len(methods)
        colspec = "l" + "ccc" * len(methods)
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{FS2D per-step computation time (ms) on Boreas "
                f"sequence {SEQUENCE_NUMBER} (N = {N}, matching step "
                f"{MATCHING_STEP}). Mean, standard deviation and median of the "
                "per-step times across pairs.}\n")
        f.write(f"\\label{{tab:fs2d_step_timing_seq{SEQUENCE_NUMBER}}}\n")
        f.write("\\small\n")
        f.write(f"\\begin{{tabular}}{{{colspec}}}\n")
        f.write("\\toprule\n")
        # header: method multicolumns
        header_cells = "{Step}"
        for m in methods:
            header_cells += f" & \\multicolumn{{3}}{{c}}{{{m}}}"
        f.write(header_cells + " \\\\\n")
        # per-method sub-header: Mean Std Median for each method
        sub_cells = " "
        for m in methods:
            sub_cells += " & Mean (ms) & Std (ms) & Median (ms)"
        f.write(sub_cells + " \\\\\n")
        f.write("\\midrule\n")
        for field, label in STEPS:
            cells = f"{label}"
            for m in methods:
                s = step_stats[(m, field)]
                cells += f" & {s['mean']:.1f} & {s['std']:.1f} & {s['median']:.1f}"
            f.write(cells + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


if __name__ == "__main__":
    main()
