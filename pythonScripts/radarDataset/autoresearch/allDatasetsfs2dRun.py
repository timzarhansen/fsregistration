#!/usr/bin/env python3
################################################################################
# allDatasetsfs2dRun.py - FS2D registration over ALL Boreas sequences with GT
#
# Runs the same FS2D (SOFT) 2D registration setup as fullSequencefs2dRun.py
# over every Boreas sequence that has ground-truth poses (applanix/
# radar_poses.csv) in DATA_DIR, using the same config_fs2d.py settings.
# MATCHING_STEP is applied identically to every sequence and ALL frames of
# each sequence are covered: pairs are (0,step), (step,2*step), ...
#
# Per-sequence and combined outlier counts (config thresholds
# OUTLIER_ROT_THRESH_DEG / OUTLIER_TRANS_THRESH_M) are written to
# results/allDatasetsCombined/ as a single CSV and printed to the console.
#
# Usage:
#     python allDatasetsfs2dRun.py
#
# Sequence selection: SEQUENCES below (None = auto-discover all sequences
# that have GT poses). Sequences WITHOUT GT are always excluded: pyboreas
# RadarFrame.pose defaults to identity when no pose file entry exists, so
# running without GT would silently produce meaningless "0 outliers".
#
# The original fullSequencefs2dRun.py and config_fs2d.py are NOT modified;
# shared helpers (config loading, worker, summary) are imported from
# fullSequencefs2dRun.py so both scripts always use the same setup.
################################################################################

import csv
import os
import sys
import time

# --- Limit per-process BLAS/OpenMP threads BEFORE numpy is imported ---------
# (importing fullSequencefs2dRun below triggers numpy; each pool worker uses
#  exactly one core, the NUM_WORKERS processes cover the machine.)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from multiprocessing import Pool

# --- Paths -------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import the single-sequence runner for its config loading, worker functions
# and summary helpers — "same setup" is guaranteed by construction.
# (Its module-level code sets the thread env vars above, adds the parent
# radarDataset dir to sys.path, and imports numpy.)
import fullSequencefs2dRun
from fullSequencefs2dRun import (
    CONFIG_FILENAME,
    RESULT_COLUMNS,
    compute_summary,
    config_hash,
    load_config,
    process_pair,
    summary_numeric_lines,
    summary_table_lines,
    worker_init,
)
from boreasDatasetLoader import load_single_sequence

# ----------------------------------------------------------------------------
# Sequence selection
# ----------------------------------------------------------------------------
# None = auto-discover: every "boreas-*" folder in DATA_DIR that contains
# ground-truth poses (applanix/radar_poses.csv). Missing poses would silently
# degenerate to identity GT (pyboreas RadarFrame.pose default), so sequences
# without GT are ALWAYS excluded.
# Set to an explicit list to run a subset, e.g. ["boreas-2020-11-26-13-58"].
SEQUENCES = None

# Sequences to skip even if they have GT (e.g. ["boreas-2024-12-03-12-54"]).
SEQUENCES_EXCLUDE = []


def has_gt_poses(data_dir: str, sequence_name: str) -> bool:
    """True if the sequence has ground-truth applanix radar poses."""
    return os.path.isfile(
        os.path.join(data_dir, sequence_name, "applanix", "radar_poses.csv")
    )


def discover_sequences(data_dir: str, exclude=()) -> list:
    """All boreas-* sequence folders that have GT poses, sorted."""
    if not os.path.isdir(data_dir):
        sys.exit(f"ERROR: DATA_DIR not found: {data_dir}")
    seqs = sorted(
        d for d in os.listdir(data_dir)
        if d.startswith("boreas-")
        and os.path.isdir(os.path.join(data_dir, d))
        and has_gt_poses(data_dir, d)
        and d not in exclude
    )
    if not seqs:
        sys.exit("ERROR: no sequences with GT poses found in " + data_dir)
    return seqs


# ============================================================================
# Config -> method config
# ============================================================================

def build_method_config(cfg):
    """FS2D method config from config_fs2d.py.

    Mirrors the dict built in fullSequencefs2dRun.main() — keep the two in
    sync manually when config keys are added there.
    """
    size_of_pixel = (2.0 * cfg.RADIUS) / cfg.N
    return {
        "N": cfg.N,
        "radius": cfg.RADIUS,
        "size_of_pixel": size_of_pixel,
        "use_clahe": cfg.USE_CLAHE,
        "use_hamming": cfg.USE_HAMMING,
        "potential_for_necessary_peak": cfg.POTENTIAL_NECCESSARY_FOR_PEAK,
        "multiple_radii": cfg.MULTIPLE_RADII,
        "use_gauss": cfg.USE_GAUSS,
        "use_direct": cfg.USE_DIRECT,
        "num_angles": cfg.NUM_ANGLES,
        "r_min": cfg.R_MIN,
        "r_max": cfg.R_MAX,
        "level_potential_rotation": cfg.LEVEL_POTENTIAL_ROTATION,
        "normalization": cfg.NORMALIZATION,
        "use_weighted_peak_score": cfg.USE_WEIGHTED_PEAK_SCORE,
        "use_phase_correlation": cfg.USE_PHASE_CORRELATION,
        "debug": cfg.DEBUG_MODE,
        # ---- hidden-component rotation scan (requires USE_DIRECT=True) ----
        "use_hidden_component_scan": cfg.USE_HIDDEN_COMPONENT_SCAN,
        "hidden_scan_win_half_rad": cfg.HIDDEN_SCAN_WIN_HALF_RAD,
        "hidden_scan_coarse_rad": cfg.HIDDEN_SCAN_COARSE_RAD,
        "hidden_scan_fine_rad": cfg.HIDDEN_SCAN_FINE_RAD,
        "hidden_scan_min_sep_rad": cfg.HIDDEN_SCAN_MIN_SEP_RAD,
        "hidden_scan_min_improv_ratio": cfg.HIDDEN_SCAN_MIN_IMPROV_RATIO,
        "hidden_scan_weak_floor_ratio": cfg.HIDDEN_SCAN_WEAK_FLOOR_RATIO,
        "hidden_scan_known_margin_rad": cfg.HIDDEN_SCAN_KNOWN_MARGIN_RAD,
        "hidden_scan_max_hidden": cfg.HIDDEN_SCAN_MAX_HIDDEN,
        "hidden_scan_include_weak": cfg.HIDDEN_SCAN_INCLUDE_WEAK,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()
    cfg = load_config()
    method_config = build_method_config(cfg)

    # --- Sequence selection -------------------------------------------------
    num_no_gt = 0
    if SEQUENCES is None:
        sequences = discover_sequences(cfg.DATA_DIR, SEQUENCES_EXCLUDE)
        num_no_gt = sum(
            1 for d in os.listdir(cfg.DATA_DIR)
            if d.startswith("boreas-")
            and os.path.isdir(os.path.join(cfg.DATA_DIR, d))
            and not has_gt_poses(cfg.DATA_DIR, d)
        )
    else:
        sequences = []
        for s in SEQUENCES:
            if s in SEQUENCES_EXCLUDE:
                continue
            if not has_gt_poses(cfg.DATA_DIR, s):
                print(f"WARNING: skipping {s}: no applanix/radar_poses.csv (no GT)")
                continue
            sequences.append(s)
        if not sequences:
            sys.exit("ERROR: no valid sequences selected")

    # --- Output file --------------------------------------------------------
    # Short readable name + config hash + sequence count (sequence list is in
    # the CSV header). Changing the config or the selected sequences produces
    # a new file; old results are kept.
    results_dir = os.path.join(SCRIPT_DIR, "results", "allDatasetsCombined")
    os.makedirs(results_dir, exist_ok=True)
    out_name = (f"fs2d_allDatasets_N{cfg.N}_r{cfg.RADIUS:g}_"
                f"s{cfg.MATCHING_STEP}_{config_hash(cfg)}_{len(sequences)}seqs.csv")
    out_path = os.path.join(results_dir, out_name)

    print("=" * 80)
    print("FS2D all-datasets Boreas registration (sequences with GT)")
    print("=" * 80)
    print(f"Sequences : {len(sequences)} with GT"
          + (f" ({num_no_gt} without GT excluded)" if num_no_gt else ""))
    print(f"Matching  : every {cfg.MATCHING_STEP}th frame from 0, ALL frames of each sequence"
          + (f" (MAX_FRAMES={cfg.MAX_FRAMES} per sequence)" if cfg.MAX_FRAMES else ""))
    print(f"Grid      : N={cfg.N}, radius={cfg.RADIUS} m, pixel_size={2.0 * cfg.RADIUS / cfg.N:.3f} m")
    print(f"Workers   : {cfg.NUM_WORKERS}")
    print(f"Output    : {out_path}")
    print()

    # --- Run every sequence -------------------------------------------------
    all_rows = []
    all_failures = []
    seq_summaries = {}   # seq_name -> (summary, pairs_ok, pairs_failed, frames)
    skipped = []
    total_pairs = 0

    for seq_idx, seq_name in enumerate(sequences, 1):
        print("=" * 80)
        print(f"[{seq_idx}/{len(sequences)}] {seq_name}")
        print("=" * 80)
        try:
            seq = load_single_sequence(cfg.DATA_DIR, seq_name)
            total_frames = seq.length
            del seq  # workers load their own copies
        except Exception as e:
            print(f"WARNING: sequence load failed, skipped: {type(e).__name__}: {e}")
            skipped.append(seq_name)
            continue

        end = total_frames if cfg.MAX_FRAMES is None else min(total_frames, cfg.MAX_FRAMES)
        if end < cfg.MATCHING_STEP:
            print(f"WARNING: {seq_name}: only {end} frames, no pairs possible — skipped")
            skipped.append(seq_name)
            continue
        pairs = [(i - cfg.MATCHING_STEP, i) for i in range(cfg.MATCHING_STEP, end, cfg.MATCHING_STEP)]
        total_pairs += len(pairs)
        print(f"Frames    : {total_frames}, pairs: {len(pairs)}")

        t0 = time.time()
        rows_ok, failures = [], []
        done = 0
        with Pool(processes=cfg.NUM_WORKERS, initializer=worker_init,
                  initargs=(cfg.DATA_DIR, seq_name, method_config, cfg.ROUND)) as pool:
            for row in pool.imap_unordered(process_pair, pairs):
                done += 1
                row["sequence"] = seq_name
                if row["status"] == "FAIL":
                    failures.append(row)
                    print(f"[{done:4d}/{len(pairs)}] pair {row['prev_frame']:5d}->{row['curr_frame']:5d} "
                          f"FAILED: {row['error']}")
                else:
                    rows_ok.append(row)
                    print(f"[{done:4d}/{len(pairs)}] pair {row['prev_frame']:5d}->{row['curr_frame']:5d} "
                          f"rot_err={row['rot_error_deg']:7.3f} deg "
                          f"trans_err={row['trans_error_m']:8.3f} m "
                          f"conf={row['confidence']:.3f} time={row['time_ms']:.0f} ms")
        seq_elapsed = time.time() - t0

        summary = compute_summary(rows_ok, cfg.OUTLIER_ROT_THRESH_DEG, cfg.OUTLIER_TRANS_THRESH_M)
        seq_summaries[seq_name] = (summary, len(rows_ok), len(failures), total_frames)
        all_rows.extend(rows_ok)
        all_failures.extend(failures)
        print(f"  -> {seq_name}: {len(rows_ok)}/{len(pairs)} pairs OK, "
              f"rot_out={summary['rot_outlier_count']}, trans_out={summary['trans_outlier_count']} "
              f"({seq_elapsed:.0f}s)")
        print()

    run_elapsed = time.time() - t_start
    all_rows.sort(key=lambda r: (r["sequence"], r["prev_frame"]))
    combined = compute_summary(all_rows, cfg.OUTLIER_ROT_THRESH_DEG, cfg.OUTLIER_TRANS_THRESH_M)

    # --- Write combined CSV --------------------------------------------------
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        # --- Combined summary at the top ---
        if all_rows:
            header, data = summary_table_lines(combined)
            writer.writerow(["# " + "-" * 78])
            writer.writerow(["# " + header])
            writer.writerow(["# " + data])
            writer.writerow(["# " + "-" * 78])
            for key, val in summary_numeric_lines(combined):
                writer.writerow([f"# {key}: {val:.6f}" if isinstance(val, float) else f"# {key}: {val}"])
        # --- Per-sequence summary ---
        writer.writerow([])
        writer.writerow(["# Per-sequence: frames | pairs | ok | failed | rot_outliers | trans_outliers | inliers | rot_mean_deg | trans_mean_m"])
        for seq_name in sequences:
            if seq_name not in seq_summaries:
                continue
            s, ok, failed, frames = seq_summaries[seq_name]
            writer.writerow([f"# seq {seq_name}: frames={frames} pairs={s['num_pairs']} ok={ok} "
                             f"failed={failed} rot_out={s['rot_outlier_count']} "
                             f"trans_out={s['trans_outlier_count']} inliers={s['num_inliers']} "
                             f"rot_mean_deg={s['rot_mean_deg']:.4f} trans_mean_m={s['trans_mean_m']:.4f}"])
        if skipped:
            writer.writerow([f"# skipped (load failed / no pairs): {', '.join(skipped)}"])
        # --- Run metadata ---
        writer.writerow([])
        writer.writerow([f"# num_sequences: {len(sequences)}"])
        writer.writerow([f"# sequence_names: {','.join(sequences)}"])
        writer.writerow([f"# num_sequences_without_gt_excluded: {num_no_gt}"])
        writer.writerow([f"# num_sequences_skipped: {len(skipped)}"])
        writer.writerow([f"# num_pairs_total: {total_pairs}"])
        writer.writerow([f"# num_pairs_ok: {len(all_rows)}"])
        writer.writerow([f"# num_pairs_failed: {len(all_failures)}"])
        writer.writerow([f"# run_start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t_start))}"])
        writer.writerow([f"# wall_time_s: {run_elapsed:.1f}"])
        writer.writerow([f"# config_file: {CONFIG_FILENAME}"])
        for key in sorted(k for k in vars(cfg) if k.isupper() and not k.startswith("_")):
            writer.writerow([f"# config_{key}: {vars(cfg)[key]}"])
        writer.writerow([])
        writer.writerow(["sequence"] + RESULT_COLUMNS)
        for row in all_rows:
            writer.writerow([row["sequence"]] + [row[c] for c in RESULT_COLUMNS])

    # Failure log (only if any pairs failed).
    if all_failures:
        fail_path = os.path.join(results_dir, out_name.replace(".csv", "_failures.log"))
        with open(fail_path, "w") as f:
            for r in all_failures:
                f.write(f"{r['sequence']} pair {r['prev_frame']}->{r['curr_frame']}: {r['error']}\n")
        print(f"\nFailures logged to: {fail_path}")

    # --- Console summary ------------------------------------------------------
    print()
    print("-" * 80)
    print(f"Done in {run_elapsed:.0f}s across {len(sequences)} sequences: "
          f"{total_pairs} pairs, {len(all_rows)} OK, {len(all_failures)} failed")
    print()
    print(f"{'sequence':<26}{'pairs':>7}{'ok':>7}{'rot_out':>9}{'trans_out':>10}{'inliers':>9}")
    for seq_name in sequences:
        if seq_name not in seq_summaries:
            continue
        s, ok, failed, frames = seq_summaries[seq_name]
        print(f"{seq_name:<26}{s['num_pairs']:>7}{ok:>7}"
              f"{s['rot_outlier_count']:>9}{s['trans_outlier_count']:>10}{s['num_inliers']:>9}")
    if all_rows:
        print(f"{'COMBINED':<26}{combined['num_pairs']:>7}{len(all_rows):>7}"
              f"{combined['rot_outlier_count']:>9}{combined['trans_outlier_count']:>10}"
              f"{combined['num_inliers']:>9}")
        header, data = summary_table_lines(combined)
        print()
        print(header)
        print(data)
    print(f"CSV: {out_path}")


if __name__ == "__main__":
    main()