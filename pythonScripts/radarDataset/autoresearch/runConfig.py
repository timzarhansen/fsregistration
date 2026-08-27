#!/usr/bin/env python
"""Run ONE FS2D config over the full sequence; save per-config CSV to
results/ensemble/<name>.csv for the max-confidence ensemble.

Usage: runConfig.py <name> <r_min> <r_max> [--norm N] [--lpr X] [--clahe B]
       [--hamming B] [--pc B] [--round B] [--num_angles N] [--potential X]
       [--weighted B] [--gauss B]
"""
import os, sys, json, time, argparse
from multiprocessing import Pool

AUTORES = '/home/tim-external/ros_ws/src/fsregistration/pythonScripts/radarDataset/autoresearch'
sys.path.insert(0, AUTORES)
RADAR_DIR = os.path.dirname(AUTORES)
if RADAR_DIR not in sys.path:
    sys.path.insert(0, RADAR_DIR)
os.chdir(AUTORES)

from boreasDatasetLoader import load_single_sequence
from fullSequencefs2dRun import (load_config, worker_init, process_pair,
                                 compute_summary, RESULT_COLUMNS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name'); ap.add_argument('r_min', type=float); ap.add_argument('r_max', type=float)
    ap.add_argument('--norm', type=int, default=None)
    ap.add_argument('--lpr', type=float, default=None)
    ap.add_argument('--clahe', type=lambda s: s.lower() in ('1','true','t','yes'))
    ap.add_argument('--hamming', type=lambda s: s.lower() in ('1','true','t','yes'))
    ap.add_argument('--pc', type=lambda s: s.lower() in ('1','true','t','yes'))
    ap.add_argument('--round', type=lambda s: s.lower() in ('1','true','t','yes'))
    ap.add_argument('--num_angles', type=int, default=None)
    ap.add_argument('--potential', type=float, default=None)
    ap.add_argument('--weighted', type=lambda s: s.lower() in ('1','true','t','yes'))
    ap.add_argument('--gauss', type=lambda s: s.lower() in ('1','true','t','yes'))
    args = ap.parse_args()

    cfg = load_config()
    mc = {
        'N': cfg.N, 'radius': cfg.RADIUS, 'size_of_pixel': (2.0 * cfg.RADIUS) / cfg.N,
        'use_clahe': cfg.USE_CLAHE, 'use_hamming': cfg.USE_HAMMING,
        'potential_for_necessary_peak': cfg.POTENTIAL_NECCESSARY_FOR_PEAK,
        'multiple_radii': cfg.MULTIPLE_RADII, 'use_gauss': cfg.USE_GAUSS,
        'use_direct': cfg.USE_DIRECT, 'num_angles': cfg.NUM_ANGLES,
        'r_min': args.r_min, 'r_max': args.r_max,
        'level_potential_rotation': cfg.LEVEL_POTENTIAL_ROTATION,
        'normalization': cfg.NORMALIZATION,
        'use_weighted_peak_score': cfg.USE_WEIGHTED_PEAK_SCORE,
        'use_phase_correlation': cfg.USE_PHASE_CORRELATION,
        'debug': False,
    }
    ov = {'norm': ('normalization', args.norm, False),
          'lpr': ('level_potential_rotation', args.lpr, False),
          'clahe': ('use_clahe', args.clahe, False),
          'hamming': ('use_hamming', args.hamming, False),
          'pc': ('use_phase_correlation', args.pc, False),
          'round': ('ROUND', args.round, True),
          'num_angles': ('num_angles', args.num_angles, False),
          'potential': ('potential_for_necessary_peak', args.potential, False),
          'weighted': ('use_weighted_peak_score', args.weighted, False),
          'gauss': ('use_gauss', args.gauss, False)}
    round_images = bool(cfg.ROUND)
    for k, (key, val, isround) in ov.items():
        if val is not None:
            if isround:
                round_images = bool(val)
            else:
                mc[key] = val

    seq = load_single_sequence(cfg.DATA_DIR, cfg.SEQUENCE_NAME)
    total = seq.length
    del seq
    end = total if cfg.MAX_FRAMES is None else min(total, cfg.MAX_FRAMES)
    pairs = [(i - cfg.MATCHING_STEP, i) for i in range(cfg.MATCHING_STEP, end, cfg.MATCHING_STEP)]

    outdir = os.path.join(AUTORES, 'results', 'ensemble')
    os.makedirs(outdir, exist_ok=True)
    outp = os.path.join(outdir, f'{args.name}.csv')
    t0 = time.time()
    with Pool(processes=cfg.NUM_WORKERS, initializer=worker_init,
              initargs=(cfg.DATA_DIR, cfg.SEQUENCE_NAME, mc, round_images)) as pool:
        rows = [r for r in pool.imap_unordered(process_pair, pairs) if r['status'] == 'OK']
    rows.sort(key=lambda r: r['prev_frame'])
    summary = compute_summary(rows, cfg.OUTLIER_ROT_THRESH_DEG, cfg.OUTLIER_TRANS_THRESH_M)
    meta = {k: v for k, v in vars(cfg).items() if k.isupper() and not k.startswith('_')}
    meta.update({f'MC_{k}': v for k, v in mc.items()})
    with open(outp, 'w', newline='') as f:
        f.write(f"# ensemble_run: {args.name}\n")
        f.write(f"# wall_time_s: {time.time()-t0:.1f}\n")
        for key in sorted(meta):
            f.write(f"# config_{key}: {meta[key]}\n")
        for k in ['rot_mean','rot_std','rot_median','trans_mean','trans_std','trans_median',
                  'rot_outliers','trans_outliers']:
            if k in summary:
                f.write(f"# summary_{k}: {summary[k]}\n")
        f.write(','.join(RESULT_COLUMNS) + '\n')
        for r in rows:
            f.write(','.join(str(r[c]) for c in RESULT_COLUMNS) + '\n')
    print(f"[runConfig] {args.name}: rot_out={summary.get('rot_outliers')} "
          f"trans_out={summary.get('trans_outliers')} ({summary.get('rot_std')} deg std) -> {outp}")

if __name__ == '__main__':
    main()