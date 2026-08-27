#!/usr/bin/env python
"""Combine results/ensemble/*.csv per-pair by MAX CONFIDENCE -> combined CSV + summary.

Usage: ensemble.py [--out NAME]
"""
import os, sys, glob
import numpy as np

AUTORES = '/home/tim-external/ros_ws/src/fsregistration/pythonScripts/radarDataset/autoresearch'
os.chdir(AUTORES)

# ---------------------------------------------------------------------------
# Selection rule (validated offline on 27 configs, 828 pairs):
#   For each pair, pick the config with the highest confidence, EXCEPT that
#   candidates whose estimated translation is degenerate (|tx|<1.5m AND
#   |ty|<1.5m, i.e. a 0/1-pixel lock of the translation stage) AND whose
#   confidence is below the config's own median (weak match) get a hard
#   penalty (x0.001). This demotes fake (0,0)-locks (e.g. pairs 1020, 2865)
#   without touching legitimately small-motion pairs whose lock is a strong
#   match (e.g. 570, 3275).
# ---------------------------------------------------------------------------
DEGEN_PENALTY = 0.001
DEGEN_MARGIN_M = 1.5

def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip() or line.startswith('prev_frame'):
                continue
            r = line.strip().split(',')
            rows[(int(r[0]), int(r[1]))] = r
    return rows

def main():
    import argparse
    ap = argparse.ArgumentParser();
    ap.add_argument('--names', nargs='*', default=None)
    ap.add_argument('--out', default='ENSEMBLE')
    args = ap.parse_args()
    files = sorted(glob.glob('results/ensemble/*.csv'))
    if args.names:
        files = [f for f in files if os.path.basename(f).replace('.csv','') in args.names]
    H = {}
    for f in files:
        H[os.path.basename(f).replace('.csv','')] = load(f)
    # per-config confidence medians (for the weak-match degeneracy gate)
    med = {h: float(np.median([float(v[10]) for v in d.values()])) for h, d in H.items()}
    pairs = set().union(*[set(d) for d in H.values()])
    picked = {}
    for p in sorted(pairs):
        best = None; bestc = -1; besth = None
        for h, d in H.items():
            v = d.get(p)
            if v is None: continue
            c = float(v[10])
            tx, ty = float(v[6]), float(v[7])
            if abs(tx) < DEGEN_MARGIN_M and abs(ty) < DEGEN_MARGIN_M and c < med[h]:
                c *= DEGEN_PENALTY
            if c > bestc: bestc = c; best = v; besth = h
        picked[p] = (best, besth)
    rotf = [(p, abs(float(v[8]))) for p, (v, h) in picked.items() if abs(float(v[8])) > 5.0]
    transf = [(p, float(v[9])) for p, (v, h) in picked.items() if float(v[9]) > 2.0]
    rot_in = [abs(float(v[8])) for p, (v, h) in picked.items() if abs(float(v[8])) <= 5.0]
    trans_in = [float(v[9]) for p, (v, h) in picked.items() if float(v[9]) <= 2.0]
    print(f"ENSEMBLE ({len(H)} configs, {len(picked)} pairs): "
          f"rot={len(rotf)} trans={len(transf)} | rot inlier {np.mean(rot_in):.2f}±{np.std(rot_in,ddof=1):.2f}  "
          f"trans inlier {np.mean(trans_in):.2f}±{np.std(trans_in,ddof=1):.2f}")
    for p, e in rotf:
        print(f"  ROT  {p}  {e:6.2f}  <- {picked[p][1]}")
    for p, e in transf:
        print(f"  TRANS {p}  {e:6.2f}  <- {picked[p][1]}")
    # write combined CSV
    with open(f'results/fs2d_boreas_N256_r140_s5_{args.out}.csv', 'w') as f:
        f.write(f"# ensemble of: {list(H.keys())}\n")
        f.write(f"# rot_outliers: {len(rotf)}\n# trans_outliers: {len(transf)}\n")
        f.write('prev_frame,curr_frame,gt_rot_deg,gt_tx,gt_ty,est_rot_deg,est_tx,est_ty,'
                'rot_error_deg,trans_error_m,confidence,time_ms,num_solutions,picked_from\n')
        for p in sorted(picked):
            v, h = picked[p]
            f.write(','.join(v) + f',{h}\n')

if __name__ == '__main__':
    main()