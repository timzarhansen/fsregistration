#!/usr/bin/env python3
"""Reference implementation of the notebook's hidden-component scan
(rotation_curve_analysis.ipynb, cells 6-7, GT-free mode) used to validate
the C++ port. Reads the same dumps the C++ run produced and prints the same
quantities as hiddenComponentScan.csv / kernel1D.csv for comparison.

Note: the refinement acceptance uses a relative tolerance (1e-12) that the
notebook does not have; without it the coordinate walk can accept 1-ulp
FP-noise improvements forever. Real improvements are >> 1e-12 relative, so
the results are unaffected.
"""
import os
import numpy as np

DATA_DIR = "/home/tim-external/ros_ws/src/fsregistration/plotting_results/2d/data"

COARSE_RAD = 0.01
FINE_RAD = 0.0004
MIN_SEP_RAD = 0.1
WIN_HALF_RAD = 0.35
MIN_IMPROV_RATIO = 2.0
WEAK_FLOOR_RATIO = 1.2
MAX_HIDDEN = 2
KNOWN_MARGIN_RAD = 0.26
REFINE_TOL = 1e-12  # C++ port addition (see docstring)

def load_lines(fname):
    out = []
    with open(os.path.join(DATA_DIR, fname)) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            vals = []
            for t in ln.replace(",", " ").split():
                try:
                    vals.append(float(t))
                except ValueError:
                    break
            if vals:
                out.append(vals)
    return out

def load_csv(fname):
    rows = load_lines(fname)
    if not rows:
        return None
    counts = [len(r) for r in rows]
    modal = max(set(counts), key=counts.count)
    rows = [r for r in rows if len(r) == modal]
    return np.array(rows) if rows else None

curve = load_csv("rotationCorrelation1D.csv")
# pre-scan persistence peaks (rotationPeaks.csv holds the post-scan updated list)
peaks = load_csv("rotationPeaks_persistence.csv")
c2R = load_csv("patCoefR_1angle.csv")
c2I = load_csv("patCoefI_1angle.csv")

x, c = curve[:, 1], curve[:, 2]
# The C++ pipeline holds the curve as float32 in memory; the CSV dump is its
# decimal representation. Quantize to float32 to reproduce the exact C++ values.
x = x.astype(np.float32).astype(np.float64)
c = c.astype(np.float32).astype(np.float64)
n = len(x) - (len(x) % 2)
x, c = x[:n], c[:n]
half = n // 2
fold_x = x[:half]
fold_c = (c[:half] + c[half:]) / 2.0

def build_kernel_acf(cR, cI, theta):
    bw = int(round(np.sqrt(len(cR))))
    bigL = bw - 1
    Q = np.zeros(2 * bw)
    for l in range(bw):
        for m in range(-l, l + 1):
            if m >= 0:
                idx = m * (bigL + 1) - m * (m - 1) // 2 + (l - m)
            else:
                idx = bigL * (bigL + 3) // 2 + 1 + (bigL + m) * (bigL + m + 1) // 2 + (l - abs(m))
            Q[m + bw] += cR[idx] ** 2 + cI[idx] ** 2
    mpos = np.arange(1, bw)
    K = 2.0 * (Q[mpos + bw] @ np.cos(np.outer(mpos, theta)))
    return K

K = build_kernel_acf(c2R[:, 0], c2I[:, 0], x)
K = (K - K.min()) / (K.max() - K.min())

# ---- compare kernel vs C++ kernel1D.csv ----------------
kCpp = load_csv("kernel1D.csv")
if kCpp is not None:
    d = np.abs(K - kCpp[:, 2])
    print(f"kernel: C++ vs Python  max|d| = {d.max():.3e}")
    USE_CPP_KERNEL = True
    if USE_CPP_KERNEL:
        print("USING the C++ kernel (identical inputs for both implementations)")
        K = kCpp[:, 2].copy()

def circ_dist(a, b, period=np.pi):
    d = np.abs(np.asarray(a) - b) % period
    return np.minimum(d, period - d)

spacing = 2 * np.pi / n
def kval(a):
    w = np.mod(np.asarray(a), 2 * np.pi)
    pos = np.minimum(w, 2 * np.pi - w) / spacing
    i0 = np.floor(pos).astype(int)
    frac = pos - i0
    i1 = (i0 + 1) % n
    return K[i0] * (1.0 - frac) + K[i1] * frac

def design_matrix(angles, th):
    return np.column_stack([kval(th - a) for a in angles] + [np.ones(len(th))])

def fit_nonneg(th, Fw, angles_in):
    kept = list(angles_in)
    while True:
        A = design_matrix(kept, th)
        coef, res, *_ = np.linalg.lstsq(A, Fw, rcond=None)
        r = res[0] if len(res) else float(np.sum((A @ coef - Fw) ** 2))
        amps = coef[:-1]
        if len(amps) == 0 or np.all(amps >= 0):
            return coef, r, kept
        del kept[int(np.argmin(amps))]

def scan_window(ctr, th, F, known_all):
    mask = circ_dist(th, ctr) < WIN_HALF_RAD
    thw, Fw = th[mask], F[mask]
    if len(thw) < 10:
        return None
    win_samp = th[mask]
    m0 = [m for m in known_all if np.min(circ_dist(m, win_samp)) < KNOWN_MARGIN_RAD]
    m0 = list(dict.fromkeys(m0))
    coef_k, r_k, kept_k = fit_nonneg(thw, Fw, m0)
    knowns = list(kept_k)
    lo, hi = ctr - WIN_HALF_RAD, ctr + WIN_HALF_RAD
    seg_lo, seg_hi = max(lo, 0.0), min(hi, np.pi)
    cand = np.arange(seg_lo + COARSE_RAD, seg_hi - COARSE_RAD, COARSE_RAD)
    if hi > np.pi:
        cand = np.concatenate([np.arange(0.0, hi - np.pi - COARSE_RAD, COARSE_RAD), cand])
    elif lo < 0.0:
        cand = np.concatenate([np.arange(np.pi + lo + COARSE_RAD, np.pi, COARSE_RAD), cand])
    if len(cand) == 0:
        return None

    def grid_resids(angles, excl):
        res = []
        for mu in cand:
            if any(circ_dist(mu, m) < MIN_SEP_RAD for m in excl):
                res.append(np.inf)
                continue
            coef, r, kept = fit_nonneg(thw, Fw, angles + [mu])
            res.append(r if kept == angles + [mu] else np.inf)
        return np.array(res)

    def refine_all(hiddens):
        cur = list(hiddens)
        coef, r_cur, kept = fit_nonneg(thw, Fw, knowns + cur)
        while True:
            changed = False
            for idx in range(len(cur)):
                c0 = cur[idx]
                others = knowns + [cur[j] for j in range(len(cur)) if j != idx]
                for a in np.arange(max(c0 - 0.044, cand[0]), min(c0 + 0.044, cand[-1]), FINE_RAD):
                    if any(circ_dist(a, o) < MIN_SEP_RAD for o in others):
                        continue
                    trial = list(cur); trial[idx] = float(a)
                    coef2, r2, kept2 = fit_nonneg(thw, Fw, knowns + trial)
                    if kept2 != knowns + trial:
                        continue
                    if r2 < r_cur - REFINE_TOL * max(1.0, r_cur):  # C++ port tolerance
                        cur[idx] = float(a); r_cur = r2; changed = True
            if not changed:
                break
        coef, r, kept = fit_nonneg(thw, Fw, knowns + cur)
        return cur, r, coef, kept

    hiddens, r_fin, coef, kept = [], r_k, coef_k, knowns
    r1a = None
    mu1 = None
    n_hidden = 0
    res1 = grid_resids(knowns, knowns)
    if not np.isinf(res1).all():
        mu1 = cand[int(np.argmin(res1))]
        hiddens, r_fin, coef, kept = refine_all([mu1])
        r1a = r_fin
        if r_k / r_fin >= MIN_IMPROV_RATIO:
            n_hidden = 1
            if MAX_HIDDEN >= 2:
                res2 = grid_resids(knowns + hiddens, knowns + hiddens)
                if not np.isinf(res2).all():
                    mu2 = cand[int(np.argmin(res2))]
                    hid2, r2, coef2, kept2 = refine_all(hiddens + [mu2])
                    if r_fin / r2 >= MIN_IMPROV_RATIO:
                        hiddens, r_fin, coef, kept = hid2, r2, coef2, kept2
                        n_hidden = 2
    mu = hiddens[0] if n_hidden else (mu1 if mu1 is not None else lo)
    amp = coef[len(knowns)] if len(hiddens) >= 1 else 0.0
    return dict(center=ctr, mu=mu, amp=amp, ratio1=(r_k / r1a) if r1a is not None else None,
                resid0=r_k, resid1=(r1a if r1a is not None else r_k), n_hidden=n_hidden,
                mu2=(hiddens[1] if n_hidden > 1 else None),
                ratio2=((r1a / r2) if n_hidden > 1 and r1a is not None else None))

th, F = fold_x, fold_c
known_all = []
if peaks is not None:
    for p in peaks[:, 0]:
        m = float(p) % np.pi
        if not any(circ_dist(m, u) < 2e-4 for u in known_all):
            known_all.append(m)
    known_all.sort()

print(f"anchors: {len(known_all)}")
print(f"{'anchor':>10} {'mu':>10} {'amp':>9} {'ratio':>8} {'resid0':>9} {'resid1':>9}  strong level")
results = []
for ctr in known_all:
    r = scan_window(ctr, th, F, known_all)
    if r is None:
        print(f"{ctr:>10.5f}  window skipped")
        continue
    results.append(r)
    if r["ratio1"] is None:
        print(f"{ctr:>10.5f}  no valid candidate (skipped)")
        continue
    strong = 1 if (r["ratio1"] is not None and r["ratio1"] >= MIN_IMPROV_RATIO) else 0
    print(f"{ctr:>10.5f} {r['mu']:>10.5f} {r['amp']:>9.6f} {r['ratio1']:>8.5f} {r['resid0']:>9.5f} {r['resid1']:>9.5f}  {strong} 1")
    if r["n_hidden"] > 1:
        print(f"{ctr:>10.5f} {r['mu2']:>10.5f} {'':>9} {r['ratio2']:>8.5f}  ... 1 2")
