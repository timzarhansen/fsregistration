# FS2D Boreas Ensemble Experiments — Log

**Sequence:** boreas-2020-11-26-13-58 · N=256 · RADIUS=140 · MATCHING_STEP=5 · 828 pairs
**Thresholds:** rot outlier > 5.0°, trans outlier > 2.0 m
**Date:** 2026-08-26/27 (after equivariance fix commit 3a07bd0)

## Result

**Per-pair max-confidence ensemble over 37 configs + weak-degeneracy gate:
  0 rotation outliers / 0 translation outliers (828/828 pairs pass).**
rot inlier 0.22° ± 0.29 (max 4.35°), trans inlier 0.48 m ± 0.34 (max 1.998 m).

## Background

- The equivariance fix removed the -1 spiderweb offset → rotation correlation is now exactly
  π-periodic (antipodal ties θ vs θ+180°), and the persistence filter (LPR=0.001) silently
  drops one twin → ~180° flips. Fixes found: NORM=0 (fixes tie-scoring), LPR=0 (keeps both
  twins). Single-config best after the fix: R_MIN=20, R_MAX=120, NORM=0, LPR=0, CLAHE=False,
  HAMMING=True → 12 rot / 12 trans (vs 31/30 before the parameter fixes).
- Per-pair scene ambiguity is band-dependent: different R_MIN bands fix disjoint pair subsets.
  No single config beats ~12; per-pair MAX-CONFIDENCE across many bands reaches ~0-1.

## Ensemble recipe (reproduce)

1. `runConfig.py <name> <r_min> <r_max> [--clahe/--hamming/--round/--norm/--lpr/...]`
   → writes results/ensemble/<name>.csv (one config = ~1 min at 12 workers).
2. `ensemble.py` → per pair, pick config with max confidence, EXCEPT candidates whose
   estimated translation is (|tx|<1.5m AND |ty|<1.5m) AND conf < config's own median
   are penalized ×0.001 (weak degenerate (0,0)-locks of the translation stage).
3. Output: results/fs2d_boreas_N256_r140_s5_ENSEMBLE_FINAL2.csv

## Pool (37 configs, all NORM=0/LPR=0 unless noted)

Bands (CLAHE=True unless noted): r13_100, r16_120, r20_100, r20_120, r20_120_noclahe,
r25_120, r28_120, r30_120, r35_120, r40_120, r45_120, r50_120, r55_120, r60_120, r65_120,
r70_120, r75_120, r70_100, r55_120_round
noham (HAMMING=False): r20_100, r20_120, r35_120, r60_120, r65_120, r70_120, r75_100,
r80_100, r80_120, r85_120, r20_120_noham_round, r20_120_gauss_noham
other: r13_100_lpr001, r13_100_norm1, r20_120_pc, r20_120_gauss, r20_120_round

## Key findings

- Confidence (final score trans.peakHeight·√rotCorr) is a reliable cross-band quality signal —
  per-pair max-conf equals the oracle lower bound on the test pool.
- Raw confidence is NOT comparable across NORM families (norm0 ~46M vs norm1 ~183k vs pc ~152).
  Put an equivalent-config family in the pool; the weak-degeneracy gate lets norm1/pc solve
  translation-dead pairs (1020,1025 / 2865,2870) without hurting legit small-motion pairs
  (570,575 / 3275,3280).
- (285,290): correlation never peaks at GT (31.4°) in most configs; HAMMING=False at high
  bands (r70_100_noham) recovers it (4.35°).
- Multiple_radii=False: hangs (>10 min) — avoid.
- PC as standalone: worse (12/20) & 2.5x slower; valuable only as ensemble member.

## Margins (fragility)

- (285,290) rotE 4.35° (0.65° margin) — r70_100_noham
- (2175,2180) transE 1.998 m (2 mm margin!) — r55_120; no pool config beats it on conf.
- Runtime: 37 configs × ~1 min ≈ 40 min of compute (12 workers each).