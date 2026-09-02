#!/usr/bin/env python3
"""
create_latex_table.py
=====================

Generates a LaTeX `table*` of registration results in the style of the paper
table, with success-rate (threshold) columns for **both** the best and the
highest match:

    Method | Noise level | Dataset | Mean Translation(m) | Mean Rotation(deg) |
    Median Translation(m) | Median Rotation(deg) | Threshold Best(%) | Threshold High(%)

The evaluation is identical to `read_in_data_files.py`
(`evaluate_file_3d_predator_matching`), copied here so that this script is
self-contained and does not trigger that module's module-level analysis loop.
Success threshold defaults to 10 deg rotation error and 0.4 m translation error.

Usage:
    python create_latex_table.py [folder] [--threshold-trans 0.4] [--threshold-rot 10]
        [--include-noise-types] [--out results_table.tex] [--label tab:results]

Examples:
    # current paperTests data (no predator rows there):
    python create_latex_table.py paperTests

    # reproduce the paper table (incl. predator / FS3D 32 | 64 rows):
    python create_latex_table.py BackupToBeSave

Notes:
    * Methods with a single estimate per scan (predator, regtr, fpfh, icp,
      ...) have no separate best match, so the Threshold Best column shows --.
    * For FS3D (soft) the statistics columns use the best match (like before);
      the two threshold columns are reported separately.
    * gauss / salt&pepper noise-type rows are included by default; use
      --skip-noise-types to only emit the plain None/high/low rows.
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Evaluation (same math as read_in_data_files.py)
# ---------------------------------------------------------------------------

def transformations_matrix(roll, pitch, yaw, x, y, z):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    rotation = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ])

    return np.array([
        [rotation[0, 0], rotation[0, 1], rotation[0, 2], x],
        [rotation[1, 0], rotation[1, 1], rotation[1, 2], y],
        [rotation[2, 0], rotation[2, 1], rotation[2, 2], z],
        [0, 0, 0, 1]
    ])


def rotation_angle_difference(r1, r2):
    r = r1.T @ r2
    cos_theta = np.clip((np.trace(r) - 1) / 2, -1, 1)
    return np.arccos(cos_theta) * 180 / np.pi


def safe_correlation(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def evaluate_file_3d_predator_matching(folder, dataset_name, threshold_trans, threshold_rot):
    """Same computation as read_in_data_files.py (silent), returns fullresults.

    Columns of resulting_data:
        0 percentage, 1 number_of_solutions, 2 trans_high, 3 rot_high,
        4 trans_best, 5 rot_best, 6 threshold_best, 7 threshold_high

    fullresults layout (identical indexing to read_in_data_files.py):
        0..3  mean trans/rot high/best, 4..7 std high/best,
        8..11 median high/best, 12/13 correlation best/high,
        14/15 percentage (threshold) best/high
    """
    data = pd.read_csv(os.path.join(folder, dataset_name)).values
    data_set_length = len(data)
    resulting_data = np.zeros((data_set_length, 8))

    for i in range(data_set_length):
        if not any(x in dataset_name for x in ('_soft_', 'results')):
            # single-estimate methods: GT at cols 2..7, estimate at cols 8..13
            gt = transformations_matrix(data[i, 2], data[i, 3], data[i, 4],
                                        data[i, 5], data[i, 6], data[i, 7])
            est = transformations_matrix(data[i, 8], data[i, 9], data[i, 10],
                                         data[i, 11], data[i, 12], data[i, 13])
            trans = np.linalg.norm(gt[0:3, 3] - est[0:3, 3])
            rot = rotation_angle_difference(gt[0:3, 0:3], est[0:3, 0:3])
            within = (trans <= threshold_trans) and (rot <= threshold_rot)

            resulting_data[i, 0] = data[i, 1]          # overlap%
            resulting_data[i, 2] = trans
            resulting_data[i, 3] = rot
            resulting_data[i, 7] = float(within)       # threshold_high
        else:
            # soft methods: GT cols 3..8, highest 9..14, best 15..20
            gt = transformations_matrix(data[i, 3], data[i, 4], data[i, 5],
                                        data[i, 6], data[i, 7], data[i, 8])
            highest = transformations_matrix(data[i, 9], data[i, 10], data[i, 11],
                                             data[i, 12], data[i, 13], data[i, 14])
            best = transformations_matrix(data[i, 15], data[i, 16], data[i, 17],
                                          data[i, 18], data[i, 19], data[i, 20])
            gt_trans = gt[0:3, 3]
            gt_rot = gt[0:3, 0:3]

            trans_high = np.linalg.norm(gt_trans - highest[0:3, 3])
            trans_best = np.linalg.norm(gt_trans - best[0:3, 3])
            rot_high = rotation_angle_difference(gt_rot, highest[0:3, 0:3])
            rot_best = rotation_angle_difference(gt_rot, best[0:3, 0:3])

            threshold_high = float((trans_high <= threshold_trans) and (rot_high <= threshold_rot))
            threshold_best = float((trans_best <= threshold_trans) and (rot_best <= threshold_rot))

            resulting_data[i, 0] = data[i, 2]          # overlap%
            resulting_data[i, 1] = data[i, 1]          # numSolutions
            resulting_data[i, 2] = trans_high
            resulting_data[i, 3] = rot_high
            resulting_data[i, 4] = trans_best
            resulting_data[i, 5] = rot_best
            resulting_data[i, 6] = threshold_best
            resulting_data[i, 7] = threshold_high

    fullresults = np.array([
        np.mean(resulting_data[:, 2]), np.mean(resulting_data[:, 3]),
        np.mean(resulting_data[:, 4]), np.mean(resulting_data[:, 5]),
        np.std(resulting_data[:, 2]), np.std(resulting_data[:, 3]),
        np.std(resulting_data[:, 4]), np.std(resulting_data[:, 5]),
        np.median(resulting_data[:, 2]), np.median(resulting_data[:, 3]),
        np.median(resulting_data[:, 4]), np.median(resulting_data[:, 5]),
        safe_correlation(resulting_data[:, 6], resulting_data[:, 0]),
        safe_correlation(resulting_data[:, 7], resulting_data[:, 0]),
        np.sum(resulting_data[:, 6]) / len(resulting_data[:, 6]) * 100,
        np.sum(resulting_data[:, 7]) / len(resulting_data[:, 7]) * 100,
    ])
    return fullresults


# ---------------------------------------------------------------------------
# Filename parsing + display names
# ---------------------------------------------------------------------------

def parse_filename(fn):
    """Returns (method, noise, noise_type, split) or None.

    Handles:
        outfile_{method}_{noise}(_{gauss|salt_pepper})?_{train|val}.csv
        results{...}_{noise}(_{gauss|salt_pepper})?_{train|val}.csv
    where noise is None|low|high.
    """
    m = re.match(
        r'^outfile_(?P<method>.+?)_(?P<noise>None|low|high)'
        r'(?:_(?P<ntype>gauss|salt_pepper))?_(?P<split>train|val)\.csv$', fn)
    if m:
        return (m.group('method'), m.group('noise'), m.group('ntype'), m.group('split'))
    m = re.match(
        r'^results(?P<rest>.+?)_(?P<noise>None|low|high)'
        r'(?:_(?P<ntype>gauss|salt_pepper))?_(?P<split>train|val)\.csv$', fn)
    if m:
        return ('results' + m.group('rest'), m.group('noise'), m.group('ntype'), m.group('split'))
    return None


def method_label(method):
    """Map raw method names to display names used in the paper table."""
    if method.startswith('results32') or method.endswith('N32'):
        return 'FS3D 32'
    if method.startswith('results64') or method.endswith('N64') or method == 'soft_64':
        return 'FS3D 64'
    return method


METHOD_ORDER = ['predator', 'regtr', 'geotransformer', 'pointreggpt',
                'hybridpoint', 'fpfh', 'icp', 'FS3D 32', 'FS3D 64']
NOISE_ORDER = ['None', 'high', 'low']
NTYPE_ORDER = [None, 'gauss', 'salt_pepper']
SPLIT_LABELS = {'train': 'train', 'val': 'val'}


def row_sort_key(row):
    return (
        METHOD_ORDER.index(row['label']) if row['label'] in METHOD_ORDER else len(METHOD_ORDER) + 1,
        NOISE_ORDER.index(row['noise']),
        NTYPE_ORDER.index(row['ntype']),
        row['split'],
    )


# ---------------------------------------------------------------------------
# LaTeX generation
# ---------------------------------------------------------------------------

def noise_display(noise, ntype):
    label = noise
    if ntype == 'gauss':
        label += ' gauss'
    elif ntype == 'salt_pepper':
        label += ' s\\&p'
    return label


def build_table(rows, threshold_trans, threshold_rot, label, caption, scale=0.8):
    lines = []
    lines.append('\\begin{table*}[!ht]')
    lines.append('    \\caption{' + caption + '}')
    lines.append('    \\centering')
    lines.append('    \\fbox{\\scalebox{%.1f}{%%' % scale)
    lines.append('    \\begin{tabular}{|c|c|c||c|c|c|c|c|c|}')
    lines.append('       \\hline')
    lines.append('        Method & Noise & Dataset & Mean Trans. (m) & Mean Rot. (deg) & ' +
                 'Median Trans. (m) & Median Rot. (deg) & Thresh. Best(\\%) & Thresh. High(\\%) \\\\')
    lines.append('        \\hline')
    lines.append('        \\hline')
    for row in rows:
        lines.append('        ' + row['tex'] + ' \\\\')
    lines.append('        \\hline')
    lines.append('    \\end{tabular}}}')
    lines.append('\\label{' + label + '}')
    lines.append('\\end{table*}')
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(
        description='Generate a LaTeX table of registration results (paper table style, '
                    'thresholds for best AND highest match).')
    ap.add_argument('folder', nargs='?', default='paperTests',
                    help='folder with outfile_*.csv / results*_*.csv '
                         '(default: paperTests; use BackupToBeSave for the paper table incl. predator)')
    ap.add_argument('--threshold-trans', type=float, default=0.4, help='translation threshold in meter')
    ap.add_argument('--threshold-rot', type=float, default=10.0, help='rotation threshold in degree')
    ap.add_argument('--skip-noise-types', action='store_true',
                    help='only emit plain noise rows (None/high/low), skip gauss / salt & pepper')
    ap.add_argument('--scale', type=float, default=0.8,
                    help='scale factor for the \\fbox{\\scalebox{...}} wrapper (default: 0.8 = 80%%)')
    ap.add_argument('--out', default='results_table.tex', help='LaTeX output file')
    ap.add_argument('--label', default='tab:results', help='LaTeX label of the table')
    ap.add_argument('--caption', default=None,
                    help='table caption (default: paper-style caption mentioning the thresholds)')
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.folder) if f.endswith('.csv'))
    if not files:
        raise FileNotFoundError('No CSV files found in folder: %s' % args.folder)

    # key -> (source_file, fullresults); results* files win over outfile_soft*
    # when both exist (e.g. BackupToBeSave has results32_... and outfile_soft_N32_...)
    computed = {}
    n_skipped_types = 0
    for fn in files:
        parsed = parse_filename(fn)
        if parsed is None:
            continue
        method, noise, ntype, split = parsed
        if ntype is not None and args.skip_noise_types:
            n_skipped_types += 1
            continue
        label = method_label(method)
        key = (label, noise, ntype, split)
        r = evaluate_file_3d_predator_matching(args.folder, fn,
                                              args.threshold_trans, args.threshold_rot)
        if key not in computed or computed[key][0].startswith('outfile_soft'):
            computed[key] = (fn, r)

    rows = []
    for (label, noise, ntype, split), (fn, r) in computed.items():
        is_soft = ('_soft_' in fn) or fn.startswith('results')
        if is_soft:
            # statistics from the best match (paper caption convention)
            mT, mR, sT, sR = r[2], r[3], r[6], r[7]
            medT, medR = r[10], r[11]
            thr_best, thr_high = r[14], r[15]
        else:
            # single estimate: no separate best match
            mT, mR, sT, sR = r[0], r[1], r[4], r[5]
            medT, medR = r[8], r[9]
            thr_best = None
            thr_high = r[15]

        thr_best_str = '--' if thr_best is None else '%.2f' % thr_best
        rows.append({
            'label': label,
            'noise': noise,
            'ntype': ntype,
            'split': split,
            'tex': '%s & %s & %s & %.2f$\\pm$%.2f & %.2f$\\pm$%.2f & %.2f & %.2f & %s & %.2f' % (
                label, noise_display(noise, ntype), SPLIT_LABELS[split],
                mT, sT, mR, sR, medT, medR, thr_best_str, thr_high),
        })
    rows.sort(key=row_sort_key)

    if not rows:
        raise ValueError('No result files matched the expected naming in %s' % args.folder)

    if args.caption is None:
        caption = ('These are results of the registration of Predator and FS3D. '
                   '\\textcolor{red}{Threshold is %.1f degree and %.1f meter translation error. '
                   'For FS3D the best results are used for the statistics; the success rate is '
                   'reported for both the best and the highest match. Methods with only a single '
                   'estimate have no separate best match (marked --).}' % (args.threshold_rot, args.threshold_trans))
    else:
        caption = args.caption

    tex = build_table(rows, args.threshold_trans, args.threshold_rot, args.label, caption, scale=args.scale)
    print(tex)
    with open(args.out, 'w') as f:
        f.write(tex)
    print('Wrote %d rows -> %s (folder=%s, skipped %d noise-type variants)' %
          (len(rows), args.out, args.folder, n_skipped_types), file=sys.stderr)


if __name__ == '__main__':
    main()