"""
Compute missing indices in batch-processed output files.

Scans CSV files in the paperTests/ directory, compares their row indices
against the expected number of rows for each split (train/val), and
writes any missing indices to outputFiles/missing/.

Expected row counts:
    - train: 20642
    - val:   1331

Usage:
    python compute_missing_indices.py
"""

import os
import pandas as pd
from pathlib import Path

PAPER_TESTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paperTests")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputFiles", "missing")

EXPECTED_TRAIN_ROWS = 20642
EXPECTED_VAL_ROWS = 1331


def get_split(filename):
    if "train" in filename:
        return "train"
    elif "val" in filename:
        return "val"
    return None


def compute_missing_indices(filepath, filename):
    split = get_split(filename)
    if split is None:
        return None

    expected_rows = EXPECTED_TRAIN_ROWS if split == "train" else EXPECTED_VAL_ROWS
    expected_indices = set(range(expected_rows))

    try:
        if filename.startswith("outfile"):
            df = pd.read_csv(filepath)
            if "index" in df.columns:
                idx_col = "index"
            else:
                idx_col = df.columns[0]
        elif filename.startswith("results"):
            df = pd.read_csv(filepath, header=None)
            idx_col = df.columns[0]
        else:
            return None
    except Exception as e:
        print(f"  [ERROR] Could not read {filename}: {e}")
        return None

    actual_indices = set(df[idx_col].values.astype(int))
    missing = sorted(expected_indices - actual_indices)
    return missing


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_files = sorted([f for f in os.listdir(PAPER_TESTS_DIR) if f.endswith(".csv")])
    print(f"Scanning {len(csv_files)} CSV files in paperTests/")

    summary = []

    for filename in csv_files:
        filepath = os.path.join(PAPER_TESTS_DIR, filename)
        missing = compute_missing_indices(filepath, filename)

        if missing is None:
            continue

        if len(missing) > 0:
            out_filename = filename.replace(".csv", "_missing.csv")
            out_path = os.path.join(OUTPUT_DIR, out_filename)

            with open(out_path, "w") as f:
                f.write(",".join(str(i) for i in missing))

            summary.append((filename, len(missing), missing))
            print(f"  [MISSING] {filename}: {len(missing)} indices -> {out_filename}")

    print(f"\n{'=' * 50}")
    if summary:
        print(f"Files with missing indices: {len(summary)}")
        for filename, count, indices in summary:
            if count <= 20:
                print(f"  {filename}: missing = {indices}")
            else:
                print(
                    f"  {filename}: missing {count} indices "
                    f"(first 10: {indices[:10]}, last 10: {indices[-10:]})"
                )
    else:
        print("No missing indices found in any file.")
    print(f"\nOutput written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
