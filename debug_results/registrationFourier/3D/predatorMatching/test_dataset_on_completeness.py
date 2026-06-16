import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PAPER_TESTS_DIR = os.path.join(os.path.dirname(__file__), "paperTests")

EXPECTED_TRAIN_ROWS = 20642
EXPECTED_VAL_ROWS = 1331

EXPECTED_OUTFILE_HEADER = [
    "index", "overlap%", "GT_roll", "GT_pitch", "GT_yaw",
    "GT_x", "GT_y", "GT_z",
    "Est_roll", "Est_pitch", "Est_yaw", "Est_x", "Est_y", "Est_z"
]


def is_outfile(filename):
    return filename.startswith("outfile")


def is_results_file(filename):
    return filename.startswith("results")


def get_split(filename):
    if "train" in filename:
        return "train"
    elif "val" in filename:
        return "val"
    return None


def check_row_count(df, filename, split):
    expected = EXPECTED_TRAIN_ROWS if split == "train" else EXPECTED_VAL_ROWS
    actual = len(df)
    passed = actual == expected
    if not passed:
        print(f"  [FAIL] Row count: expected {expected}, got {actual}")
    return passed, f"rows={actual}, expected={expected}"


def check_header(df, filename):
    passed = True
    issues = []
    actual_cols = list(df.columns)
    if actual_cols != EXPECTED_OUTFILE_HEADER:
        missing = set(EXPECTED_OUTFILE_HEADER) - set(actual_cols)
        extra = set(actual_cols) - set(EXPECTED_OUTFILE_HEADER)
        if missing:
            issues.append(f"Missing columns: {missing}")
        if extra:
            issues.append(f"Extra columns: {extra}")
        passed = False
        for issue in issues:
            print(f"  [FAIL] Header - {issue}")
    return passed, "; ".join(issues) if issues else "OK"


def check_nan_inf(df, filename):
    passed = True
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    issues = []

    for col in numeric_cols:
        nan_count = df[col].isna().sum()
        inf_mask = np.isinf(df[col].dropna().values) if df[col].dropna().shape[0] > 0 else np.array([])
        inf_count = int(inf_mask.sum())

        if nan_count > 0:
            issues.append(f"  '{col}': {nan_count} NaN")
            passed = False
        if inf_count > 0:
            issues.append(f"  '{col}': {inf_count} Inf")
            passed = False

    return passed, "; ".join(issues) if issues else "No NaN/Inf found"


def get_index_col(df, filename):
    if "index" in df.columns:
        return "index"
    first_col = df.columns[0]
    if df[first_col].dtype in [np.int64, np.int32, np.float64]:
        return first_col
    return None


def check_index_completeness(df, filename):
    passed = True
    issues = []
    idx_col = get_index_col(df, filename)
    if idx_col is None:
        issues.append("No usable index column found")
        return False, "; ".join(issues)

    indices = df[idx_col].values
    expected_indices = np.arange(len(df))

    if not np.array_equal(indices, expected_indices):
        missing = set(expected_indices) - set(indices.astype(int))
        extra = set(indices.astype(int)) - set(expected_indices)
        if missing:
            issues.append(f"Missing indices ({len(missing)}): {sorted(list(missing))[:20]}{'...' if len(missing) > 20 else ''}")
        if extra:
            issues.append(f"Extra indices ({len(extra)}): {sorted(list(extra))[:20]}{'...' if len(extra) > 20 else ''}")
        passed = False

    return passed, "; ".join(issues) if issues else "Contiguous 0..N-1"


def check_duplicate_indices(df, filename):
    passed = True
    idx_col = get_index_col(df, filename)
    if idx_col is None:
        return True, "No usable index column"

    dups = df[idx_col].value_counts()
    dups = dups[dups > 1]
    if len(dups) > 0:
        passed = False
        items = dups.head(10).items()
        detail = ", ".join(f"idx={idx} x{cnt}" for idx, cnt in items)
        if len(dups) > 10:
            detail += f" ... ({len(dups)} total)"
        print(f"  [FAIL] Duplicate indices: {detail}")

    return passed, f"{len(dups)} duplicate index values" if not passed else "No duplicates"


def check_overlap_range(df, filename):
    passed = True
    if "overlap%" not in df.columns:
        return True, "No 'overlap%' column"

    col = df["overlap%"]
    below = (col < 0).sum()
    above = (col > 1).sum()
    issues = []
    if below > 0:
        issues.append(f"{below} values < 0")
        passed = False
    if above > 0:
        issues.append(f"{above} values > 1")
        passed = False

    return passed, "; ".join(issues) if issues else "All in [0, 1]"


def check_empty_rows(df, filename):
    passed = True
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return True, "No numeric columns"

    all_zero = (df[numeric_cols] == 0).all(axis=1)
    all_nan = df[numeric_cols].isna().all(axis=1)
    empty_mask = all_zero | all_nan
    empty_count = empty_mask.sum()

    if empty_count > 0:
        passed = False
        idx_col = get_index_col(df, filename)
        empty_indices = df.loc[empty_mask, idx_col].values.tolist() if idx_col else []
        print(f"  [FAIL] Empty rows ({empty_count}): indices {empty_indices[:20]}{'...' if empty_count > 20 else ''}")

    return passed, f"{empty_count} empty rows" if not passed else "No empty rows"


def check_results_file_specific(df, filename):
    passed = True
    issues = []

    expected_cols = 21
    actual_cols = df.shape[1]
    if actual_cols != expected_cols:
        issues.append(f"Expected {expected_cols} columns, got {actual_cols}")
        passed = False

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        overlap_col = numeric_cols[2] if len(numeric_cols) > 2 else None
        if overlap_col:
            below = (df[overlap_col] < 0).sum()
            above = (df[overlap_col] > 1).sum()
            if below > 0 or above > 0:
                issues.append(f"Overlap col (idx 2): {below} < 0, {above} > 1")
                passed = False

    first_col = df.columns[0]
    expected_indices = np.arange(len(df))
    if not np.array_equal(df[first_col].values, expected_indices):
        missing = set(expected_indices) - set(df[first_col].values.astype(int))
        if missing:
            issues.append(f"Missing indices ({len(missing)}): {sorted(list(missing))[:20]}{'...' if len(missing) > 20 else ''}")
            passed = False

    dups = df[first_col].value_counts()
    dups = dups[dups > 1]
    if len(dups) > 0:
        issues.append(f"{len(dups)} duplicate index values")
        passed = False

    return passed, "; ".join(issues) if issues else "OK"


def test_file(filepath, filename):
    split = get_split(filename)
    if split is None:
        print(f"\n{'=' * 60}")
        print(f"SKIP {filename} - unable to determine split (train/val)")
        print(f"{'=' * 60}")
        return

    expected_rows = EXPECTED_TRAIN_ROWS if split == "train" else EXPECTED_VAL_ROWS
    print(f"\n{'=' * 60}")
    print(f"Testing: {filename}  (split={split}, expected rows={expected_rows})")
    print(f"{'=' * 60}")

    try:
        if is_outfile(filename):
            df = pd.read_csv(filepath)
        elif is_results_file(filename):
            df = pd.read_csv(filepath, header=None)
        else:
            df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  [FAIL] Could not read file: {e}")
        return

    results = {}

    results["Row count"] = check_row_count(df, filename, split)

    if is_outfile(filename):
        results["Header"] = check_header(df, filename)
        results["Overlap range"] = check_overlap_range(df, filename)
        results["Index completeness"] = check_index_completeness(df, filename)
        results["Duplicate indices"] = check_duplicate_indices(df, filename)
        results["Empty rows"] = check_empty_rows(df, filename)
    elif is_results_file(filename):
        results["File-specific checks"] = check_results_file_specific(df, filename)
        results["Index completeness"] = check_index_completeness(df, filename)
        results["Duplicate indices"] = check_duplicate_indices(df, filename)
        results["Empty rows"] = check_empty_rows(df, filename)

    results["NaN / Inf"] = check_nan_inf(df, filename)

    file_passed = all(r[0] for r in results.values())
    print(f"\n  Result: {'[PASS]' if file_passed else '[FAIL]'}")
    for check_name, (passed, detail) in results.items():
        status = "OK" if passed else "FAIL"
        print(f"    [{status}] {check_name}: {detail}")

    return file_passed, results


def main():
    if not os.path.isdir(PAPER_TESTS_DIR):
        print(f"Error: paperTests directory not found at {PAPER_TESTS_DIR}")
        sys.exit(1)

    csv_files = sorted([f for f in os.listdir(PAPER_TESTS_DIR) if f.endswith(".csv")])
    if not csv_files:
        print("No CSV files found in paperTests directory.")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files in paperTests/")
    print(f"Expected: train={EXPECTED_TRAIN_ROWS} rows, val={EXPECTED_VAL_ROWS} rows")

    all_passed = True
    summary = []

    for filename in csv_files:
        filepath = os.path.join(PAPER_TESTS_DIR, filename)
        try:
            passed, results = test_file(filepath, filename)
        except Exception as e:
            print(f"\n  [ERROR] Unexpected error testing {filename}: {e}")
            passed = False
            results = {}
        all_passed = all_passed and passed
        summary.append((filename, passed, results))

    print(f"\n\n{'#' * 60}")
    print("SUMMARY")
    print(f"{'#' * 60}")

    passed_count = sum(1 for _, p, _ in summary if p)
    failed_count = sum(1 for _, p, _ in summary if not p)

    for filename, passed, results in summary:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {filename}")
        if not passed:
            for check_name, (cp, detail) in results.items():
                if not cp:
                    print(f"         - {check_name}: {detail}")

    print(f"\nTotal: {passed_count} passed, {failed_count} failed, {len(csv_files)} files")

    if all_passed:
        print("\nAll files passed all checks.")
    else:
        print("\nSome files failed checks. See details above.")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
