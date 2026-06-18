import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
import os


# Set to True to force all methods to use threshold_high
force_high_for_all = False

results_csv = "results_analysis.csv"
output_dir = "pdf_export"
os.makedirs(output_dir, exist_ok=True)

soft_registration_methods = [
    "results32_0_4_12_0.001_0.001_2",
    "results64_0_8_24_0.001_0.001_2",
]

# Noise type configurations: (key, label, filename_suffix)
noise_type_configs = [
    (None, "Both", "both"),
    ("salt_pepper", "Salt & Pepper", "salt_pepper"),
    ("gauss", "Gaussian", "gaussian"),
]


def get_metric(method: str) -> str:
    if force_high_for_all:
        return "threshold_high"
    if method in soft_registration_methods:
        return "threshold_best"
    return "threshold_high"


def parse_filename(filename: str):
    m = filename.replace(".csv", "")

    if m.startswith("outfile_"):
        pat = r"^outfile_(.*?)_(None|low|high)(?:_(salt_pepper|gauss))?_(train|val)$"
        match = re.match(pat, m)
        if match:
            method = match.group(1)
            noise = match.group(2)
            noise_type = match.group(3)  # None, "salt_pepper", or "gauss"
            split = match.group(4)
            return method, noise, noise_type, split
        else:
            return None, None, None, None
    else:
        pat = r"^(results\d+_\d+_\d+_\d+_[\d.]+_[\d.]+_\d+)_(None|low|high)_(train|val)$"
        match = re.match(pat, m)
        if match:
            method = match.group(1)
            noise = match.group(2)
            split = match.group(3)
            return method, noise, None, split
        else:
            return None, None, None, None


noise_order = {"None": 0, "low": 1, "high": 2}
noise_labels = ["None", "Low", "High"]

df = pd.read_csv(results_csv)

data = {}
for _, row in df.iterrows():
    fn = row["FileName"]
    method, noise, noise_type, split = parse_filename(fn)
    if method is None:
        continue
    key = (method, split, noise_type)
    if key not in data:
        data[key] = {}
    m = get_metric(method)
    data[key][noise_order[noise]] = row[m]

methods = sorted(set(k[0] for k in data.keys()))
splits = ["train", "val"]

method_labels = {
    "results32_0_4_12_0.001_0.001_2": "Results-32",
    "results64_0_8_24_0.001_0.001_2": "Results-64",
    "fpfh": "FPFH",
    "geotransformer": "GeoTransformer",
    "hybridpoint": "HybridPoint",
    "icp": "ICP",
    "pointreggpt": "PointRegGPT",
    "predator": "Predator",
    "regtr": "RegTR",
}

colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
]

for noise_key, noise_label, filename_suffix in noise_type_configs:
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    for ax, split in zip(axes, splits):
        for idx, method in enumerate(methods):
            key = (method, split, noise_key)
            if key not in data:
                continue
            vals = data[key]
            x = sorted(vals.keys())
            y = [vals[i] for i in x]

            # Include None (clean) noise level from "both" if missing for this noise type
            if 0 not in vals and (method, split, None) in data:
                vals_none = data[(method, split, None)]
                if 0 in vals_none:
                    combined = dict(vals)
                    combined[0] = vals_none[0]
                    x = sorted(combined.keys())
                    y = [combined[i] for i in x]

            label = method_labels.get(method, method)
            ax.plot(x, y, marker="o", color=colors[idx % len(colors)],
                    linewidth=1.5, markersize=6, label=label)

        if force_high_for_all:
            ylabel = "Threshold High (%)"
        else:
            ylabel = "Threshold (%) — soft reg: best, others: high"
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 105)
        ax.set_yticks(np.arange(0, 106, 10))
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{split.capitalize()} Set — {noise_label} Noise")
        if split == "val":
            ax.set_xticks(list(noise_order.values()))
            ax.set_xticklabels(noise_labels)
            ax.set_xlabel("Noise Level")

        if split == "train":
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, title="Method",
                       loc="upper center", bbox_to_anchor=(0.5, 0.02),
                       ncol=5, fontsize=7)

    if force_high_for_all:
        fig.suptitle(f"Threshold Comparison (threshold_high) — {noise_label} Noise", fontsize=13)
    else:
        fig.suptitle(f"Threshold Comparison (soft reg: best, others: high) — {noise_label} Noise", fontsize=13)
    fig.tight_layout()
    output_pdf = os.path.join(output_dir, f"threshold_comparison_{filename_suffix}.pdf")
    plt.savefig(output_pdf, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output_pdf}")
