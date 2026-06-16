import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re


# Set to True to force all methods to use threshold_high
force_high_for_all = False

results_csv = "results_analysis.csv"
output_pdf = "threshold_comparison.pdf"

soft_registration_methods = [
    "results32_0_4_12_0.001_0.001_2",
    "results64_0_8_24_0.001_0.001_2",
]


def get_metric(method: str) -> str:
    if force_high_for_all:
        return "threshold_high"
    if method in soft_registration_methods:
        return "threshold_best"
    return "threshold_high"

df = pd.read_csv(results_csv)


def parse_filename(filename: str):
    m = filename.replace(".csv", "")

    if m.startswith("outfile_"):
        parts = m.split("_")
        method = parts[1]
        noise = parts[2]
        split = parts[3]
    else:
        pat = r"^(results\d+_\d+_\d+_\d+_[\d.]+_[\d.]+_\d+)_(None|low|high)_(train|val)$"
        match = re.match(pat, m)
        if match:
            method = match.group(1)
            noise = match.group(2)
            split = match.group(3)
        else:
            return None, None, None
    return method, noise, split


noise_order = {"None": 0, "low": 1, "high": 2}
noise_labels = ["None", "Low", "High"]

data = {}
for _, row in df.iterrows():
    fn = row["FileName"]
    method, noise, split = parse_filename(fn)
    if method is None:
        continue
    key = (method, split)
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

fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

for ax, split in zip(axes, splits):
    for idx, method in enumerate(methods):
        key = (method, split)
        if key not in data:
            continue
        vals = data[key]
        x = sorted(vals.keys())
        y = [vals[i] for i in x]
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
    ax.set_title(f"{split.capitalize()} Set")
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
    fig.suptitle("Threshold Comparison (threshold_high)", fontsize=13)
else:
    fig.suptitle("Threshold Comparison (soft reg: best, others: high)", fontsize=13)
fig.tight_layout()
plt.savefig(output_pdf, format="pdf", dpi=300, bbox_inches="tight")
print(f"Plot saved to {output_pdf}")
