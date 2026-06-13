# %% [markdown]
# # Full-sample extended R-squared bar chart
#
# Creates a grouped bar chart comparing full-sample extended-model R-squared
# values across Ames Housing, King County, and California Housing.

# %%
from pathlib import Path

import os
import tempfile

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "Results"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "thesis_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

SUMMARY_RESULTS_PATH = RESULTS_DIR / "all_results_corrected_standard_errors.csv"

DATASET_LABELS = {
    "Ames Housing": "Ames",
    "King County": "King County",
    "California Housing": "California",
}
DATASET_ORDER = list(DATASET_LABELS)
MODEL_ORDER = ["OLS", "Random Forest", "XGBoost", "TabPFN"]
MODEL_COLORS = {
    "OLS": "#3b6ba8",
    "Random Forest": "#e69138",
    "XGBoost": "#5a9c5d",
    "TabPFN": "#d9534f",
}


# %%
summary = pd.read_csv(SUMMARY_RESULTS_PATH)

extended = summary[
    summary["dataset"].isin(DATASET_ORDER)
    & summary["specification"].eq("Extended")
    & summary["model"].isin(MODEL_ORDER)
].copy()

full_sample_sizes = extended.groupby("dataset")["sample_size"].transform("max")
plot_data = extended[extended["sample_size"].eq(full_sample_sizes)].copy()

plot_data["dataset"] = pd.Categorical(
    plot_data["dataset"], categories=DATASET_ORDER, ordered=True
)
plot_data["model"] = pd.Categorical(
    plot_data["model"], categories=MODEL_ORDER, ordered=True
)
plot_data = plot_data.sort_values(["dataset", "model"])

r2_by_dataset_model = plot_data.pivot(
    index="dataset", columns="model", values="r2_mean"
).loc[DATASET_ORDER, MODEL_ORDER]
r2_se_by_dataset_model = plot_data.pivot(
    index="dataset", columns="model", values="r2_corrected_se"
).loc[DATASET_ORDER, MODEL_ORDER]


# %%
fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)

x = np.arange(len(DATASET_ORDER))
bar_width = 0.2
offsets = (np.arange(len(MODEL_ORDER)) - (len(MODEL_ORDER) - 1) / 2) * bar_width

for offset, model in zip(offsets, MODEL_ORDER):
    ax.bar(
        x + offset,
        r2_by_dataset_model[model],
        yerr=r2_se_by_dataset_model[model],
        width=bar_width,
        color=MODEL_COLORS[model],
        edgecolor=MODEL_COLORS[model],
        error_kw={
            "ecolor": "#3a3a3a",
            "elinewidth": 1.1,
            "capsize": 3,
            "capthick": 1.1,
        },
        label=model,
    )

ax.set_xticks(x)
ax.set_xticklabels([DATASET_LABELS[dataset] for dataset in DATASET_ORDER])
ax.set_ylim(0.55, 0.95)
ax.set_yticks(np.arange(0.55, 0.951, 0.05))
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda value, _: f"{value:.2f}".rstrip("0").rstrip("."))
)
ax.grid(axis="y", color="#8a8a8a", linewidth=1.2)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#7a7a7a")
ax.spines["bottom"].set_color("#7a7a7a")
ax.tick_params(axis="both", colors="#4d4d4d")

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.09),
    ncol=len(MODEL_ORDER),
    frameon=False,
    handlelength=0.8,
    handletextpad=0.3,
    columnspacing=0.9,
)

fig.tight_layout()

png_path = RESULTS_DIR / "full_sample_extended_r2_bar_chart.png"
pdf_path = RESULTS_DIR / "full_sample_extended_r2_bar_chart.pdf"
fig.savefig(png_path, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

print(f"Saved {png_path}")
print(f"Saved {pdf_path}")
print(r2_by_dataset_model.round(3))
print(r2_se_by_dataset_model.round(3))
