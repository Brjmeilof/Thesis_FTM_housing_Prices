# %% [markdown]
# # R-squared by sample size
#
# Creates one baseline figure and one extended figure for each dataset. Each
# line shows the mean 5-fold R-squared for a model, with error bars showing the
# Nadeau-Bengio corrected standard error across folds.

# %%
from pathlib import Path

import os
import tempfile

RESULTS_DIR = Path("../Results")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "thesis_matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd

SUMMARY_RESULTS_PATH = RESULTS_DIR / "all_results_corrected_standard_errors.csv"

DATASETS = {
    "Ames Housing": "ames",
    "King County": "king_county",
    "California Housing": "california",
}

MODEL_ORDER = ["OLS", "Random Forest", "XGBoost", "TabPFN"]


# %%
summary_results = pd.read_csv(SUMMARY_RESULTS_PATH)

summary = summary_results.loc[
    summary_results["dataset"].isin(DATASETS)
    & summary_results["model"].isin(MODEL_ORDER),
    [
        "dataset",
        "specification",
        "sample_size",
        "model",
        "r2_mean",
        "r2_corrected_se",
    ],
].copy()
summary = summary.sort_values(["dataset", "specification", "sample_size", "model"])


# %%
def plot_dataset_specification(dataset, specification):
    data = summary[
        summary["dataset"].eq(dataset) & summary["specification"].eq(specification)
    ].copy()
    if data.empty:
        raise ValueError(f"No {dataset} results found for {specification}.")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    models = [model for model in MODEL_ORDER if model in set(data["model"])]
    for model in models:
        model_data = data[data["model"].eq(model)].sort_values("sample_size")
        ax.errorbar(
            model_data["sample_size"],
            model_data["r2_mean"],
            yerr=model_data["r2_corrected_se"],
            marker="o",
            linewidth=2,
            capsize=4,
            label=model,
        )

    ax.set_xlabel("Sample size")
    ax.set_ylabel("R-squared")
    ax.set_xticks(sorted(data["sample_size"].unique()))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Model", frameon=False)

    y_min = max(0, data["r2_mean"].sub(data["r2_corrected_se"]).min() - 0.03)
    y_max = min(1, data["r2_mean"].add(data["r2_corrected_se"]).max() + 0.03)
    ax.set_ylim(y_min, y_max)

    fig.tight_layout()
    output_path = (
        RESULTS_DIR
        / f"{DATASETS[dataset]}_{specification.lower()}_sample_size_r2.png"
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# %%
for dataset in DATASETS:
    for specification in ["Parsimonious", "Extended"]:
        path = plot_dataset_specification(dataset, specification)
        print(f"Saved {path}")
