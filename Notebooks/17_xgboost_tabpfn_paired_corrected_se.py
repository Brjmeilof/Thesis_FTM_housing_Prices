# %% [markdown]
# # Paired XGBoost vs TabPFN corrected standard errors
#
# Computes paired Nadeau-Bengio corrected standard errors for fold-level
# performance differences between XGBoost and TabPFN.

# %%
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "Results"
FOLD_RESULTS_PATH = RESULTS_DIR / "all_results_corrected_standard_errors_fold_results.csv"

MODEL_ORDER = ["XGBoost", "TabPFN"]
GROUP_COLUMNS = ["dataset", "specification", "sample_size"]
METRICS = ["rmse", "mae", "r2"]
N_TEST_N_TRAIN_RATIO = 0.25

FULL_SAMPLE_SIZES = {
    "Ames Housing": 2930,
    "King County": 21613,
    "California Housing": 20640,
}

# Exact two-sided 95% t critical values for small df commonly produced here.
T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


# %%
fold_results = pd.read_csv(FOLD_RESULTS_PATH)
paired_rows = fold_results[fold_results["model"].isin(MODEL_ORDER)].copy()

summary_rows = []

for group_values, group in paired_rows.groupby(GROUP_COLUMNS, sort=True):
    group_key = dict(zip(GROUP_COLUMNS, group_values))

    for metric in METRICS:
        aligned = group.pivot_table(
            index="fold",
            columns="model",
            values=metric,
            aggfunc="first",
        ).dropna(subset=MODEL_ORDER)

        n_folds = len(aligned)
        if n_folds < 2:
            continue

        if metric in ["rmse", "mae"]:
            differences = aligned["XGBoost"] - aligned["TabPFN"]
        else:
            differences = aligned["TabPFN"] - aligned["XGBoost"]

        mean_difference = differences.mean()
        sd_difference = differences.std(ddof=1)
        corrected_se_difference = np.sqrt(
            (1 / n_folds) + N_TEST_N_TRAIN_RATIO
        ) * sd_difference

        df = n_folds - 1
        if df not in T_CRIT_95:
            raise ValueError(f"No configured 95% t critical value for df={df}.")
        t_crit = T_CRIT_95[df]

        summary_rows.append(
            {
                **group_key,
                "metric": metric.upper() if metric == "r2" else metric.upper(),
                "mean_difference": mean_difference,
                "sd_difference": sd_difference,
                "corrected_se_difference": corrected_se_difference,
                "ci_lower": mean_difference - t_crit * corrected_se_difference,
                "ci_upper": mean_difference + t_crit * corrected_se_difference,
                "n_folds": n_folds,
            }
        )

summary = pd.DataFrame(summary_rows).rename(
    columns={"specification": "feature_specification"}
)

summary = summary[
    [
        "dataset",
        "feature_specification",
        "sample_size",
        "metric",
        "mean_difference",
        "sd_difference",
        "corrected_se_difference",
        "ci_lower",
        "ci_upper",
        "n_folds",
    ]
].sort_values(["dataset", "feature_specification", "sample_size", "metric"])

full_sample = summary[
    summary.apply(
        lambda row: FULL_SAMPLE_SIZES.get(row["dataset"]) == row["sample_size"],
        axis=1,
    )
].copy()

summary_path = RESULTS_DIR / "xgboost_tabpfn_paired_corrected_se_differences.csv"
full_sample_path = (
    RESULTS_DIR / "xgboost_tabpfn_full_sample_paired_corrected_se_differences.csv"
)

summary.to_csv(summary_path, index=False)
full_sample.to_csv(full_sample_path, index=False)

print(f"Saved {summary_path}")
print(f"Saved {full_sample_path}")
print(full_sample.round(4).to_string(index=False))
