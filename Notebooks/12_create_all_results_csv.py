# %% [markdown]
# # Create consolidated result CSVs
#
# This notebook combines the final summary-level and fold-level results across
# Ames, King County, and California Housing.

# %%
import pandas as pd
from IPython.display import display

# %%
summary_sources = [
    ("Ames Housing", "../Results/ames_sample_size_combined_summary.csv"),
    ("King County", "../Results/king_county_sample_size_combined_summary.csv"),
    ("California Housing", "../Results/california_sample_size_combined_summary.csv"),
]

summary_frames = []
for dataset, path in summary_sources:
    frame = pd.read_csv(path)
    if "dataset" not in frame.columns:
        frame.insert(0, "dataset", dataset)
    else:
        frame["dataset"] = frame["dataset"].fillna(dataset)
    if "zipcode_representation" not in frame.columns:
        frame["zipcode_representation"] = "Not applicable"
    if "result_source" not in frame.columns:
        frame["result_source"] = "5-fold sample-size experiment"
    summary_frames.append(frame)

all_results_summary = pd.concat(summary_frames, ignore_index=True)

summary_columns = [
    "dataset",
    "specification",
    "sample_size",
    "outer_folds",
    "model",
    "zipcode_representation",
    "rmse_mean",
    "rmse_std",
    "mae_mean",
    "mae_std",
    "r2_mean",
    "r2_std",
    "result_source",
]
all_results_summary = all_results_summary[summary_columns]

dataset_order = {
    "Ames Housing": 1,
    "King County": 2,
    "California Housing": 3,
}
model_order = {
    "OLS": 1,
    "Ridge": 2,
    "Random Forest": 3,
    "XGBoost": 4,
    "TabPFN": 5,
}
specification_order = {
    "Parsimonious": 1,
    "Extended": 2,
}

all_results_summary["_dataset_order"] = all_results_summary["dataset"].map(dataset_order)
all_results_summary["_specification_order"] = all_results_summary["specification"].map(
    specification_order
)
all_results_summary["_model_order"] = all_results_summary["model"].map(model_order)
all_results_summary = all_results_summary.sort_values(
    ["_dataset_order", "_specification_order", "sample_size", "_model_order"]
).drop(columns=["_dataset_order", "_specification_order", "_model_order"])

all_results_summary.to_csv("../Results/all_results_summary.csv", index=False)
display(all_results_summary)

# %%
fold_sources = [
    ("Ames Housing", "../Results/ames_sample_size_fold_results.csv"),
    ("King County", "../Results/king_county_sample_size_fold_results.csv"),
    ("California Housing", "../Results/california_sample_size_fold_results.csv"),
]

fold_frames = []
for dataset, path in fold_sources:
    frame = pd.read_csv(path)
    if "dataset" not in frame.columns:
        frame.insert(0, "dataset", dataset)
    else:
        frame["dataset"] = frame["dataset"].fillna(dataset)
    if "zipcode_representation" not in frame.columns:
        frame["zipcode_representation"] = "Not applicable"
    fold_frames.append(frame)

all_results_folds = pd.concat(fold_frames, ignore_index=True)
fold_columns = [
    "dataset",
    "specification",
    "sample_size",
    "outer_folds",
    "model",
    "fold",
    "rmse",
    "mae",
    "r2",
    "best_params",
    "zipcode_representation",
]
all_results_folds = all_results_folds[fold_columns]

all_results_folds["_dataset_order"] = all_results_folds["dataset"].map(dataset_order)
all_results_folds["_specification_order"] = all_results_folds["specification"].map(
    specification_order
)
all_results_folds["_model_order"] = all_results_folds["model"].map(model_order)
all_results_folds = all_results_folds.sort_values(
    [
        "_dataset_order",
        "_specification_order",
        "sample_size",
        "_model_order",
        "fold",
    ]
).drop(columns=["_dataset_order", "_specification_order", "_model_order"])

all_results_folds.to_csv("../Results/all_results_fold_results.csv", index=False)
display(all_results_folds)

