# %% [markdown]
# # King County: final model comparison
#
# This notebook consolidates the already-computed full-dataset King County
# results. It does not fit or evaluate any models.
#
# For the extended specification, the earlier one-hot encoded TabPFN result is
# replaced by the faster and slightly more accurate category-coded zipcode
# result. All other results are loaded unchanged from the original experiment.

# %%
import pandas as pd
from IPython.display import display

# %%
main_results = pd.read_csv("../Results/king_county_summary.csv")
category_tabpfn_folds = pd.read_csv(
    "../Results/king_county_tabpfn_category_zipcode_fold_results.csv"
)

category_tabpfn_summary = (
    category_tabpfn_folds[["rmse", "mae", "r2"]]
    .agg(["mean", "std"])
    .transpose()
)

category_tabpfn_row = pd.DataFrame(
    [
        {
            "dataset": "King County",
            "specification": "Extended",
            "sample_size": int(category_tabpfn_folds["sample_size"].iloc[0]),
            "outer_folds": 5,
            "model": "TabPFN",
            "rmse_mean": category_tabpfn_summary.loc["rmse", "mean"],
            "rmse_std": category_tabpfn_summary.loc["rmse", "std"],
            "mae_mean": category_tabpfn_summary.loc["mae", "mean"],
            "mae_std": category_tabpfn_summary.loc["mae", "std"],
            "r2_mean": category_tabpfn_summary.loc["r2", "mean"],
            "r2_std": category_tabpfn_summary.loc["r2", "std"],
            "zipcode_representation": "Category-coded",
        }
    ]
)

final_results = main_results[
    ~(
        (main_results["specification"] == "Extended")
        & (main_results["model"] == "TabPFN")
    )
].copy()
final_results["zipcode_representation"] = "Not applicable"
final_results.loc[
    final_results["specification"].eq("Extended"),
    "zipcode_representation",
] = "One-hot encoded"

final_results = pd.concat(
    [final_results, category_tabpfn_row],
    ignore_index=True,
)

model_order = ["Ridge", "Random Forest", "XGBoost", "TabPFN"]
final_results["model"] = pd.Categorical(
    final_results["model"],
    categories=model_order,
    ordered=True,
)
final_results = final_results.sort_values(["specification", "model"])
final_results["model"] = final_results["model"].astype(str)

final_results.to_csv("../Results/king_county_final_comparison.csv", index=False)

# %% [markdown]
# ## Final comparison table

# %%
comparison_table = final_results[
    [
        "specification",
        "model",
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "mae_std",
        "r2_mean",
        "r2_std",
        "zipcode_representation",
    ]
].copy()

numeric_columns = [
    "rmse_mean",
    "rmse_std",
    "mae_mean",
    "mae_std",
    "r2_mean",
    "r2_std",
]
comparison_table[numeric_columns] = comparison_table[numeric_columns].round(6)
display(comparison_table)

# %% [markdown]
# | Specification | Model | Mean RMSE | RMSE SD | Mean MAE | MAE SD | Mean R2 | R2 SD | Zipcode representation |
# |---|---|---:|---:|---:|---:|---:|---:|---|
# | Extended | Ridge | 0.188067 | 0.001125 | 0.137878 | 0.000839 | 0.872444 | 0.001662 | One-hot encoded |
# | Extended | Random Forest | 0.176938 | 0.001877 | 0.124391 | 0.001157 | 0.887057 | 0.003761 | One-hot encoded |
# | Extended | XGBoost | 0.167961 | 0.001547 | 0.121212 | 0.001147 | 0.898238 | 0.002828 | One-hot encoded |
# | Extended | **TabPFN** | **0.154373** | 0.001546 | **0.107448** | 0.000981 | **0.914030** | 0.002725 | Category-coded |
# | Parsimonious | Ridge | 0.356760 | 0.005287 | 0.285473 | 0.002603 | 0.540920 | 0.013622 | Not applicable |
# | Parsimonious | Random Forest | 0.336722 | 0.005044 | 0.264344 | 0.002400 | 0.591005 | 0.013677 | Not applicable |
# | Parsimonious | XGBoost | 0.333340 | 0.004605 | 0.262972 | 0.003181 | 0.599168 | 0.013343 | Not applicable |
# | Parsimonious | **TabPFN** | **0.326966** | 0.005218 | **0.255251** | 0.003270 | **0.614307** | 0.015016 | Not applicable |
