# %% [markdown]
# # California Housing: sample-size sensitivity experiment
#
# This notebook evaluates the California Housing models on nested subsamples of
# 1,000, 2,500, 5,000, and 10,000 observations. The full-dataset results are
# loaded from the already-computed full California experiment and are not rerun.

# %%
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor
from xgboost import XGBRegressor

RANDOM_STATE = 2
SAMPLE_SIZES = [1000, 2500, 5000, 10000]
FOLD_RESULTS_PATH = "../Results/california_sample_size_fold_results.csv"

# %%
raw_df = pd.read_csv("../Data/California_Housing.csv")
df = pd.DataFrame(
    {
        "MedInc": raw_df["median_income"],
        "HouseAge": raw_df["housing_median_age"],
        "AveRooms": raw_df["total_rooms"] / raw_df["households"],
        "AveBedrms": raw_df["total_bedrooms"] / raw_df["households"],
        "Population": raw_df["population"],
        "AveOccup": raw_df["population"] / raw_df["households"],
        "Latitude": raw_df["latitude"],
        "Longitude": raw_df["longitude"],
        "MedHouseVal": raw_df["median_house_value"],
    }
)
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["MedHouseVal"]).copy()
df["log_MedHouseVal"] = np.log(df["MedHouseVal"])

parsimonious_features_california = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
]

extended_features_california = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

print(f"Full observations: {len(df):,}")
print(f"Nested sample sizes: {SAMPLE_SIZES}")
print(f"Capped target observations in full data: {(df['MedHouseVal'] >= 500001).sum():,}")

# %% [markdown]
# A single seeded permutation defines the nested subsamples:
# 1,000 is contained in 2,500, 2,500 in 5,000, and 5,000 in 10,000.

# %%
rng = np.random.default_rng(RANDOM_STATE)
sample_order = rng.permutation(df.index.to_numpy())
sample_indices = {size: sample_order[:size] for size in SAMPLE_SIZES}


def make_preprocessor(features):
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                features,
            )
        ],
        verbose_feature_names_out=False,
    )


specifications = {
    "Parsimonious": (
        parsimonious_features_california,
        make_preprocessor(parsimonious_features_california),
    ),
    "Extended": (
        extended_features_california,
        make_preprocessor(extended_features_california),
    ),
}

ridge_param_grid = {"model__alpha": [0.001, 0.01, 0.1, 1, 10, 100]}

models = [
    ("OLS", LinearRegression(), None),
    ("Ridge", Ridge(), ridge_param_grid),
    (
        "Random Forest",
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
        {
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [1, 5],
        },
    ),
    (
        "XGBoost",
        XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        },
    ),
    (
        "TabPFN",
        TabPFNRegressor(
            ignore_pretraining_limits=True,
            random_state=RANDOM_STATE,
            show_progress_bar=False,
        ),
        None,
    ),
]

# %% [markdown]
# ## Nested cross-validation
#
# The outer loop uses 5 folds. Ridge, Random Forest, and XGBoost tune their
# hyperparameters with a 3-fold inner loop. TabPFN is not tuned.

# %%
outer_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

if pd.io.common.file_exists(FOLD_RESULTS_PATH):
    saved_fold_results = pd.read_csv(FOLD_RESULTS_PATH)
else:
    saved_fold_results = pd.DataFrame()


def already_completed(sample_size, specification, model_name, fold):
    if saved_fold_results.empty:
        return False
    completed = (
        (saved_fold_results["sample_size"] == sample_size)
        & (saved_fold_results["specification"] == specification)
        & (saved_fold_results["model"] == model_name)
        & (saved_fold_results["fold"] == fold)
    )
    return completed.any()


for sample_size in SAMPLE_SIZES:
    sample = df.loc[sample_indices[sample_size]]

    for specification, (features, preprocessor) in specifications.items():
        X = sample[features]
        y = sample["log_MedHouseVal"]

        for model_name, estimator, param_grid in models:
            for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
                if already_completed(sample_size, specification, model_name, fold):
                    print(
                        f"n={sample_size}, {specification}, "
                        f"{model_name}: skipping saved fold {fold}/5"
                    )
                    continue

                pipeline = Pipeline(
                    [
                        ("preprocessor", clone(preprocessor)),
                        ("model", clone(estimator)),
                    ]
                )

                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

                if param_grid:
                    search = GridSearchCV(
                        pipeline,
                        param_grid=param_grid,
                        scoring="neg_root_mean_squared_error",
                        cv=inner_cv,
                        n_jobs=-1,
                    )
                    search.fit(X_train, y_train)
                    fitted_model = search.best_estimator_
                    best_params = search.best_params_
                else:
                    fitted_model = pipeline.fit(X_train, y_train)
                    best_params = {}

                prediction = fitted_model.predict(X_test)
                fold_result = {
                    "dataset": "California Housing",
                    "specification": specification,
                    "sample_size": sample_size,
                    "outer_folds": 5,
                    "model": model_name,
                    "fold": fold,
                    "rmse": np.sqrt(mean_squared_error(y_test, prediction)),
                    "mae": mean_absolute_error(y_test, prediction),
                    "r2": r2_score(y_test, prediction),
                    "best_params": best_params,
                }
                saved_fold_results = pd.concat(
                    [saved_fold_results, pd.DataFrame([fold_result])],
                    ignore_index=True,
                )
                saved_fold_results.to_csv(FOLD_RESULTS_PATH, index=False)
                print(
                    f"n={sample_size}, {specification}, "
                    f"{model_name}: completed fold {fold}/5"
                )

# %%
fold_results = saved_fold_results.copy()
summary = (
    fold_results.groupby(
        ["dataset", "specification", "sample_size", "outer_folds", "model"]
    )[["rmse", "mae", "r2"]]
    .agg(["mean", "std"])
    .reset_index()
)

summary_export = summary.copy()
summary_export.columns = [
    "dataset",
    "specification",
    "sample_size",
    "outer_folds",
    "model",
    "rmse_mean",
    "rmse_std",
    "mae_mean",
    "mae_std",
    "r2_mean",
    "r2_std",
]

full_results = pd.read_csv("../Results/california_summary.csv")
combined_summary = pd.concat([summary_export, full_results], ignore_index=True)
model_order = ["OLS", "Ridge", "Random Forest", "XGBoost", "TabPFN"]
combined_summary["model"] = pd.Categorical(
    combined_summary["model"],
    categories=model_order,
    ordered=True,
)
combined_summary = combined_summary.sort_values(
    ["specification", "sample_size", "model"]
)
combined_summary["model"] = combined_summary["model"].astype(str)

summary_export.to_csv("../Results/california_sample_size_summary.csv", index=False)
combined_summary.to_csv(
    "../Results/california_sample_size_combined_summary.csv",
    index=False,
)

display(combined_summary)

# %% [markdown]
# ## Verified RMSE Summary
#
# | Specification | Sample size | OLS | Ridge | Random Forest | XGBoost | TabPFN |
# |---|---:|---:|---:|---:|---:|---:|
# | Parsimonious | 1,000 | 0.408140 | 0.408015 | 0.360347 | 0.358317 | **0.339519** |
# | Parsimonious | 2,500 | 0.407463 | 0.406959 | 0.348430 | 0.349847 | **0.337001** |
# | Parsimonious | 5,000 | 0.411817 | 0.412380 | 0.352729 | 0.351315 | **0.342626** |
# | Parsimonious | 10,000 | 0.405792 | 0.405910 | 0.353076 | 0.351194 | **0.344082** |
# | Parsimonious | 20,640 | 0.409996 | 0.410091 | 0.352557 | 0.351300 | **0.345741** |
# | Extended | 1,000 | 0.381379 | 0.381125 | 0.301076 | 0.263005 | **0.227712** |
# | Extended | 2,500 | 0.341089 | 0.341075 | 0.265374 | 0.243057 | **0.203359** |
# | Extended | 5,000 | 0.336575 | 0.336519 | 0.251779 | 0.236938 | **0.193971** |
# | Extended | 10,000 | 0.348842 | 0.348840 | 0.241352 | 0.236543 | **0.191183** |
# | Extended | 20,640 | 0.355332 | 0.355439 | 0.231832 | 0.233187 | **0.187873** |
#
# The full-dataset rows are loaded from the already-computed California full
# experiment. The `10,000` row is included because the requested nesting logic
# explicitly mentioned a 10,000-observation subsample.
