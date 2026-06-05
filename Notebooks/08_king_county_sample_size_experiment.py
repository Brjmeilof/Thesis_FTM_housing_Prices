# %% [markdown]
# # King County: sample-size sensitivity experiment
#
# This notebook evaluates the King County models on nested subsamples of
# 1,000, 2,500, and 5,000 observations. The full-dataset results are loaded
# from the already-computed final comparison table and are not rerun.
#
# The extended TabPFN specification uses category-coded `zipcode`, matching the
# improved King County TabPFN experiment. Ridge, Random Forest, and XGBoost use
# the one-hot encoded extended specification from the final comparison.

# %%
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from tabpfn import TabPFNRegressor
from xgboost import XGBRegressor

RANDOM_STATE = 2
SAMPLE_SIZES = [1000, 2500, 5000]
FOLD_RESULTS_PATH = "../Results/king_county_sample_size_fold_results.csv"

# %%
df = pd.read_csv("../Data/kc_house_data.csv").dropna(subset=["price"]).copy()
df["log_price"] = np.log(df["price"])
df["zipcode"] = df["zipcode"].astype(str)

parsimonious_features_king = [
    "sqft_living",
    "bathrooms",
    "bedrooms",
    "floors",
    "sqft_lot",
    "yr_built",
]

extended_features_king = [
    "sqft_living",
    "sqft_lot",
    "bathrooms",
    "bedrooms",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
]

extended_numeric_features_king = [
    feature for feature in extended_features_king if feature != "zipcode"
]
extended_categorical_features_king = ["zipcode"]

print(f"Full observations: {len(df):,}")
print(f"Nested sample sizes: {SAMPLE_SIZES}")

# %% [markdown]
# A single seeded permutation defines the nested subsamples:
# the first 1,000 observations are contained in the first 2,500, and the first
# 2,500 are contained in the first 5,000.

# %%
rng = np.random.default_rng(RANDOM_STATE)
sample_order = rng.permutation(df.index.to_numpy())
sample_indices = {size: sample_order[:size] for size in SAMPLE_SIZES}


def numeric_preprocessor(features):
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


parsimonious_preprocessor = numeric_preprocessor(parsimonious_features_king)

extended_onehot_preprocessor = ColumnTransformer(
    [
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            extended_numeric_features_king,
        ),
        (
            "categorical",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            ),
            extended_categorical_features_king,
        ),
    ],
    verbose_feature_names_out=False,
)

extended_category_preprocessor = ColumnTransformer(
    [
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            extended_numeric_features_king,
        ),
        (
            "categorical",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "ordinal",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]
            ),
            extended_categorical_features_king,
        ),
    ],
    verbose_feature_names_out=False,
)

ridge_param_grid = {"model__alpha": [0.001, 0.01, 0.1, 1, 10, 100]}

standard_models = [
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
]

tabpfn = TabPFNRegressor(
    ignore_pretraining_limits=True,
    random_state=RANDOM_STATE,
    show_progress_bar=False,
)

jobs = []
for specification, features, preprocessor in [
    ("Parsimonious", parsimonious_features_king, parsimonious_preprocessor),
    ("Extended", extended_features_king, extended_onehot_preprocessor),
]:
    for model_name, estimator, param_grid in standard_models:
        jobs.append(
            {
                "specification": specification,
                "features": features,
                "preprocessor": preprocessor,
                "model_name": model_name,
                "estimator": estimator,
                "param_grid": param_grid,
                "zipcode_representation": (
                    "One-hot encoded" if specification == "Extended" else "Not applicable"
                ),
            }
        )

jobs.append(
    {
        "specification": "Parsimonious",
        "features": parsimonious_features_king,
        "preprocessor": parsimonious_preprocessor,
        "model_name": "TabPFN",
        "estimator": tabpfn,
        "param_grid": None,
        "zipcode_representation": "Not applicable",
    }
)
jobs.append(
    {
        "specification": "Extended",
        "features": extended_features_king,
        "preprocessor": extended_category_preprocessor,
        "model_name": "TabPFN",
        "estimator": TabPFNRegressor(
            categorical_features_indices=[15],
            ignore_pretraining_limits=True,
            random_state=RANDOM_STATE,
            show_progress_bar=False,
        ),
        "param_grid": None,
        "zipcode_representation": "Category-coded",
    }
)

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

    for job in jobs:
        X = sample[job["features"]]
        y = sample["log_price"]

        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
            if already_completed(
                sample_size,
                job["specification"],
                job["model_name"],
                fold,
            ):
                print(
                    f"n={sample_size}, {job['specification']}, "
                    f"{job['model_name']}: skipping saved fold {fold}/5"
                )
                continue

            pipeline = Pipeline(
                [
                    ("preprocessor", clone(job["preprocessor"])),
                    ("model", clone(job["estimator"])),
                ]
            )

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            if job["param_grid"]:
                search = GridSearchCV(
                    pipeline,
                    param_grid=job["param_grid"],
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
                "dataset": "King County",
                "specification": job["specification"],
                "sample_size": sample_size,
                "outer_folds": 5,
                "model": job["model_name"],
                "fold": fold,
                "rmse": np.sqrt(mean_squared_error(y_test, prediction)),
                "mae": mean_absolute_error(y_test, prediction),
                "r2": r2_score(y_test, prediction),
                "best_params": best_params,
                "zipcode_representation": job["zipcode_representation"],
            }
            saved_fold_results = pd.concat(
                [saved_fold_results, pd.DataFrame([fold_result])],
                ignore_index=True,
            )
            saved_fold_results.to_csv(FOLD_RESULTS_PATH, index=False)
            print(
                f"n={sample_size}, {job['specification']}, "
                f"{job['model_name']}: completed fold {fold}/5"
            )

# %%
fold_results = saved_fold_results.copy()
summary = (
    fold_results.groupby(
        [
            "dataset",
            "specification",
            "sample_size",
            "outer_folds",
            "model",
            "zipcode_representation",
        ]
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
    "zipcode_representation",
    "rmse_mean",
    "rmse_std",
    "mae_mean",
    "mae_std",
    "r2_mean",
    "r2_std",
]

full_results = pd.read_csv("../Results/king_county_final_comparison.csv")
full_results = full_results[
    [
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
    ]
]

combined_summary = pd.concat([summary_export, full_results], ignore_index=True)
model_order = ["Ridge", "Random Forest", "XGBoost", "TabPFN"]
combined_summary["model"] = pd.Categorical(
    combined_summary["model"],
    categories=model_order,
    ordered=True,
)
combined_summary = combined_summary.sort_values(
    ["specification", "sample_size", "model"]
)
combined_summary["model"] = combined_summary["model"].astype(str)

summary_export.to_csv("../Results/king_county_sample_size_summary.csv", index=False)
combined_summary.to_csv(
    "../Results/king_county_sample_size_combined_summary.csv",
    index=False,
)

display(combined_summary)

# %% [markdown]
# ## Verified RMSE Summary
#
# | Specification | Sample size | Ridge | Random Forest | XGBoost | TabPFN |
# |---|---:|---:|---:|---:|---:|
# | Parsimonious | 1,000 | 0.368096 | 0.365157 | 0.353646 | **0.345251** |
# | Parsimonious | 2,500 | 0.361837 | 0.354256 | 0.347143 | **0.336494** |
# | Parsimonious | 5,000 | 0.357988 | 0.345835 | 0.340159 | **0.334469** |
# | Parsimonious | 21,613 | 0.356760 | 0.336722 | 0.333340 | **0.326966** |
# | Extended | 1,000 | 0.207583 | 0.223803 | 0.205554 | **0.184276** |
# | Extended | 2,500 | 0.191946 | 0.199839 | 0.181544 | **0.167711** |
# | Extended | 5,000 | 0.188180 | 0.192086 | 0.172947 | **0.161251** |
# | Extended | 21,613 | 0.188067 | 0.176938 | 0.167961 | **0.154373** |
#
# The full-dataset rows are loaded from the already-computed final King County
# comparison. The extended TabPFN rows use category-coded `zipcode`.
