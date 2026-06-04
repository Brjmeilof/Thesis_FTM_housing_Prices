# %% [markdown]
# # King County housing: full-dataset model comparison
#
# This analysis compares parsimonious and extended feature specifications on
# all King County observations using 5-fold outer / 3-fold inner nested
# cross-validation. The target is the logarithm of price.
#
# Ridge replaces OLS and its regularization parameter is selected within each
# outer training fold. Random Forest and XGBoost are tuned in the same manner,
# while TabPFN is used without explicit hyperparameter tuning.

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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tabpfn import TabPFNRegressor
from xgboost import XGBRegressor

RANDOM_STATE = 2
FOLD_RESULTS_PATH = "../Results/king_county_fold_results.csv"

# %%
df = pd.read_csv("../Data/kc_house_data.csv").dropna(subset=["price"]).copy()
df["log_price"] = np.log(df["price"])

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

print(f"Observations: {len(df):,}")
print(f"Parsimonious features: {len(parsimonious_features_king)}")
print(f"Extended features: {len(extended_features_king)}")

# %% [markdown]
# Numeric values are median-imputed and standardized within each training fold.
# `zipcode` is treated as a categorical location indicator and one-hot encoded.
# Standardization makes the Ridge alpha grid meaningful across differently
# scaled variables and does not change tree-based model split possibilities.

# %%
parsimonious_preprocessor = ColumnTransformer(
    [
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            parsimonious_features_king,
        )
    ],
    verbose_feature_names_out=False,
)

extended_preprocessor = ColumnTransformer(
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
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            ),
            extended_categorical_features_king,
        ),
    ],
    verbose_feature_names_out=False,
)

specifications = {
    "Parsimonious": (parsimonious_features_king, parsimonious_preprocessor),
    "Extended": (extended_features_king, extended_preprocessor),
}

ridge_param_grid = {
    "model__alpha": [0.001, 0.01, 0.1, 1, 10, 100],
}

models = [
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
# The outer loop uses five folds to estimate out-of-sample performance. All
# preprocessing and model selection occur exclusively within the corresponding
# outer training data.

# %%
outer_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
if pd.io.common.file_exists(FOLD_RESULTS_PATH):
    saved_fold_results = pd.read_csv(FOLD_RESULTS_PATH)
else:
    saved_fold_results = pd.DataFrame()


def evaluate_specification(specification, features, preprocessor):
    global saved_fold_results
    X = df[features]
    y = df["log_price"]
    fold_results = []

    for model_name, estimator, param_grid in models:
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
            if not saved_fold_results.empty:
                completed = (
                    (saved_fold_results["specification"] == specification)
                    & (saved_fold_results["model"] == model_name)
                    & (saved_fold_results["fold"] == fold)
                ).any()
                if completed:
                    print(f"{specification}, {model_name}: skipping saved fold {fold}/5")
                    continue

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            pipeline = Pipeline(
                [
                    ("preprocessor", clone(preprocessor)),
                    ("model", clone(estimator)),
                ]
            )

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
                "dataset": "King County",
                "specification": specification,
                "sample_size": len(df),
                "outer_folds": 5,
                "model": model_name,
                "fold": fold,
                "rmse": np.sqrt(mean_squared_error(y_test, prediction)),
                "mae": mean_absolute_error(y_test, prediction),
                "r2": r2_score(y_test, prediction),
                "best_params": best_params,
            }
            fold_results.append(fold_result)
            saved_fold_results = pd.concat(
                [saved_fold_results, pd.DataFrame([fold_result])],
                ignore_index=True,
            )
            saved_fold_results.to_csv(FOLD_RESULTS_PATH, index=False)
            print(f"{specification}, {model_name}: completed fold {fold}/5")

    return pd.DataFrame(fold_results)


for specification, (features, preprocessor) in specifications.items():
    evaluate_specification(specification, features, preprocessor)

fold_results = saved_fold_results.copy()

# %%
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

fold_results.to_csv("../Results/king_county_fold_results.csv", index=False)
summary_export.to_csv("../Results/king_county_summary.csv", index=False)

display(summary)
display(fold_results)

# %% [markdown]
# ## Verified full-dataset results
#
# | Specification | Model | Mean RMSE | RMSE SD | Mean MAE | MAE SD | Mean R2 | R2 SD |
# |---|---|---:|---:|---:|---:|---:|---:|
# | Parsimonious | Ridge | 0.356760 | 0.005287 | 0.285473 | 0.002603 | 0.540920 | 0.013622 |
# | Parsimonious | Random Forest | 0.336722 | 0.005044 | 0.264344 | 0.002400 | 0.591005 | 0.013677 |
# | Parsimonious | XGBoost | 0.333340 | 0.004605 | 0.262972 | 0.003181 | 0.599168 | 0.013343 |
# | Parsimonious | TabPFN | **0.326966** | 0.005218 | **0.255251** | 0.003270 | **0.614307** | 0.015016 |
# | Extended | Ridge | 0.188067 | 0.001125 | 0.137878 | 0.000839 | 0.872444 | 0.001662 |
# | Extended | Random Forest | 0.176938 | 0.001877 | 0.124391 | 0.001157 | 0.887057 | 0.003761 |
# | Extended | XGBoost | 0.167961 | 0.001547 | 0.121212 | 0.001147 | 0.898238 | 0.002828 |
# | Extended | TabPFN | **0.155473** | 0.001502 | **0.108372** | 0.000830 | **0.912799** | 0.002803 |
