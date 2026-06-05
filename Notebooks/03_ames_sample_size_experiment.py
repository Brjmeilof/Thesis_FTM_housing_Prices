# %% [markdown]
# # Ames Housing: sample-size sensitivity experiment
#
# This experiment compares the parsimonious and extended feature specifications
# at 1,000, 2,000, and all 2,930 observations. It uses identical, nested random
# samples for both specifications and 5-fold outer / 3-fold inner nested CV.

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
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor
from xgboost import XGBRegressor

RANDOM_STATE = 2
SUBSAMPLE_SIZES = [1000, 2000]

# %%
df = pd.read_csv("../Data/AmesHousing.csv").dropna(subset=["SalePrice"]).copy()
df["log_SalePrice"] = np.log(df["SalePrice"])
SAMPLE_SIZES = SUBSAMPLE_SIZES + [len(df)]

parsimonious_features = [
    "Gr Liv Area",
    "Full Bath",
    "Bedroom AbvGr",
    "Garage Cars",
    "Garage Area",
    "TotRms AbvGrd",
]

extended_numeric_features = [
    "Gr Liv Area",
    "Total Bsmt SF",
    "1st Flr SF",
    "2nd Flr SF",
    "Full Bath",
    "Half Bath",
    "Bedroom AbvGr",
    "TotRms AbvGrd",
    "Garage Cars",
    "Garage Area",
    "Year Built",
    "Year Remod/Add",
    "Overall Qual",
    "Overall Cond",
    "Fireplaces",
    "Wood Deck SF",
    "Open Porch SF",
    "Lot Area",
]

extended_categorical_features = [
    "Neighborhood",
    "House Style",
    "Bldg Type",
    "Kitchen Qual",
    "Exter Qual",
    "Bsmt Qual",
    "Garage Qual",
    "Central Air",
    "Paved Drive",
]
extended_features = extended_numeric_features + extended_categorical_features

parsimonious_preprocessor = ColumnTransformer(
    [("numeric", SimpleImputer(strategy="median"), parsimonious_features)],
    verbose_feature_names_out=False,
)

parsimonious_ridge_preprocessor = ColumnTransformer(
    [
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            parsimonious_features,
        )
    ],
    verbose_feature_names_out=False,
)

extended_preprocessor = ColumnTransformer(
    [
        (
            "numeric",
            SimpleImputer(strategy="median"),
            extended_numeric_features,
        ),
        (
            "categorical",
            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="Missing"),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            ),
            extended_categorical_features,
        ),
    ],
    verbose_feature_names_out=False,
)

extended_ridge_preprocessor = ColumnTransformer(
    [
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            extended_numeric_features,
        ),
        (
            "categorical",
            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="Missing"),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            ),
            extended_categorical_features,
        ),
    ],
    verbose_feature_names_out=False,
)

specifications = {
    "Parsimonious": (parsimonious_features, parsimonious_preprocessor),
    "Extended": (extended_features, extended_preprocessor),
}

ridge_preprocessors = {
    "Parsimonious": parsimonious_ridge_preprocessor,
    "Extended": extended_ridge_preprocessor,
}

# %% [markdown]
# A single seeded permutation defines the subsamples. The first 1,000 rows are
# contained within the first 2,000 rows, and both feature specifications use
# exactly the same observations at each sample size.

# %%
rng = np.random.default_rng(RANDOM_STATE)
sample_order = rng.permutation(df.index.to_numpy())
sample_indices = {size: sample_order[:size] for size in SAMPLE_SIZES}

inner_cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

models = [
    ("OLS", LinearRegression(), None),
    (
        "Ridge",
        Ridge(),
        {
            "model__alpha": [0.001, 0.01, 0.1, 1, 10, 100],
        },
    ),
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


def evaluate_sample(specification, features, preprocessor, sample_size):
    sample = df.loc[sample_indices[sample_size]]
    X = sample[features]
    y = sample["log_SalePrice"]
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []

    for model_name, estimator, param_grid in models:
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            active_preprocessor = (
                ridge_preprocessors[specification]
                if model_name == "Ridge"
                else preprocessor
            )

            pipeline = Pipeline(
                [
                    ("preprocessor", clone(active_preprocessor)),
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
            fold_results.append(
                {
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
            )
            print(
                f"{specification}, n={sample_size}, {model_name}: "
                f"completed fold {fold}/5"
            )

    return pd.DataFrame(fold_results)


sample_results = pd.concat(
    [
        evaluate_sample(specification, features, preprocessor, sample_size)
        for specification, (features, preprocessor) in specifications.items()
        for sample_size in SAMPLE_SIZES
    ],
    ignore_index=True,
)

# %%
sample_summary = (
    sample_results.groupby(["specification", "sample_size", "outer_folds", "model"])[
        ["rmse", "mae", "r2"]
    ]
    .agg(["mean", "std"])
    .reset_index()
)
display(sample_summary)

sample_results.to_csv("../Results/ames_sample_size_fold_results.csv", index=False)
sample_summary_export = sample_summary.copy()
sample_summary_export.columns = [
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
sample_summary_export.to_csv("../Results/ames_sample_size_summary.csv", index=False)

# %%
sample_summary_export["result_source"] = "5-fold sample-size experiment"
combined_summary = sample_summary_export.copy()
combined_summary.to_csv("../Results/ames_sample_size_combined_summary.csv", index=False)
display(combined_summary)

# %% [markdown]
# ## Verified RMSE summary including Ridge
#
# Ridge was added after the original Ames sample-size run and tuned with a
# 3-fold inner grid search over `alpha = [0.001, 0.01, 0.1, 1, 10, 100]`.
# Its rows are included in the saved CSV files:
#
# | Specification | Sample size | OLS | Ridge | Random Forest | XGBoost | TabPFN |
# |---|---:|---:|---:|---:|---:|---:|
# | Parsimonious | 1,000 | 0.236075 | 0.236194 | 0.226967 | 0.222343 | **0.208530** |
# | Parsimonious | 2,000 | 0.240026 | 0.239977 | 0.217613 | 0.215926 | **0.205178** |
# | Parsimonious | 2,930 | 0.237455 | 0.237450 | 0.211452 | 0.209523 | **0.201965** |
# | Extended | 1,000 | 0.151978 | 0.149649 | 0.161853 | 0.143656 | **0.131240** |
# | Extended | 2,000 | 0.152694 | 0.153621 | 0.150353 | 0.137431 | **0.128497** |
# | Extended | 2,930 | 0.140997 | 0.140814 | 0.138895 | 0.129579 | **0.119426** |
