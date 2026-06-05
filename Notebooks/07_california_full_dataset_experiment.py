# %% [markdown]
# # California Housing: full-dataset model comparison
#
# This analysis compares parsimonious and extended feature specifications on
# all California Housing observations using 5-fold outer / 3-fold inner nested
# cross-validation. The target is the logarithm of median house value.
#
# The requested feature names follow the scikit-learn California Housing
# convention and are derived from the raw local CSV where necessary.

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
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor
from xgboost import XGBRegressor

RANDOM_STATE = 2
FOLD_RESULTS_PATH = "../Results/california_fold_results.csv"

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

print(f"Observations: {len(df):,}")
print(f"Parsimonious features: {len(parsimonious_features_california)}")
print(f"Extended features: {len(extended_features_california)}")
print(f"Capped target observations: {(df['MedHouseVal'] >= 500001).sum():,}")

# %% [markdown]
# Numeric values are median-imputed and standardized within each training fold.
# Standardization makes the Ridge alpha grid meaningful across differently
# scaled features.

# %%
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
# Every completed fold is saved immediately, allowing the experiment to resume
# without repeating finished work.

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
    y = df["log_MedHouseVal"]

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
                "dataset": "California Housing",
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
            saved_fold_results = pd.concat(
                [saved_fold_results, pd.DataFrame([fold_result])],
                ignore_index=True,
            )
            saved_fold_results.to_csv(FOLD_RESULTS_PATH, index=False)
            print(f"{specification}, {model_name}: completed fold {fold}/5")


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

summary_export.to_csv("../Results/california_summary.csv", index=False)

display(summary)
display(fold_results)

# %% [markdown]
# ## Verified full-dataset results
#
# | Specification | Model | Mean RMSE | RMSE SD | Mean MAE | MAE SD | Mean R2 | R2 SD |
# |---|---|---:|---:|---:|---:|---:|---:|
# | Parsimonious | Ridge | 0.410091 | 0.008898 | 0.312064 | 0.003638 | 0.480511 | 0.020203 |
# | Parsimonious | Random Forest | 0.352557 | 0.007746 | 0.265923 | 0.005130 | 0.616006 | 0.016317 |
# | Parsimonious | XGBoost | 0.351300 | 0.008035 | 0.265124 | 0.005453 | 0.618729 | 0.016896 |
# | Parsimonious | TabPFN | **0.345741** | 0.008157 | **0.259652** | 0.005941 | **0.630696** | 0.016756 |
# | Extended | Ridge | 0.355439 | 0.009794 | 0.269151 | 0.004251 | 0.609686 | 0.019655 |
# | Extended | Random Forest | 0.231832 | 0.009864 | 0.159436 | 0.004983 | 0.833806 | 0.013739 |
# | Extended | XGBoost | 0.233187 | 0.007530 | 0.166392 | 0.004289 | 0.831938 | 0.010628 |
# | Extended | TabPFN | **0.187873** | 0.009664 | **0.118991** | 0.003669 | **0.890783** | 0.010996 |
