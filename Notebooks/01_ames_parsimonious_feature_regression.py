# %% [markdown]
# # Ames Housing: parsimonious feature specification
#
# This analysis compares OLS, Random Forest, XGBoost, and TabPFN using six core
# structural housing variables. The outer cross-validation loop uses five folds.

# %%
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from tabpfn import TabPFNRegressor
from xgboost import XGBRegressor

RANDOM_STATE = 2

# %%
df = pd.read_csv("../Data/AmesHousing.csv")

features = [
    "Gr Liv Area",
    "Full Bath",
    "Bedroom AbvGr",
    "Garage Cars",
    "Garage Area",
    "TotRms AbvGrd",
]
target = "SalePrice"

df_model = df[features + [target]].dropna(subset=[target]).copy()
df_model["log_SalePrice"] = np.log(df_model[target])

X = df_model[features]
y = df_model["log_SalePrice"]

print(f"Observations: {len(df_model)}")
print(f"Parsimonious features: {len(features)}")

# %% [markdown]
# Missing predictor values are median-imputed within each training fold. Random
# Forest and XGBoost are tuned using 3-fold inner cross-validation.

# %%
outer_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)


def evaluate_nested_cv(model_name, estimator, param_grid=None):
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        X_train_cv = X.iloc[train_idx]
        X_test_cv = X.iloc[test_idx]
        y_train_cv = y.iloc[train_idx]
        y_test_cv = y.iloc[test_idx]

        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", clone(estimator)),
            ]
        )

        if param_grid:
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring="neg_root_mean_squared_error",
                cv=inner_cv,
                n_jobs=-1,
            )
            search.fit(X_train_cv, y_train_cv)
            fitted_model = search.best_estimator_
            best_params = search.best_params_
        else:
            fitted_model = pipeline.fit(X_train_cv, y_train_cv)
            best_params = {}

        y_pred_cv = fitted_model.predict(X_test_cv)
        fold_results.append(
            {
                "model": model_name,
                "fold": fold,
                "rmse": np.sqrt(mean_squared_error(y_test_cv, y_pred_cv)),
                "mae": mean_absolute_error(y_test_cv, y_pred_cv),
                "r2": r2_score(y_test_cv, y_pred_cv),
                "best_params": best_params,
            }
        )
        print(f"{model_name}: completed outer fold {fold}/5")

    return pd.DataFrame(fold_results)


models = [
    ("OLS", LinearRegression(), None),
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

cv_results = pd.concat(
    [
        evaluate_nested_cv(model_name, estimator, param_grid)
        for model_name, estimator, param_grid in models
    ],
    ignore_index=True,
)

# %%
cv_summary = (
    cv_results.groupby("model")[["rmse", "mae", "r2"]]
    .agg(["mean", "std"])
    .sort_values(("rmse", "mean"))
)

display(cv_summary)
display(cv_results)

# %% [markdown]
# ## Verified full-run results
#
# These results come from the completed 5-fold outer / 3-fold inner nested
# cross-validation run using all 2,930 observations.
#
# | Model | Mean RMSE | RMSE SD | Mean MAE | MAE SD | Mean R2 | R2 SD |
# |---|---:|---:|---:|---:|---:|---:|
# | TabPFN | 0.201965 | 0.016294 | 0.139957 | 0.006172 | 0.753402 | 0.036504 |
# | XGBoost | 0.209523 | 0.011986 | 0.148970 | 0.005775 | 0.735212 | 0.026364 |
# | Random Forest | 0.211452 | 0.013984 | 0.148038 | 0.005996 | 0.730115 | 0.032013 |
# | OLS | 0.237455 | 0.022729 | 0.166787 | 0.009069 | 0.658501 | 0.061105 |
