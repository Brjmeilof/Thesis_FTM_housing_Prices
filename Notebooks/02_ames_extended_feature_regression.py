# %% [markdown]
# # Ames Housing: extended feature specification
#
# This analysis repeats the parsimonious Ames comparison using a richer set of
# structural, quality, amenity, age, and location variables. All preprocessing
# is fitted within each cross-validation training fold to prevent data leakage.

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

# %%
df = pd.read_csv("../Data/AmesHousing.csv")

numeric_features = [
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

categorical_features = [
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

extended_features = numeric_features + categorical_features
target = "SalePrice"

df_model = df[extended_features + [target]].copy()
df_model = df_model.dropna(subset=[target])
df_model["log_SalePrice"] = np.log(df_model[target])

X = df_model[extended_features]
y = df_model["log_SalePrice"]

print(f"Observations: {len(df_model)}")
print(f"Raw extended features: {len(extended_features)}")
print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

# %% [markdown]
# Numeric missing values are median-imputed. Categorical missing values are
# assigned an explicit `Missing` category and then one-hot encoded. The
# preprocessing pipeline is cloned and fitted separately inside every fold.

# %%
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_features,
        ),
        (
            "categorical",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="constant",
                            fill_value="Missing",
                        ),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]
            ),
            categorical_features,
        ),
    ],
    verbose_feature_names_out=False,
)

ridge_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        ),
        (
            "categorical",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="constant",
                            fill_value="Missing",
                        ),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]
            ),
            categorical_features,
        ),
    ],
    verbose_feature_names_out=False,
)

preview_preprocessor = clone(preprocessor).fit(X)
print(f"Features after preprocessing: {len(preview_preprocessor.get_feature_names_out())}")

# %% [markdown]
# ## Nested cross-validation
#
# The outer loop uses 5 folds to estimate out-of-sample performance. Random
# Forest, XGBoost, and Ridge are tuned using a 3-fold grid search within each
# outer training fold. OLS and TabPFN are not tuned.

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

        active_preprocessor = ridge_preprocessor if model_name == "Ridge" else preprocessor

        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(active_preprocessor)),
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
# Results from the complete 5-fold outer / 3-fold inner nested cross-validation
# run using all 2,930 observations.
#
# | Model | Mean RMSE | RMSE SD | Mean MAE | MAE SD | Mean R2 | R2 SD |
# |---|---:|---:|---:|---:|---:|---:|
# | TabPFN | 0.119426 | 0.016591 | 0.077258 | 0.003968 | 0.912864 | 0.023524 |
# | XGBoost | 0.129579 | 0.013063 | 0.085851 | 0.003375 | 0.898162 | 0.019673 |
# | Random Forest | 0.138895 | 0.012219 | 0.093178 | 0.004301 | 0.883225 | 0.019251 |
# | Ridge | 0.140814 | 0.023726 | 0.091521 | 0.001972 | 0.878131 | 0.040159 |
# | OLS | 0.140997 | 0.023014 | 0.092164 | 0.001338 | 0.877946 | 0.038979 |
