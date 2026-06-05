# %% [markdown]
# # Ames Housing: add Ridge regression to existing comparisons
#
# This script computes only the missing Ridge rows for the Ames sample-size
# experiment. It reuses the same nested sample order, feature specifications,
# and 5-fold outer / 3-fold inner cross-validation design as
# `03_ames_sample_size_experiment.py`.

# %%
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 2
FOLD_RESULTS_PATH = "../Results/ames_sample_size_fold_results.csv"

# %%
df = pd.read_csv("../Data/AmesHousing.csv").dropna(subset=["SalePrice"]).copy()
df["log_SalePrice"] = np.log(df["SalePrice"])
SAMPLE_SIZES = [1000, 2000, len(df)]

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

ridge_param_grid = {
    "model__alpha": [0.001, 0.01, 0.1, 1, 10, 100],
}

# %%
rng = np.random.default_rng(RANDOM_STATE)
sample_order = rng.permutation(df.index.to_numpy())
sample_indices = {size: sample_order[:size] for size in SAMPLE_SIZES}

outer_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

fold_results = pd.read_csv(FOLD_RESULTS_PATH)


def already_completed(sample_size, specification, fold):
    completed = (
        (fold_results["sample_size"] == sample_size)
        & (fold_results["specification"] == specification)
        & (fold_results["model"] == "Ridge")
        & (fold_results["fold"] == fold)
    )
    return completed.any()


for sample_size in SAMPLE_SIZES:
    sample = df.loc[sample_indices[sample_size]]

    for specification, (features, preprocessor) in specifications.items():
        X = sample[features]
        y = sample["log_SalePrice"]

        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
            if already_completed(sample_size, specification, fold):
                print(f"n={sample_size}, {specification}, Ridge: skipping fold {fold}/5")
                continue

            pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    ("model", Ridge()),
                ]
            )
            search = GridSearchCV(
                pipeline,
                param_grid=ridge_param_grid,
                scoring="neg_root_mean_squared_error",
                cv=inner_cv,
                n_jobs=-1,
            )

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            search.fit(X_train, y_train)
            prediction = search.best_estimator_.predict(X_test)

            fold_result = {
                "specification": specification,
                "sample_size": sample_size,
                "outer_folds": 5,
                "model": "Ridge",
                "fold": fold,
                "rmse": np.sqrt(mean_squared_error(y_test, prediction)),
                "mae": mean_absolute_error(y_test, prediction),
                "r2": r2_score(y_test, prediction),
                "best_params": search.best_params_,
            }
            fold_results = pd.concat(
                [fold_results, pd.DataFrame([fold_result])],
                ignore_index=True,
            )
            fold_results.to_csv(FOLD_RESULTS_PATH, index=False)
            print(f"n={sample_size}, {specification}, Ridge: completed fold {fold}/5")

# %%
summary = (
    fold_results.groupby(["specification", "sample_size", "outer_folds", "model"])[
        ["rmse", "mae", "r2"]
    ]
    .agg(["mean", "std"])
    .reset_index()
)

summary_export = summary.copy()
summary_export.columns = [
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
summary_export.to_csv("../Results/ames_sample_size_summary.csv", index=False)

summary_export["result_source"] = "5-fold sample-size experiment"
model_order = ["OLS", "Ridge", "Random Forest", "XGBoost", "TabPFN"]
summary_export["model"] = pd.Categorical(
    summary_export["model"],
    categories=model_order,
    ordered=True,
)
combined_summary = summary_export.sort_values(
    ["specification", "sample_size", "model"]
)
combined_summary["model"] = combined_summary["model"].astype(str)
combined_summary.to_csv("../Results/ames_sample_size_combined_summary.csv", index=False)

display(combined_summary)

