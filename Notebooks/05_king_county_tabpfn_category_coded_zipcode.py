# %% [markdown]
# # King County TabPFN with category-coded zipcode
#
# This experiment evaluates TabPFN on the full King County dataset using the
# extended feature specification and 5-fold cross-validation. Unlike the main
# King County experiment, `zipcode` is represented by one ordinal-encoded
# categorical column instead of approximately 70 one-hot columns.
#
# All preprocessing is fitted within each training fold. TabPFN is explicitly
# informed that the final transformed column is categorical.

# %%
import time

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from tabpfn import TabPFNRegressor

RANDOM_STATE = 2
RESULTS_PATH = "../Results/king_county_tabpfn_category_zipcode_fold_results.csv"

# %%
df = pd.read_csv("../Data/kc_house_data.csv").dropna(subset=["price"]).copy()
df["log_price"] = np.log(df["price"])
df["zipcode"] = df["zipcode"].astype(str)

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

numeric_features = [
    feature for feature in extended_features_king if feature != "zipcode"
]
categorical_features = ["zipcode"]

X = df[extended_features_king]
y = df["log_price"]

preprocessor = ColumnTransformer(
    [
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
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
            categorical_features,
        ),
    ],
    verbose_feature_names_out=False,
)

print(f"Observations: {len(df):,}")
print(f"Raw features: {len(extended_features_king)}")
print("Transformed features: 16")
print(f"Distinct zipcodes: {df['zipcode'].nunique()}")

# %% [markdown]
# ## Five-fold evaluation
#
# The transformed zipcode is the final column, so
# `categorical_features_indices=[15]` tells TabPFN to treat its integer codes as
# categories rather than as a continuous ordered measurement.

# %%
outer_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
if pd.io.common.file_exists(RESULTS_PATH):
    fold_results = pd.read_csv(RESULTS_PATH)
else:
    fold_results = pd.DataFrame()

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
    if not fold_results.empty and (fold_results["fold"] == fold).any():
        print(f"Skipping saved fold {fold}/5")
        continue

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                TabPFNRegressor(
                    categorical_features_indices=[15],
                    ignore_pretraining_limits=True,
                    random_state=RANDOM_STATE,
                    show_progress_bar=False,
                ),
            ),
        ]
    )

    started = time.perf_counter()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    elapsed_seconds = time.perf_counter() - started

    result = {
        "dataset": "King County",
        "specification": "Extended",
        "zipcode_representation": "Category-coded",
        "sample_size": len(df),
        "outer_folds": 5,
        "model": "TabPFN",
        "fold": fold,
        "rmse": np.sqrt(mean_squared_error(y_test, prediction)),
        "mae": mean_absolute_error(y_test, prediction),
        "r2": r2_score(y_test, prediction),
        "elapsed_seconds": elapsed_seconds,
    }
    fold_results = pd.concat([fold_results, pd.DataFrame([result])], ignore_index=True)
    fold_results.to_csv(RESULTS_PATH, index=False)
    print(f"Completed fold {fold}/5 in {elapsed_seconds / 60:.1f} minutes")

# %%
summary = (
    fold_results[["rmse", "mae", "r2", "elapsed_seconds"]]
    .agg(["mean", "std"])
    .transpose()
)
summary.to_csv("../Results/king_county_tabpfn_category_zipcode_summary.csv")

display(summary)
display(fold_results)

# %% [markdown]
# ## Verified results and comparison
#
# | Zipcode representation | Transformed features | Mean RMSE | Mean MAE | Mean R2 |
# |---|---:|---:|---:|---:|
# | One-hot encoded | approximately 85 | 0.155473 | 0.108372 | 0.912799 |
# | Category-coded | 16 | **0.154373** | **0.107448** | **0.914030** |
#
# The category-coded run averaged approximately 10.0 minutes per fold. The
# final two checkpointed folds of the earlier one-hot run averaged roughly
# 17 minutes per fold. Runtime comparisons are approximate because the earlier
# experiment did not record per-fold elapsed time directly.
