# %% [markdown]
# # Add OLS to King County and California sample-size comparisons
#
# This notebook computes only the missing OLS rows for the King County and
# California Housing sample-size experiments. Existing Ridge, Random Forest,
# XGBoost, and TabPFN rows are not recomputed.

# %%
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 2


def update_summary_files(fold_path, summary_path, combined_path, full_path=None, has_zipcode=False):
    fold_results = pd.read_csv(fold_path)
    group_cols = ["dataset", "specification", "sample_size", "outer_folds", "model"]
    if has_zipcode:
        group_cols.append("zipcode_representation")

    summary = (
        fold_results.groupby(group_cols)[["rmse", "mae", "r2"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_export = summary.copy()
    metric_cols = [
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "mae_std",
        "r2_mean",
        "r2_std",
    ]
    summary_export.columns = group_cols + metric_cols
    summary_export.to_csv(summary_path, index=False)

    if full_path:
        full_results = pd.read_csv(full_path)
        if has_zipcode:
            needed_cols = group_cols + metric_cols
        else:
            needed_cols = group_cols + metric_cols
        full_results = full_results[needed_cols]
        full_sizes = set(full_results["sample_size"].unique())
        full_ols = summary_export[
            summary_export["sample_size"].isin(full_sizes)
            & summary_export["model"].eq("OLS")
        ]
        smaller_summary = summary_export[~summary_export["sample_size"].isin(full_sizes)]
        combined = pd.concat([smaller_summary, full_results, full_ols], ignore_index=True)
    else:
        combined = summary_export

    model_order = ["OLS", "Ridge", "Random Forest", "XGBoost", "TabPFN"]
    combined["model"] = pd.Categorical(
        combined["model"],
        categories=model_order,
        ordered=True,
    )
    combined = combined.sort_values(["specification", "sample_size", "model"])
    combined["model"] = combined["model"].astype(str)
    combined.to_csv(combined_path, index=False)
    return combined


def evaluate_ols_rows(df, target, specifications, sample_sizes, fold_path, dataset, full_size):
    rng = np.random.default_rng(RANDOM_STATE)
    sample_order = rng.permutation(df.index.to_numpy())
    sample_indices = {size: sample_order[:size] for size in sample_sizes if size != full_size}
    sample_indices[full_size] = df.index.to_numpy()

    outer_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_results = pd.read_csv(fold_path)

    def completed(sample_size, specification, fold):
        return (
            (fold_results["sample_size"] == sample_size)
            & (fold_results["specification"] == specification)
            & (fold_results["model"] == "OLS")
            & (fold_results["fold"] == fold)
        ).any()

    for sample_size in sample_sizes:
        sample = df.loc[sample_indices[sample_size]]

        for specification, features, preprocessor, zipcode_representation in specifications:
            X = sample[features]
            y = sample[target]

            for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
                if completed(sample_size, specification, fold):
                    print(f"{dataset}, n={sample_size}, {specification}, OLS: skipping fold {fold}/5")
                    continue

                model = Pipeline(
                    [
                        ("preprocessor", preprocessor),
                        ("model", LinearRegression()),
                    ]
                )

                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

                model.fit(X_train, y_train)
                prediction = model.predict(X_test)

                fold_result = {
                    "dataset": dataset,
                    "specification": specification,
                    "sample_size": sample_size,
                    "outer_folds": 5,
                    "model": "OLS",
                    "fold": fold,
                    "rmse": np.sqrt(mean_squared_error(y_test, prediction)),
                    "mae": mean_absolute_error(y_test, prediction),
                    "r2": r2_score(y_test, prediction),
                    "best_params": {},
                }
                if zipcode_representation is not None:
                    fold_result["zipcode_representation"] = zipcode_representation

                fold_results = pd.concat(
                    [fold_results, pd.DataFrame([fold_result])],
                    ignore_index=True,
                )
                fold_results.to_csv(fold_path, index=False)
                print(f"{dataset}, n={sample_size}, {specification}, OLS: completed fold {fold}/5")


# %% [markdown]
# ## King County OLS

# %%
king_df = pd.read_csv("../Data/kc_house_data.csv").dropna(subset=["price"]).copy()
king_df["log_price"] = np.log(king_df["price"])
king_df["zipcode"] = king_df["zipcode"].astype(str)

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
extended_numeric_features_king = [feature for feature in extended_features_king if feature != "zipcode"]

king_numeric_preprocessor = ColumnTransformer(
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

king_extended_preprocessor = ColumnTransformer(
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
            ["zipcode"],
        ),
    ],
    verbose_feature_names_out=False,
)

king_specs = [
    ("Parsimonious", parsimonious_features_king, king_numeric_preprocessor, "Not applicable"),
    ("Extended", extended_features_king, king_extended_preprocessor, "One-hot encoded"),
]

evaluate_ols_rows(
    king_df,
    "log_price",
    king_specs,
    [1000, 2500, 5000, len(king_df)],
    "../Results/king_county_sample_size_fold_results.csv",
    "King County",
    len(king_df),
)

king_combined = update_summary_files(
    "../Results/king_county_sample_size_fold_results.csv",
    "../Results/king_county_sample_size_summary.csv",
    "../Results/king_county_sample_size_combined_summary.csv",
    full_path="../Results/king_county_final_comparison.csv",
    has_zipcode=True,
)

display(king_combined)

# %% [markdown]
# ## California OLS

# %%
raw_california = pd.read_csv("../Data/California_Housing.csv")
california_df = pd.DataFrame(
    {
        "MedInc": raw_california["median_income"],
        "HouseAge": raw_california["housing_median_age"],
        "AveRooms": raw_california["total_rooms"] / raw_california["households"],
        "AveBedrms": raw_california["total_bedrooms"] / raw_california["households"],
        "Population": raw_california["population"],
        "AveOccup": raw_california["population"] / raw_california["households"],
        "Latitude": raw_california["latitude"],
        "Longitude": raw_california["longitude"],
        "MedHouseVal": raw_california["median_house_value"],
    }
)
california_df = california_df.replace([np.inf, -np.inf], np.nan).dropna(
    subset=["MedHouseVal"]
).copy()
california_df["log_MedHouseVal"] = np.log(california_df["MedHouseVal"])

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


def california_preprocessor(features):
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


california_specs = [
    (
        "Parsimonious",
        parsimonious_features_california,
        california_preprocessor(parsimonious_features_california),
        None,
    ),
    (
        "Extended",
        extended_features_california,
        california_preprocessor(extended_features_california),
        None,
    ),
]

evaluate_ols_rows(
    california_df,
    "log_MedHouseVal",
    california_specs,
    [1000, 2500, 5000, 10000, len(california_df)],
    "../Results/california_sample_size_fold_results.csv",
    "California Housing",
    len(california_df),
)

california_combined = update_summary_files(
    "../Results/california_sample_size_fold_results.csv",
    "../Results/california_sample_size_summary.csv",
    "../Results/california_sample_size_combined_summary.csv",
    full_path="../Results/california_summary.csv",
)

display(california_combined)
