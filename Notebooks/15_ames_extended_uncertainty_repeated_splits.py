# %% [markdown]
# # Ames Housing extended specification: repeated-split predictive uncertainty
#
# Repeats the Ames extended uncertainty comparison over five fixed
# train/validation/test splits. The target is log(SalePrice), matching the main
# thesis experiments and the single-split uncertainty extension.

# %%
from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "thesis_matplotlib"))
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from tabpfn import TabPFNRegressor
from xgboost import XGBRegressor

SEEDS = [42, 123, 456, 789, 2025]
QUANTILES = [0.05, 0.50, 0.95]
RESULTS_DIR = Path("../Results/uncertainty_ames_extended_repeated_splits")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = RESULTS_DIR / "ames_extended_uncertainty_repeated_predictions.csv"
SPLIT_SUMMARY_PATH = RESULTS_DIR / "ames_extended_uncertainty_split_summary.csv"
AGGREGATED_SUMMARY_PATH = RESULTS_DIR / "ames_extended_uncertainty_aggregated_summary.csv"
FIGURE_PATH = RESULTS_DIR / "ames_extended_uncertainty_repeated_coverage_width.png"


# %%
df = pd.read_csv("../Data/AmesHousing.csv").dropna(subset=["SalePrice"]).copy()
df["log_SalePrice"] = np.log(df["SalePrice"])

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


def make_extended_preprocessor():
    return ColumnTransformer(
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


X = df[extended_features]
y = df["log_SalePrice"]


# %%
def make_split(seed):
    # 60% training, 20% validation, and 20% test. The same split is shared by
    # all three models within each repetition.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=seed,
    )
    return X_train, X_val, X_train_val, X_test, y_train, y_val, y_train_val, y_test


def sort_quantiles(q05, q50, q95):
    """Remove occasional quantile crossing by sorting per observation."""
    stacked = np.vstack([q05, q50, q95]).T
    sorted_quantiles = np.sort(stacked, axis=1)
    crossings = np.any(np.diff(stacked, axis=1) < 0, axis=1).sum()
    return sorted_quantiles[:, 0], sorted_quantiles[:, 1], sorted_quantiles[:, 2], int(crossings)


def pinball(y_true, y_pred, alpha):
    return mean_pinball_loss(y_true, y_pred, alpha=alpha)


def summarize_predictions(seed, predictions):
    y_true = predictions["y_true"].to_numpy()
    q05 = predictions["q05"].to_numpy()
    q50 = predictions["q50"].to_numpy()
    q95 = predictions["q95"].to_numpy()
    pinball_05 = pinball(y_true, q05, 0.05)
    pinball_50 = pinball(y_true, q50, 0.50)
    pinball_95 = pinball(y_true, q95, 0.95)

    return {
        "seed": seed,
        "model": predictions["model"].iloc[0],
        "coverage_90": np.mean((y_true >= q05) & (y_true <= q95)),
        "avg_interval_width": np.mean(q95 - q05),
        "pinball_05": pinball_05,
        "pinball_50": pinball_50,
        "pinball_95": pinball_95,
        "pinball_mean": np.mean([pinball_05, pinball_50, pinball_95]),
        "quantile_crossings_fixed": int(predictions["quantile_crossings_fixed"].max()),
    }


# %%
def fit_predict_ols(seed, X_train_val, X_test, y_train_val, y_test):
    # Classical OLS prediction intervals include residual uncertainty and
    # uncertainty in the estimated conditional mean.
    preprocessor = make_extended_preprocessor().fit(X_train_val)
    X_train_design = preprocessor.transform(X_train_val)
    X_test_design = preprocessor.transform(X_test)

    X_train_design = sm.add_constant(X_train_design, has_constant="add")
    X_test_design = sm.add_constant(X_test_design, has_constant="add")

    model = sm.OLS(y_train_val.to_numpy(), X_train_design).fit()
    prediction_frame = model.get_prediction(X_test_design).summary_frame(alpha=0.10)

    return pd.DataFrame(
        {
            "seed": seed,
            "ames_index": y_test.index,
            "model": "OLS",
            "y_true": y_test.to_numpy(),
            "q05": prediction_frame["obs_ci_lower"].to_numpy(),
            "q50": prediction_frame["mean"].to_numpy(),
            "q95": prediction_frame["obs_ci_upper"].to_numpy(),
            "quantile_crossings_fixed": 0,
            "xgb_validation_pinball_mean": np.nan,
            "xgb_best_params": "",
        }
    )


# %%
def fit_xgboost_quantile(alpha, params, seed, X_fit, y_fit):
    model = Pipeline(
        [
            ("preprocessor", make_extended_preprocessor()),
            (
                "model",
                XGBRegressor(
                    objective="reg:quantileerror",
                    quantile_alpha=alpha,
                    random_state=seed,
                    n_jobs=1,
                    **params,
                ),
            ),
        ]
    )
    model.fit(X_fit, y_fit)
    return model


def select_xgboost_params(seed, X_train, X_val, y_train, y_val):
    # Native XGBoost quantile loss is available in the installed environment.
    # One shared parameter set is selected by average validation pinball loss.
    param_grid = [
        {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
        }
        for n_estimators in [100, 300]
        for learning_rate in [0.05, 0.1]
        for max_depth in [2, 3]
        for subsample in [0.8, 1.0]
        for colsample_bytree in [0.8, 1.0]
    ]

    best_params = None
    best_loss = np.inf
    for params in param_grid:
        losses = []
        for alpha in QUANTILES:
            model = fit_xgboost_quantile(alpha, params, seed, X_train, y_train)
            validation_prediction = model.predict(X_val)
            losses.append(pinball(y_val, validation_prediction, alpha))
        mean_loss = float(np.mean(losses))
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_params = params

    return best_params, best_loss


def fit_predict_xgboost(seed, X_train, X_val, X_train_val, X_test, y_train, y_val, y_train_val, y_test):
    best_params, validation_pinball = select_xgboost_params(
        seed,
        X_train,
        X_val,
        y_train,
        y_val,
    )

    quantile_predictions = []
    for alpha in QUANTILES:
        model = fit_xgboost_quantile(alpha, best_params, seed, X_train_val, y_train_val)
        quantile_predictions.append(model.predict(X_test))

    q05, q50, q95, crossings = sort_quantiles(*quantile_predictions)
    return pd.DataFrame(
        {
            "seed": seed,
            "ames_index": y_test.index,
            "model": "XGBoost Quantile",
            "y_true": y_test.to_numpy(),
            "q05": q05,
            "q50": q50,
            "q95": q95,
            "quantile_crossings_fixed": crossings,
            "xgb_validation_pinball_mean": validation_pinball,
            "xgb_best_params": str(best_params),
        }
    )


# %%
def fit_predict_tabpfn(seed, X_train_val, X_test, y_train_val, y_test):
    # TabPFN exposes predictive quantiles directly; no ad hoc approximation is
    # used. The split seed is also used as the model random state.
    model = Pipeline(
        [
            ("preprocessor", make_extended_preprocessor()),
            (
                "model",
                TabPFNRegressor(
                    ignore_pretraining_limits=True,
                    random_state=seed,
                    show_progress_bar=False,
                ),
            ),
        ]
    )
    model.fit(X_train_val, y_train_val)
    quantile_output = model.predict(
        X_test,
        output_type="quantiles",
        quantiles=QUANTILES,
    )

    q05, q50, q95 = [np.asarray(values) for values in quantile_output]
    q05, q50, q95, crossings = sort_quantiles(q05, q50, q95)
    return pd.DataFrame(
        {
            "seed": seed,
            "ames_index": y_test.index,
            "model": "TabPFN",
            "y_true": y_test.to_numpy(),
            "q05": q05,
            "q50": q50,
            "q95": q95,
            "quantile_crossings_fixed": crossings,
            "xgb_validation_pinball_mean": np.nan,
            "xgb_best_params": "",
        }
    )


# %%
prediction_frames = []
summary_rows = []

for seed in SEEDS:
    print(f"Running seed {seed}...")
    (
        X_train,
        X_val,
        X_train_val,
        X_test,
        y_train,
        y_val,
        y_train_val,
        y_test,
    ) = make_split(seed)

    seed_predictions = [
        fit_predict_ols(seed, X_train_val, X_test, y_train_val, y_test),
        fit_predict_xgboost(
            seed,
            X_train,
            X_val,
            X_train_val,
            X_test,
            y_train,
            y_val,
            y_train_val,
            y_test,
        ),
        fit_predict_tabpfn(seed, X_train_val, X_test, y_train_val, y_test),
    ]

    for model_predictions in seed_predictions:
        prediction_frames.append(model_predictions)
        summary_rows.append(summarize_predictions(seed, model_predictions))

predictions = pd.concat(prediction_frames, ignore_index=True)
predictions.to_csv(PREDICTIONS_PATH, index=False)

split_summary = pd.DataFrame(summary_rows)
split_summary = split_summary[
    [
        "seed",
        "model",
        "coverage_90",
        "avg_interval_width",
        "pinball_05",
        "pinball_50",
        "pinball_95",
        "pinball_mean",
        "quantile_crossings_fixed",
    ]
]
split_summary.to_csv(SPLIT_SUMMARY_PATH, index=False)


# %%
metrics = [
    "coverage_90",
    "avg_interval_width",
    "pinball_05",
    "pinball_50",
    "pinball_95",
    "pinball_mean",
]
aggregated_summary = (
    split_summary.groupby("model")[metrics]
    .agg(["mean", "std"])
    .reset_index()
)
aggregated_summary.columns = [
    "model",
    *[
        f"{metric}_{stat}"
        for metric in metrics
        for stat in ["mean", "sd"]
    ],
]
aggregated_summary = aggregated_summary.sort_values("pinball_mean_mean")
aggregated_summary.to_csv(AGGREGATED_SUMMARY_PATH, index=False)


# %%
model_order = ["OLS", "XGBoost Quantile", "TabPFN"]
plot_data = aggregated_summary.set_index("model").loc[model_order].reset_index()
x = np.arange(len(model_order))

fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
axes[0].errorbar(
    x,
    plot_data["coverage_90_mean"],
    yerr=plot_data["coverage_90_sd"],
    fmt="o",
    capsize=4,
    color="#4C78A8",
)
axes[0].axhline(0.90, color="#D62728", linestyle="--", linewidth=1.5, label="Target")
axes[0].set_xticks(x)
axes[0].set_xticklabels(model_order, rotation=20)
axes[0].set_ylabel("Mean 90% coverage")
axes[0].set_ylim(0, 1)
axes[0].legend(frameon=False)

axes[1].errorbar(
    x,
    plot_data["avg_interval_width_mean"],
    yerr=plot_data["avg_interval_width_sd"],
    fmt="o",
    capsize=4,
    color="#F58518",
)
axes[1].set_xticks(x)
axes[1].set_xticklabels(model_order, rotation=20)
axes[1].set_ylabel("Mean interval width")

fig.tight_layout()
fig.savefig(FIGURE_PATH, bbox_inches="tight")
plt.close(fig)


# %%
best_pinball_model = aggregated_summary.iloc[0]["model"]
closest_coverage_model = (
    aggregated_summary.assign(
        coverage_distance=lambda data: (data["coverage_90_mean"] - 0.90).abs()
    )
    .sort_values("coverage_distance")
    .iloc[0]["model"]
)

print("Repeated-split Ames extended uncertainty comparison complete.")
print(f"Saved split summary: {SPLIT_SUMMARY_PATH}")
print(f"Saved aggregated summary: {AGGREGATED_SUMMARY_PATH}")
print(f"Saved predictions: {PREDICTIONS_PATH}")
print(f"Saved figure: {FIGURE_PATH}")
print("\nAggregated summary:")
print(aggregated_summary.to_string(index=False))
print("\nBrief interpretation:")
print(
    f"{best_pinball_model} has the lowest average pinball loss across the five "
    f"splits, while {closest_coverage_model} is closest to the nominal 90% "
    "coverage target on average. Compare coverage together with interval width: "
    "higher coverage is easier to obtain with wider intervals."
)
print(
    "TabPFN remains best on the overall pinball metric after averaging across "
    "splits."
    if best_pinball_model == "TabPFN"
    else "TabPFN does not remain best on the overall pinball metric after "
    "averaging across splits."
)
