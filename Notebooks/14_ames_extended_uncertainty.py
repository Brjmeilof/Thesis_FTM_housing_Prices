# %% [markdown]
# # Ames Housing extended specification: predictive uncertainty
#
# This secondary extension compares 90% prediction intervals for OLS, XGBoost
# quantile regression, and TabPFN on the Ames Housing extended specification.
# The target remains log(SalePrice), matching the main thesis experiments.

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

RANDOM_STATE = 2
QUANTILES = [0.05, 0.50, 0.95]
RESULTS_DIR = Path("../Results/uncertainty_ames_extended")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = RESULTS_DIR / "ames_extended_uncertainty_predictions.csv"
SUMMARY_PATH = RESULTS_DIR / "ames_extended_uncertainty_summary.csv"
COVERAGE_WIDTH_FIGURE_PATH = RESULTS_DIR / "ames_extended_uncertainty_coverage_width.png"
EXAMPLE_INTERVAL_FIGURE_PATH = RESULTS_DIR / "ames_extended_uncertainty_example_intervals.png"


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

X = df[extended_features]
y = df["log_SalePrice"]

# A simple holdout design keeps this extension compact and easy to write up:
# 60% training, 20% validation for XGBoost tuning, and 20% final test.
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=RANDOM_STATE,
)


# %%
def sort_quantiles(q05, q50, q95):
    """Remove occasional quantile crossing by sorting per observation."""
    stacked = np.vstack([q05, q50, q95]).T
    sorted_quantiles = np.sort(stacked, axis=1)
    crossings = np.any(np.diff(stacked, axis=1) < 0, axis=1).sum()
    return sorted_quantiles[:, 0], sorted_quantiles[:, 1], sorted_quantiles[:, 2], int(crossings)


def pinball(y_true, y_pred, alpha):
    return mean_pinball_loss(y_true, y_pred, alpha=alpha)


def summarize_predictions(predictions):
    y_true = predictions["y_true"].to_numpy()
    q05 = predictions["q05"].to_numpy()
    q50 = predictions["q50"].to_numpy()
    q95 = predictions["q95"].to_numpy()

    return {
        "coverage_90": np.mean((y_true >= q05) & (y_true <= q95)),
        "avg_interval_width": np.mean(q95 - q05),
        "pinball_05": pinball(y_true, q05, 0.05),
        "pinball_50": pinball(y_true, q50, 0.50),
        "pinball_95": pinball(y_true, q95, 0.95),
    }


# %%
def fit_predict_ols():
    # Classical OLS prediction intervals are computed on the transformed design
    # matrix. The interval is for a new observation, so it includes residual
    # uncertainty as well as uncertainty in the estimated mean.
    preprocessor = extended_preprocessor.fit(X_train_val)
    X_train_design = preprocessor.transform(X_train_val)
    X_test_design = preprocessor.transform(X_test)

    X_train_design = sm.add_constant(X_train_design, has_constant="add")
    X_test_design = sm.add_constant(X_test_design, has_constant="add")

    model = sm.OLS(y_train_val.to_numpy(), X_train_design).fit()
    prediction_frame = model.get_prediction(X_test_design).summary_frame(alpha=0.10)

    q05 = prediction_frame["obs_ci_lower"].to_numpy()
    q50 = prediction_frame["mean"].to_numpy()
    q95 = prediction_frame["obs_ci_upper"].to_numpy()

    return pd.DataFrame(
        {
            "model": "OLS",
            "y_true": y_test.to_numpy(),
            "q05": q05,
            "q50": q50,
            "q95": q95,
            "quantile_crossings_fixed": 0,
        },
        index=y_test.index,
    )


# %%
def fit_xgboost_quantile(alpha, params, X_fit, y_fit):
    model = Pipeline(
        [
            ("preprocessor", extended_preprocessor),
            (
                "model",
                XGBRegressor(
                    objective="reg:quantileerror",
                    quantile_alpha=alpha,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    **params,
                ),
            ),
        ]
    )
    model.fit(X_fit, y_fit)
    return model


def select_xgboost_params():
    # Native XGBoost quantile loss is available in the installed XGBoost 3.2.0.
    # We tune one shared parameter set by minimizing average validation pinball
    # loss across the three requested quantiles.
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
            model = fit_xgboost_quantile(alpha, params, X_train, y_train)
            validation_prediction = model.predict(X_val)
            losses.append(pinball(y_val, validation_prediction, alpha))
        mean_loss = float(np.mean(losses))
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_params = params

    return best_params, best_loss


def fit_predict_xgboost():
    best_params, validation_pinball = select_xgboost_params()

    quantile_predictions = []
    for alpha in QUANTILES:
        model = fit_xgboost_quantile(alpha, best_params, X_train_val, y_train_val)
        quantile_predictions.append(model.predict(X_test))

    q05, q50, q95, crossings = sort_quantiles(*quantile_predictions)
    predictions = pd.DataFrame(
        {
            "model": "XGBoost Quantile",
            "y_true": y_test.to_numpy(),
            "q05": q05,
            "q50": q50,
            "q95": q95,
            "quantile_crossings_fixed": crossings,
            "xgb_validation_pinball_mean": validation_pinball,
            "xgb_best_params": str(best_params),
        },
        index=y_test.index,
    )
    return predictions


# %%
def fit_predict_tabpfn():
    # TabPFNRegressor exposes predictive quantiles directly in this installed
    # version. No ad hoc approximation is used here.
    model = Pipeline(
        [
            ("preprocessor", extended_preprocessor),
            (
                "model",
                TabPFNRegressor(
                    ignore_pretraining_limits=True,
                    random_state=RANDOM_STATE,
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
            "model": "TabPFN",
            "y_true": y_test.to_numpy(),
            "q05": q05,
            "q50": q50,
            "q95": q95,
            "quantile_crossings_fixed": crossings,
        },
        index=y_test.index,
    )


# %%
prediction_frames = [
    fit_predict_ols(),
    fit_predict_xgboost(),
    fit_predict_tabpfn(),
]

predictions = pd.concat(prediction_frames).reset_index(names="ames_index")
predictions.to_csv(PREDICTIONS_PATH, index=False)

summary_rows = []
for model_name, model_predictions in predictions.groupby("model", sort=False):
    row = {"model": model_name}
    row.update(summarize_predictions(model_predictions))
    row["pinball_mean"] = np.mean(
        [row["pinball_05"], row["pinball_50"], row["pinball_95"]]
    )
    row["quantile_crossings_fixed"] = int(
        model_predictions["quantile_crossings_fixed"].max()
    )
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary = summary[
    [
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
summary.to_csv(SUMMARY_PATH, index=False)


# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
axes[0].bar(summary["model"], summary["coverage_90"], color="#4C78A8")
axes[0].axhline(0.90, color="#D62728", linestyle="--", linewidth=1.5, label="Target")
axes[0].set_ylabel("90% interval coverage")
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis="x", rotation=20)
axes[0].legend(frameon=False)

axes[1].bar(summary["model"], summary["avg_interval_width"], color="#F58518")
axes[1].set_ylabel("Average interval width")
axes[1].tick_params(axis="x", rotation=20)

fig.tight_layout()
fig.savefig(COVERAGE_WIDTH_FIGURE_PATH, bbox_inches="tight")
plt.close(fig)


# %%
tabpfn_examples = (
    predictions[predictions["model"].eq("TabPFN")]
    .assign(abs_error=lambda data: (data["y_true"] - data["q50"]).abs())
    .sort_values("abs_error")
    .head(20)["ames_index"]
)
example_predictions = predictions[predictions["ames_index"].isin(tabpfn_examples)].copy()
example_predictions["example_id"] = pd.Categorical(
    example_predictions["ames_index"],
    categories=list(tabpfn_examples),
    ordered=True,
)
example_predictions = example_predictions.sort_values(["example_id", "model"])

fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
model_offsets = {"OLS": -0.22, "XGBoost Quantile": 0.0, "TabPFN": 0.22}
colors = {"OLS": "#4C78A8", "XGBoost Quantile": "#54A24B", "TabPFN": "#E45756"}
for model_name, group in example_predictions.groupby("model", sort=False):
    x_positions = np.arange(len(tabpfn_examples)) + model_offsets[model_name]
    ax.errorbar(
        x_positions,
        group["q50"],
        yerr=[
            group["q50"] - group["q05"],
            group["q95"] - group["q50"],
        ],
        fmt="o",
        capsize=3,
        markersize=4,
        linewidth=1.3,
        label=model_name,
        color=colors[model_name],
    )

y_true = (
    predictions[predictions["model"].eq("TabPFN")]
    .set_index("ames_index")
    .loc[list(tabpfn_examples), "y_true"]
    .to_numpy()
)
ax.scatter(np.arange(len(tabpfn_examples)), y_true, color="black", marker="x", label="Observed")
ax.set_xlabel("Example test observations")
ax.set_ylabel("log(SalePrice)")
ax.set_xticks(np.arange(len(tabpfn_examples)))
ax.set_xticklabels(range(1, len(tabpfn_examples) + 1))
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(EXAMPLE_INTERVAL_FIGURE_PATH, bbox_inches="tight")
plt.close(fig)


# %%
print("Ames extended uncertainty comparison complete.")
print(f"Train observations: {len(X_train):,}")
print(f"Validation observations: {len(X_val):,}")
print(f"Test observations: {len(X_test):,}")
print(f"Saved predictions: {PREDICTIONS_PATH}")
print(f"Saved summary: {SUMMARY_PATH}")
print(f"Saved figures: {COVERAGE_WIDTH_FIGURE_PATH}, {EXAMPLE_INTERVAL_FIGURE_PATH}")
print(summary.to_string(index=False))
