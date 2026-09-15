
# This file:
# 1. Loads the DRN project data.
# 2. Splits the sample chronologically into T1 and T2.
# 3. Tunes Random Forest hyperparameters using T1 -> T2.
# 4. Compares the selected RF against OLS on T2.
# 5. Saves the tuning and validation results.


import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.data import load_panel, PREDICTOR_COLS


# ============================================================
# 1. OUTPUT PATHS
# ============================================================

OUTPUT_DIR = Path("exploratory/bali_replication_outputs")

PARAMS_PATH = OUTPUT_DIR / "best_rf_params.json"
GRID_RESULTS_PATH = OUTPUT_DIR / "rf_tuning_results.csv"
VALIDATION_RESULTS_PATH = OUTPUT_DIR / "validation_results.csv"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. LOAD DATA
# ============================================================
# The panel is already aligned so predictors dated t are paired
# with the realized one-month-ahead return R_{t+1}.
# ============================================================

df = load_panel(
    cols=["cusip", *PREDICTOR_COLS],
    targets=("retxrf",),
)

target = "retxrf_realized_return"


# ============================================================
# 3. CHRONOLOGICAL T1 / T2 SPLIT
# ============================================================
#
# T1 = first 36 months
#      model estimation
#
# T2 = next 24 months
#      RF hyperparameter validation
#
# Everything after T2 is reserved for the true OOS exercise
# in bali_oos.py.
# ============================================================

months = sorted(
    df["date"].unique()
)

train_months = months[:36]
validation_months = months[36:60]

train_mask = df["date"].isin(
    train_months
)

validation_mask = df["date"].isin(
    validation_months
)


# ============================================================
# 4. PREPARE T1 AND T2 DATA
# ============================================================

X_train = df.loc[
    train_mask,
    PREDICTOR_COLS,
]

y_train = df.loc[
    train_mask,
    target,
]

X_validation = df.loc[
    validation_mask,
    PREDICTOR_COLS,
]

y_validation = df.loc[
    validation_mask,
    target,
]


# ============================================================
# 5. ZERO-BENCHMARK R-SQUARED
# ============================================================
#
# R² = 1 - sum((y - y_hat)^2) / sum(y^2)
#
# The benchmark forecast is zero.
# ============================================================

def r2_oos(y_true, y_pred):

    numerator = (
        (y_true - y_pred) ** 2
    ).sum()

    denominator = (
        y_true ** 2
    ).sum()

    return 1 - numerator / denominator


# ============================================================
# 6. RANDOM FOREST HYPERPARAMETER TUNING
# ============================================================
#
# Small prespecified grid used for this targeted validation.
# This is not claimed to be Bali et al.'s exact tuning grid.
#
# Selected hyperparameters from the completed run:
#
# max_depth = 4
# min_samples_leaf = 50
#
# n_estimators and max_features remain fixed throughout:
#
# n_estimators = 200
# max_features = "sqrt"
# ============================================================

candidate_depths = [
    4,
    8,
    12,
]

candidate_leaf_sizes = [
    50,
    200,
]

best_r2 = float("-inf")
best_params = None

grid_results = []


for depth in candidate_depths:

    for leaf_size in candidate_leaf_sizes:

        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=depth,
            min_samples_leaf=leaf_size,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )

        rf.fit(
            X_train,
            y_train,
        )

        validation_predictions = rf.predict(
            X_validation
        )

        validation_r2 = r2_oos(
            y_validation.to_numpy(),
            validation_predictions,
        )

        grid_results.append(
            {
                "max_depth": depth,
                "min_samples_leaf": leaf_size,
                "validation_r2": validation_r2,
            }
        )

        if validation_r2 > best_r2:

            best_r2 = validation_r2

            best_params = {
                "max_depth": depth,
                "min_samples_leaf": leaf_size,
            }


# ============================================================
# 7. SAVE RF TUNING RESULTS
# ============================================================

grid_results_df = pd.DataFrame(
    grid_results
)

grid_results_df.to_csv(
    GRID_RESULTS_PATH,
    index=False,
)


with open(
    PARAMS_PATH,
    "w",
) as file:

    json.dump(
        best_params,
        file,
        indent=4,
    )


# ============================================================
# 8. OLS VALIDATION BENCHMARK
# ============================================================

ols = LinearRegression()

ols.fit(
    X_train,
    y_train,
)

ols_validation_predictions = ols.predict(
    X_validation
)

ols_validation_r2 = r2_oos(
    y_validation.to_numpy(),
    ols_validation_predictions,
)


# ============================================================
# 9. SAVE VALIDATION RESULTS
# ============================================================

validation_results = pd.DataFrame(
    {
        "model": [
            "OLS",
            "Random Forest",
        ],
        "validation_r2": [
            ols_validation_r2,
            best_r2,
        ],
    }
)

validation_results.to_csv(
    VALIDATION_RESULTS_PATH,
    index=False,
)


# ============================================================
# 10. DISPLAY VALIDATION RESULTS
# ============================================================

print("\nVALIDATION RESULTS")
print("------------------")

print(
    "Best RF parameters:",
    best_params,
)

print(
    "OLS validation R2:",
    ols_validation_r2,
)

print(
    "RF validation R2:",
    best_r2,
)

print(
    "RF minus OLS:",
    best_r2 - ols_validation_r2,
)