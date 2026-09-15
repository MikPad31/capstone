import gc
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
MONTHLY_DIR = OUTPUT_DIR / "monthly_oos"

PARAMS_PATH = OUTPUT_DIR / "best_rf_params.json"
COMBINED_OUTPUT_PATH = OUTPUT_DIR / "bali_oos_predictions.parquet"
FINAL_RESULTS_PATH = OUTPUT_DIR / "bali_final_results.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MONTHLY_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD SELECTED RANDOM FOREST HYPERPARAMETERS
# ============================================================
# Hyperparameters were selected using T1 -> T2 in
# bali_replication.py and remain fixed throughout T3.
#
# Selected values:
# max_depth = 4
# min_samples_leaf = 50
# ============================================================

if not PARAMS_PATH.exists():
    raise FileNotFoundError(
        "best_rf_params.json does not exist. "
        "Run bali_replication.py first."
    )

with open(PARAMS_PATH, "r") as file:
    best_params = json.load(file)


# ============================================================
# 3. LOAD DATA
# ============================================================
# The panel is already aligned so predictors dated t are paired
# with the realized one-month-ahead return R_{t+1}.
# No additional target shifting is required.
# ============================================================

df = load_panel(
    cols=["cusip", *PREDICTOR_COLS],
    targets=("retxrf",),
)

target = "retxrf_realized_return"


# ============================================================
# 4. DEFINE TRUE OOS PERIOD
# ============================================================
# T1 = first 36 months
# T2 = next 24 months
# T3 = all remaining months
# ============================================================

months = sorted(df["date"].unique())
oos_months = months[60:]


# ============================================================
# 5. MONTHLY EXPANDING-WINDOW OOS FORECASTS
# ============================================================
# For each OOS signal month t:
#
# 1. Train on all observations dated before t.
# 2. Fit OLS and Random Forest.
# 3. Use X_t to generate a forecast of R_{t+1}.
# 4. Save that month's predictions immediately.
#
# Example:
#
# Signal month: Jul 2007
# Historical signal dates: Jul 2002 -> Jun 2007
# Forecast target: next-month realized return
#
# Each month is checkpointed to disk. Existing monthly files
# are skipped automatically if the script is restarted.
# ============================================================

for i, current_month in enumerate(oos_months):

    current_month = pd.Timestamp(current_month)

    month_file = (
        MONTHLY_DIR
        / f"{current_month:%Y-%m}.parquet"
    )

    if month_file.exists():
        continue

    print(
        f"[{i + 1}/{len(oos_months)}] "
        f"Predicting signal month {current_month:%Y-%m}"
    )

    historical_mask = df["date"] < current_month
    test_mask = df["date"] == current_month

    X_historical = df.loc[
        historical_mask,
        PREDICTOR_COLS,
    ]

    y_historical = df.loc[
        historical_mask,
        target,
    ]

    X_test = df.loc[
        test_mask,
        PREDICTOR_COLS,
    ]

    y_test = df.loc[
        test_mask,
        target,
    ]


    # OLS
    ols = LinearRegression()

    ols.fit(
        X_historical,
        y_historical,
    )

    ols_predictions = ols.predict(X_test)


    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(
        X_historical,
        y_historical,
    )

    rf_predictions = rf.predict(X_test)


    # Store predictions
    month_results = df.loc[
        test_mask,
        ["date", "cusip"],
    ].copy()

    month_results["actual_return"] = y_test.to_numpy()
    month_results["ols_prediction"] = ols_predictions
    month_results["rf_prediction"] = rf_predictions


    # Save monthly checkpoint safely
    temp_file = (
        MONTHLY_DIR
        / f"{current_month:%Y-%m}.tmp.parquet"
    )

    month_results.to_parquet(
        temp_file,
        index=False,
    )

    temp_file.replace(month_file)


    # Free memory before moving to the next month
    del X_historical
    del y_historical
    del X_test
    del y_test
    del ols
    del rf
    del ols_predictions
    del rf_predictions
    del month_results

    gc.collect()


# ============================================================
# 6. VERIFY ALL OOS MONTHS ARE AVAILABLE
# ============================================================

expected_month_files = [
    MONTHLY_DIR
    / f"{pd.Timestamp(month):%Y-%m}.parquet"
    for month in oos_months
]

missing_files = [
    file
    for file in expected_month_files
    if not file.exists()
]

if missing_files:
    raise RuntimeError(
        "OOS forecasting is incomplete. "
        f"{len(missing_files)} monthly files are missing."
    )


# ============================================================
# 7. COMBINE SAVED OOS PREDICTIONS
# ============================================================

monthly_results = [
    pd.read_parquet(month_file)
    for month_file in expected_month_files
]

oos_results = pd.concat(
    monthly_results,
    ignore_index=True,
)


# Verify final OOS sample size
expected_oos_observations = (
    df["date"].isin(oos_months)
).sum()

if len(oos_results) != expected_oos_observations:
    raise RuntimeError(
        "Combined OOS observation count does not "
        "match the expected sample size."
    )


# Save combined predictions safely
combined_temp_path = (
    OUTPUT_DIR
    / "bali_oos_predictions.tmp.parquet"
)

oos_results.to_parquet(
    combined_temp_path,
    index=False,
)

combined_temp_path.replace(
    COMBINED_OUTPUT_PATH
)


# ============================================================
# 8. BALI-STYLE OOS R-SQUARED
# ============================================================
# Zero-return benchmark:
#
# R²_OOS = 1 - sum((y - y_hat)^2) / sum(y^2)
# ============================================================

def r2_oos(y_true, y_pred):
    numerator = ((y_true - y_pred) ** 2).sum()
    denominator = (y_true ** 2).sum()

    return 1 - numerator / denominator


# ============================================================
# 9. FINAL POOLED OOS RESULTS
# ============================================================
# Forecasts are generated month by month, but the reported
# statistic is one pooled OOS R² for each model across all
# T3 bond-month observations.
# ============================================================

actual_returns = (
    oos_results["actual_return"].to_numpy()
)

ols_oos_r2 = r2_oos(
    actual_returns,
    oos_results["ols_prediction"].to_numpy(),
)

rf_oos_r2 = r2_oos(
    actual_returns,
    oos_results["rf_prediction"].to_numpy(),
)

rf_minus_ols = (
    rf_oos_r2 - ols_oos_r2
)


# ============================================================
# 10. SAVE FINAL RESULTS
# ============================================================

final_results = pd.DataFrame(
    {
        "metric": [
            "OLS OOS R2",
            "Random Forest OOS R2",
            "RF minus OLS",
        ],
        "value": [
            ols_oos_r2,
            rf_oos_r2,
            rf_minus_ols,
        ],
    }
)

final_results.to_csv(
    FINAL_RESULTS_PATH,
    index=False,
)


# ============================================================
# 11. DISPLAY FINAL RESULTS
# ============================================================

print("\nFINAL OOS RESULTS")
print("-----------------")
print("OLS OOS R2:", ols_oos_r2)
print("Random Forest OOS R2:", rf_oos_r2)
print("RF minus OLS:", rf_minus_ols)

if rf_oos_r2 > ols_oos_r2:
    print("Bali qualitative RF > OLS finding recovered: YES")
else:
    print("Bali qualitative RF > OLS finding recovered: NO")