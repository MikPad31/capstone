"""
Summary statistics for the data section of the report and slides.

    1. panel: bond-months, bonds, issuers, months; predictor missingness, range and exact zeros;
       share of bond-months in a multi-bond issuer-month
    2. targets: L0 / L1 / IC / L2 on the common sample at h=1/3/12 over the full panel (mean, SD,
       1st/99th percentiles), and how L1's sum of squares splits into IC and L2
    3. the monthly cross-sectional mean of L0 and L1 at h=1, for the time-series figure

IC and L2 are built with `harness.project`, the same transform the models are scored on, and
checked against the saved test-period labels in output/phase2/preds when those exist.
Writes output/data_summary/{panel.csv, targets.csv, monthly_h1.csv, summary.md}.

RUN: DSE4101_DATA_DIR=... python -m exploratory.data_summary
"""

# %% Setup
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR
from src.data import PREDICTOR_COLS, load_panel
from src.harness import IC, L0, L1, L2, build_labels, eligible_rows, label_col, project
from src.runlog import check, finish, log, report

OUT = OUTPUT_DIR / "data_summary"
PREDS_DIR = OUTPUT_DIR / "phase2" / "preds"
HORIZONS = (1, 3, 12)
OOS_START = pd.Period("2015-01", freq="M")  # as in phase2_grid.py; still provisional in src/config.py
OUT.mkdir(parents=True, exist_ok=True)
_start = time.time()

# %% 1. Panel and predictors
X = load_panel(cols=list(PREDICTOR_COLS), targets=())[list(PREDICTOR_COLS)].to_numpy(dtype=np.float32)
panel = {
    "n_predictors": X.shape[1],
    "predictor_nan_cells": int(np.isnan(X).sum()),
    "predictor_min": float(np.nanmin(X)),
    "predictor_max": float(np.nanmax(X)),
    "predictor_zero_share": float((X == 0).mean()),
}
del X

df = load_panel(cols=["date", "cusip", "gvkey"], targets=("retx", "retxrf"))
months = pd.PeriodIndex(pd.to_datetime(df["date"]), freq="M")
has_key = df["gvkey"].notna().to_numpy()
size = (df.loc[has_key, ["gvkey"]].assign(_m=months[has_key])
        .groupby(["gvkey", "_m"], observed=True)["gvkey"].transform("size"))
panel.update({
    "bond_months": len(df),
    "bonds": int(df["cusip"].nunique()),
    "issuers": int(df["gvkey"].nunique()),
    "first_month": str(months.min()),
    "last_month": str(months.max()),
    "n_months": int(months.nunique()),
    "rows_with_gvkey": int(has_key.sum()),
    "share_in_multibond_issuer_month": float((size >= 2).mean()),
    "median_bonds_per_issuer_month": float(df.loc[has_key, ["gvkey"]].assign(_m=months[has_key])
                                           .groupby(["gvkey", "_m"], observed=True).size().median()),
})
check(panel["predictor_nan_cells"] == 0, "predictors have no missing values")
check(panel["predictor_min"] >= -1 and panel["predictor_max"] <= 1, "predictors lie in [-1, 1]")
pd.Series(panel, name="value").to_csv(OUT / "panel.csv", index_label="stat")
report(f"{panel['bond_months']:,} bond-months, {panel['bonds']:,} bonds, {panel['issuers']:,} issuers, "
       f"{panel['n_months']} months ({panel['first_month']} to {panel['last_month']})")

# %% 2. Targets on the common sample
build_labels(df, HORIZONS)
rows, monthly = [], None
for h in HORIZONS:
    m = eligible_rows(df, L2, h, common_sample=True)
    frame = df.loc[m, ["date", "cusip", "gvkey"]]
    l1 = df.loc[m, label_col(L1, h)]
    y = {"L0": df.loc[m, label_col(L0, h)].astype("float64"),
         "L1": l1.astype("float64"),
         "IC": project(frame, l1, IC),
         "L2": project(frame, l1, L2)}
    gap = float((y["L1"] - y["IC"] - y["L2"]).abs().max())
    check(gap < 1e-12, f"h={h}: L1 = IC + L2 row by row (max gap {gap:.2e})")
    ss = {lv: float((v ** 2).sum()) for lv, v in y.items()}
    for lv, v in y.items():
        rows.append({"horizon": h, "level": lv, "n": len(v), "bonds": int(frame["cusip"].nunique()),
                     "issuers": int(frame["gvkey"].nunique()), "mean": v.mean(), "sd": v.std(),
                     "p01": v.quantile(0.01), "p99": v.quantile(0.99), "sum_sq": ss[lv],
                     "share_of_L1_sum_sq": ss[lv] / ss["L1"] if lv in ("IC", "L2") else np.nan})

    # the saved test-period labels must be exactly these
    mo = months[m]
    test = mo >= OOS_START
    for lv in ("IC", "L2"):
        path = PREDS_DIR / f"enet_{lv}_h{h}_all_common.parquet"
        if not path.exists():
            log(f"{path.name} not found; skipping the cross-check against saved labels")
            continue
        saved = pd.read_parquet(path, columns=["date", "cusip", "y"])
        saved["date"] = pd.PeriodIndex(pd.to_datetime(saved["date"]), freq="M")
        ours = pd.DataFrame({"date": mo[test], "cusip": frame["cusip"].astype(str).to_numpy()[test],
                             "ours": y[lv].to_numpy()[test]})
        j = saved.assign(cusip=saved["cusip"].astype(str)).merge(ours, on=["date", "cusip"], how="left")
        diff = float((j["y"] - j["ours"]).abs().max())
        check(j["ours"].notna().all() and diff < 1e-12,
              f"{lv} h={h}: labels match the {len(saved):,} saved test rows (max diff {diff:.2e})")

    if h == 1:
        monthly = (pd.DataFrame({"month": mo.astype(str), "L0": y["L0"].to_numpy(), "L1": y["L1"].to_numpy()})
                   .groupby("month").agg(L0_mean=("L0", "mean"), L1_mean=("L1", "mean"), n_bonds=("L0", "size")))

targets = pd.DataFrame(rows)
targets.to_csv(OUT / "targets.csv", index=False)
monthly.to_csv(OUT / "monthly_h1.csv")

# %% Report
pct = targets.assign(**{c: targets[c] * 100 for c in ("mean", "sd", "p01", "p99", "share_of_L1_sum_sq")})
table = pct.pivot_table(index="level", columns="horizon", values=["mean", "sd"]).reindex(["L0", "L1", "IC", "L2"])
split = pct[pct["level"].isin(["IC", "L2"])].pivot_table(index="level", columns="horizon", values="share_of_L1_sum_sq")
n = targets[targets["level"] == "L0"].set_index("horizon")[["n", "bonds", "issuers"]]
md = ["# Data summary", f"_generated {time.strftime('%Y-%m-%d %H:%M')}_",
      "\n### Panel\n", pd.Series(panel, name="value").to_frame().to_markdown(),
      "\n### Common sample, full panel\n", n.to_markdown(),
      "\n### Target returns (%), mean and SD\n", table.to_markdown(floatfmt=".2f"),
      "\n### Share of L1's sum of squares (%)\n", split.to_markdown(floatfmt=".1f")]
(OUT / "summary.md").write_text("\n".join(md) + "\n")
print(table.to_string(float_format=lambda v: f"{v:.2f}"), flush=True)
print(split.to_string(float_format=lambda v: f"{v:.1f}"), flush=True)
log(f"wrote {OUT}")

finish(_start)
