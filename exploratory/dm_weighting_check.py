"""
Does the DM test's month-equal weighting matter for our conclusions?

OOS R^2 pools every test bond-month, so a month counts in proportion to its number of bonds. The
modified DM test (src/inference.py) averages the loss differential across bonds within each month
first, so every month counts equally. This reruns every reported test with each month's
differential summed rather than averaged (bond-month weights, matching the pooled R^2) and
reports whether any 5% verdict changes, for:

    1. each model against the zero forecast, main grid, raw forecasts (as in scores.csv)
    2. each model against the zero forecast, primary specification (as in primary_scores.csv)
    3. each pair of models, main grid, raw forecasts (as in dm_tests.csv)

Reads only the saved forecasts in output/phase2/preds. Writes output/phase2/dm_weighting_check.csv.

RUN: python -m exploratory.dm_weighting_check
"""

# %% Setup
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from exploratory.phase2_results import OUT, PRIMARY_YHAT, common_rows, load_cell
from src.inference import dm_lags, newey_west_se
from src.runlog import check, finish, log, report

MODELS = ("ols", "enet", "rf")
LEVELS = ("L0", "L1", "IC", "L2")
HORIZONS = (1, 3, 12)
PAIRS = (("ols", "enet"), ("ols", "rf"), ("enet", "rf"))


def both_weightings(loss_diff: np.ndarray, months: np.ndarray, horizon: int) -> dict[str, float]:
    """DM statistic and p-value with month-equal weights (as reported) and bond-month weights."""

    g = pd.DataFrame({"m": months, "d": loss_diff}).groupby("m")["d"].agg(["mean", "sum", "size"]).sort_index()
    lags = dm_lags(horizon, len(g))
    out = {"n_months": len(g), "min_bonds": int(g["size"].min()), "max_bonds": int(g["size"].max())}
    for name, col in (("month_equal", "mean"), ("bond_weighted", "sum")):
        s = g[col].to_numpy()
        stat = float(s.mean() / newey_west_se(s, lags))
        out[f"stat_{name}"] = stat
        out[f"p_{name}"] = math.erfc(abs(stat) / math.sqrt(2))
    out["verdict_changes"] = (out["p_month_equal"] < 0.05) != (out["p_bond_weighted"] < 0.05)
    return out


# %% Run
_start = time.time()
scores = pd.read_csv(OUT / "scores.csv")
rows_primary = common_rows(scores)
out = []
for level in LEVELS:
    for h in HORIZONS:
        cells = {m: load_cell(f"{m}_{level}_h{h}_all_common") for m in MODELS}
        for m, p in cells.items():
            for spec, frame, col in (("main", p, "yhat"), ("primary", p.loc[rows_primary], PRIMARY_YHAT)):
                yy = frame["y"].to_numpy()
                d = yy ** 2 - (yy - frame[col].to_numpy()) ** 2
                out.append({"spec": spec, "test": f"{m} vs zero", "level": level, "horizon": h,
                            **both_weightings(d, frame["date"].to_numpy(), h)})
        for a, b in PAIRS:
            y = cells[a]["y"].to_numpy()
            check(np.array_equal(y, cells[b]["y"].to_numpy()), f"{a} and {b} share rows at {level} h={h}")
            d = (y - cells[a]["yhat"].to_numpy()) ** 2 - (y - cells[b]["yhat"].to_numpy()) ** 2
            out.append({"spec": "main", "test": f"{b} vs {a}", "level": level, "horizon": h,
                        **both_weightings(d, cells[a]["date"].to_numpy(), h)})

res = pd.DataFrame(out)

# the month-equal column must reproduce the reported tests exactly
reported = pd.concat([
    pd.read_csv(OUT / "dm_tests.csv").query("predictor_set == 'all'").assign(spec="main"),
    pd.read_csv(OUT / "primary_scores.csv").query("predictor_set == 'all'")
      .assign(spec="primary", test=lambda t: t["model"] + " vs zero", stat=lambda t: t["dm_stat_vs_zero"]),
])
j = res.merge(reported[["spec", "test", "level", "horizon", "stat"]], on=["spec", "test", "level", "horizon"], how="left")
gap = float((j["stat_month_equal"] - j["stat"]).abs().max())
check(j["stat"].notna().all() and gap < 1e-9, f"month-equal statistics match the reported tests (max gap {gap:.1e})")
res.to_csv(OUT / "dm_weighting_check.csv", index=False)
for (spec, kind), g in res.groupby(["spec", res["test"].str.endswith("zero")]):
    report(f"{spec}, {'vs zero' if kind else 'model vs model'}: 5% verdict changes in "
           f"{int(g['verdict_changes'].sum())} of {len(g)} tests")
report(f"bonds per month: {res['min_bonds'].min():,} to {res['max_bonds'].max():,}")
log(f"wrote {OUT / 'dm_weighting_check.csv'}")
finish(_start)
