"""
Tables and inference for the Phase 2 grid, from the saved predictions only (no refitting).

    1. OOS R^2 tables, model x level x horizon, against zero (primary) and the prevailing mean
    2. adding-up check: L1 label = IC label + L2 label, row by row, on the common sample
    3. modified Diebold-Mariano tests (Gu-Kelly-Xiu): EN vs OLS, RF vs OLS, RF vs EN per cell;
       each model vs the zero forecast (is R^2 > 0 distinguishable from noise?); and, for L2,
       all 341 predictors vs the 40 within-issuer-varying ones
    4. external benchmark: the DNR authors' own shipped forecasts, scored on our h=1 test rows

Writes output/phase2/tables.md and the CSVs behind it.

RUN: DSE4101_DATA_DIR=... python -m exploratory.phase2_results
"""

# %% Setup
from __future__ import annotations

import time
from itertools import combinations

import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR
from src.data import load_predictions
from src.harness import r2_from_sums, fold_sums
from src.inference import dm_test
from src.runlog import check, finish, log, report

OUT = OUTPUT_DIR / "phase2"
PREDS_DIR = OUT / "preds"
LEVEL_ORDER = ["L0", "L1", "IC", "L2"]
MODEL_ORDER = ["ols", "enet", "rf"]
YHAT = "yhat"  # raw forecasts are primary; yhat_proj is the adding-up-consistent variant


def load_cell(name: str, cols=("row", "date", "y", "yhat", "yhat_proj", "ybar")) -> pd.DataFrame | None:
    path = PREDS_DIR / f"{name}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path, columns=list(cols)).set_index("row").sort_index()


def fmt(v: float) -> str:
    return "" if pd.isna(v) else f"{v:+.4f}"


def stars(p: float) -> str:
    return "" if pd.isna(p) else "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


# %% 1. R^2 tables
def r2_tables(scores: pd.DataFrame) -> str:
    lines = []
    sc = scores[(scores["predictor_set"] == "all") & (scores["sample"] == "common")]
    for metric, label in [("r2_zero", "vs zero forecast"), ("r2_hist", "vs prevailing training mean"),
                          ("r2_zero_proj", "vs zero, forecasts projected onto the level's constraint")]:
        lines.append(f"\n### OOS R² {label}\n")
        t = sc.pivot_table(index=["model", "level"], columns="horizon", values=metric)
        t = t.reindex([(m, lv) for m in MODEL_ORDER for lv in LEVEL_ORDER if (m, lv) in t.index])
        lines.append(t.to_markdown(floatfmt="+.4f"))
    n = sc.pivot_table(index="level", columns="horizon", values="n", aggfunc="first")
    lines.append("\n### Scored bond-months (identical across levels on the common sample)\n")
    lines.append(n.reindex([lv for lv in LEVEL_ORDER if lv in n.index]).to_markdown())
    return "\n".join(lines)


# %% 2. Adding up
def check_adding_up(scores: pd.DataFrame) -> None:
    for (model, h), _ in scores[scores["predictor_set"] == "all"].groupby(["model", "horizon"]):
        parts = {lv: load_cell(f"{model}_{lv}_h{h}_all_common", cols=("row", "y")) for lv in ("L1", "IC", "L2")}
        if any(p is None for p in parts.values()):
            continue
        l1, ic, l2 = (parts[k]["y"] for k in ("L1", "IC", "L2"))
        same_rows = l1.index.equals(ic.index) and l1.index.equals(l2.index)
        gap = float((l1 - ic - l2).abs().max()) if same_rows else np.inf
        check(same_rows and gap < 1e-12, f"{model} h={h}: L1 = IC + L2 on identical rows (max gap {gap:.2e})")


# %% 3. DM tests
def dm_table(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cells = scores[(scores["sample"] == "common")]

    # model vs model, and model vs zero, within each (level, horizon, predictor set)
    for (lv, h, ps), grp in cells.groupby(["level", "horizon", "predictor_set"]):
        preds = {m: load_cell(f"{m}_{lv}_h{h}_{ps}_common") for m in grp["model"]}
        preds = {m: p for m, p in preds.items() if p is not None}
        for m, p in preds.items():
            r = dm_test(p["y"], np.zeros(len(p)), p[YHAT], p["date"], h)
            rows.append({"test": f"{m} vs zero", "level": lv, "horizon": h, "predictor_set": ps, **r})
        for m1, m2 in combinations([m for m in MODEL_ORDER if m in preds], 2):
            a, b = preds[m1], preds[m2]
            check(a.index.equals(b.index), f"{lv} h={h} {ps}: {m1} and {m2} scored on identical rows")
            r = dm_test(a["y"], a[YHAT], b[YHAT], a["date"], h)
            rows.append({"test": f"{m2} vs {m1}", "level": lv, "horizon": h, "predictor_set": ps, **r})

    # L2: all predictors vs within-issuer-varying only, per model
    for (m, h), _ in cells[cells["level"] == "L2"].groupby(["model", "horizon"]):
        a = load_cell(f"{m}_L2_h{h}_within_common")
        b = load_cell(f"{m}_L2_h{h}_all_common")
        if a is None or b is None:
            continue
        check(a.index.equals(b.index), f"{m} L2 h={h}: predictor-set comparison on identical rows")
        r = dm_test(a["y"], a[YHAT], b[YHAT], a["date"], h)
        rows.append({"test": f"{m}: all vs within", "level": "L2", "horizon": h, "predictor_set": "all|within", **r})

    return pd.DataFrame(rows)


# %% 4. DNR shipped forecasts as an external benchmark
def dnr_benchmark() -> pd.DataFrame:
    """
    The forecasts shipped with the DNR data, scored on our h=1 common-sample test rows. How the
    authors estimated them (sample split, refit schedule, tuning) is not documented in the
    download and has not been checked here. Scored against our winsorized labels and against
    their raw realized returns, so the effect of winsorizing is visible.
    """

    rows = []
    for lv, target in (("L0", "retxrf"), ("L1", "retx")):
        ours = load_cell(f"ols_{lv}_h1_all_common", cols=("row", "date", "cusip", "y"))
        if ours is None:
            continue
        theirs = load_predictions(target)
        theirs["date"] = theirs["signal_date"].dt.to_period("M").astype(str)
        theirs["cusip"] = theirs["cusip"].astype(str)
        merged = ours.merge(theirs, on=["date", "cusip"], how="inner")
        for model, g in merged.groupby("model_key"):
            rows.append({
                "level": lv, "dnr_model": model, "n": len(g),
                "coverage": len(g) / len(ours),
                "r2_zero_our_label": r2_from_sums(fold_sums(g["y"], g["prediction"]), 0.0),
                "r2_zero_raw_return": r2_from_sums(fold_sums(g["realized_return"], g["prediction"]), 0.0),
            })
    return pd.DataFrame(rows)


# %% Run
def main() -> None:
    _start = time.time()
    scores = pd.read_csv(OUT / "scores.csv")
    report(f"{len(scores)} scored cells")

    check_adding_up(scores)

    dm = dm_table(scores)
    dm.to_csv(OUT / "dm_tests.csv", index=False)

    dnr = dnr_benchmark()
    dnr.to_csv(OUT / "dnr_benchmark.csv", index=False)

    md = ["# Phase 2 results", f"_generated {time.strftime('%Y-%m-%d %H:%M')} from output/phase2/_", r2_tables(scores)]
    if len(dm):
        dm_show = dm.assign(
            stat=dm["stat"].map(lambda v: f"{v:+.2f}") + dm["p_value"].map(stars),
            p_value=dm["p_value"].map(lambda v: f"{v:.3f}"),
        )[["test", "level", "horizon", "predictor_set", "stat", "p_value", "n_months", "lags"]]
        md += ["\n### Modified Diebold–Mariano tests\n",
               "Positive stat: the second-named model (or, for 'vs zero', the model) has lower MSE. "
               "Newey–West, lags = max(h−1, ⌊4(T/100)^(2/9)⌋). * 10%, ** 5%, *** 1%.\n",
               dm_show.sort_values(["test", "level", "horizon"]).to_markdown(index=False)]
    if len(dnr):
        md += ["\n### DNR shipped forecasts on our h=1 test rows\n", dnr.to_markdown(index=False, floatfmt="+.4f")]
    (OUT / "tables.md").write_text("\n".join(md) + "\n")
    log(f"wrote {OUT / 'tables.md'}")

    finish(_start)


if __name__ == "__main__":
    main()
