"""
Tables and inference for the Phase 2 grid, from the saved predictions only (no refitting).

Primary specification (section 6, reported first, following the Phase 2 proposal feedback):
every level and horizon scored on the same bond-months -- those present in every cell -- with
each forecast projected onto its target's constraint: L2 forecasts demeaned within issuer-month
(L2 sums to zero there by construction, so any issuer-wide part of a forecast is pure error),
IC forecasts averaged within it, L0 and L1 unchanged. Sections 1-5 use each horizon's own
common sample and raw forecasts; they are the appendix.

The IC target is fitted and scored on bond-month rows, so an issuer-month counts once per bond:
issuers with more bonds carry more weight in IC's R^2 and DM tests.

    1. OOS R^2 tables, model x level x horizon, against zero (primary) and the prevailing mean
    2. adding-up check: L1 label = IC label + L2 label, row by row, on the common sample
    3. modified Diebold-Mariano tests (Gu-Kelly-Xiu): EN vs OLS, RF vs OLS, RF vs EN per cell;
       each model vs the zero forecast (is R^2 > 0 distinguishable from noise?); and, for L2,
       all 341 predictors vs the 40 within-issuer-varying ones
    4. external benchmark: the DNR authors' own shipped forecasts, scored on our h=1 test rows
    5. sub-period robustness: every cell re-scored within each `config.SUBPERIODS` block of
       forecast-origin months (pre-COVID / COVID / post-COVID). No DM tests per sub-period: the
       12-month COVID block leaves too few months for a Newey-West standard error.
    6. primary specification: sections 1, 3 and 5 again on the all-horizon common sample with
       projected forecasts, plus the replication inside the pipeline (OLS / EN / RF, L0, h=1),
       scored on three samples to separate the test-window effect from bond composition

Writes output/phase2/tables.md and the CSVs behind it (primary_*.csv for section 6).

RUN: DSE4101_DATA_DIR=... python -m exploratory.phase2_results
"""

# %% Setup
from __future__ import annotations

import time
from itertools import combinations

import numpy as np
import pandas as pd

from src.config import OOS_START, OUTPUT_DIR, SUBPERIODS
from src.data import load_predictions
from src.harness import r2_from_sums, fold_sums, score, score_by_period
from src.inference import dm_test
from src.runlog import check, finish, log, report

OUT = OUTPUT_DIR / "phase2"
PREDS_DIR = OUT / "preds"
LEVEL_ORDER = ["L0", "L1", "IC", "L2"]
MODEL_ORDER = ["ols", "enet", "rf"]
YHAT = "yhat"  # sections 1-5 (appendix): raw forecasts
PRIMARY_YHAT = "yhat_proj"  # section 6 (primary): forecasts projected onto each target's constraint


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


# %% 5. Sub-periods
def subperiod_scores(scores: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for cell in scores[scores["sample"] == "common"].itertuples():
        p = load_cell(cell.cell)
        if p is None:
            continue
        by = score_by_period(p, SUBPERIODS, YHAT)
        check(int(by["n"].sum()) == len(p),
              f"{cell.cell}: sub-periods cover every scored row exactly once ({int(by['n'].sum()):,} of {len(p):,})")
        parts.append(by.assign(cell=cell.cell, model=cell.model, level=cell.level,
                               horizon=cell.horizon, predictor_set=cell.predictor_set))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def subperiod_tables(sub: pd.DataFrame) -> str:
    periods = [name for name, _, _ in SUBPERIODS]
    spans = ", ".join(f"{name} = {first} to {last}" for name, first, last in SUBPERIODS)
    lines = ["\n### Sub-period robustness\n",
             f"Same out-of-sample predictions, re-scored within blocks of forecast-origin months ({spans}). "
             "At h=12 a late-2019 forecast's label already includes 2020 returns.\n"]

    main = sub[sub["predictor_set"] == "all"]
    for metric, label in [("r2_zero", "vs zero forecast"), ("r2_hist", "vs prevailing training mean")]:
        t = main.pivot_table(index=["model", "level", "period"], columns="horizon", values=metric)
        t = t.reindex([(m, lv, per) for m in MODEL_ORDER for lv in LEVEL_ORDER for per in periods
                       if (m, lv, per) in t.index])
        lines += [f"\n#### OOS R² {label}, by sub-period\n", t.to_markdown(floatfmt="+.4f")]

    n_months = main.pivot_table(index="period", columns="horizon", values="n_months", aggfunc="first")
    lines += ["\n#### Forecast-origin months per sub-period\n", n_months.reindex(periods).to_markdown()]

    l2 = sub[(sub["level"] == "L2") & sub["predictor_set"].isin(["all", "within"])]
    if l2["predictor_set"].nunique() == 2:
        t = l2.pivot_table(index=["model", "period", "predictor_set"], columns="horizon", values="r2_zero")
        t = t.reindex([(m, per, ps) for m in MODEL_ORDER for per in periods for ps in ("all", "within")
                       if (m, per, ps) in t.index])
        lines += ["\n#### L2 OOS R² vs zero: all 341 predictors vs the 40 within-issuer ones, by sub-period\n",
                  t.to_markdown(floatfmt="+.4f")]
    return "\n".join(lines)


# %% 6. Primary specification: one sample for every level and horizon, projected forecasts
def common_rows(scores: pd.DataFrame) -> pd.Index:
    """Panel rows present in every main-grid cell, i.e. scored at every level and every horizon."""

    rows = None
    for cell in scores[(scores["sample"] == "common") & (scores["predictor_set"] == "all")]["cell"]:
        idx = load_cell(cell, cols=("row",)).index
        rows = idx if rows is None else rows.intersection(idx)
    return rows


def primary_cell(name: str, rows: pd.Index) -> pd.DataFrame | None:
    p = load_cell(name)
    return None if p is None else p.loc[rows]


def primary_scores(scores: pd.DataFrame, rows: pd.Index) -> pd.DataFrame:
    out = []
    for cell in scores[scores["sample"] == "common"].itertuples():
        p = primary_cell(cell.cell, rows)
        if p is None:
            continue
        r = dm_test(p["y"], np.zeros(len(p)), p[PRIMARY_YHAT], p["date"], cell.horizon)
        out.append({"cell": cell.cell, "model": cell.model, "level": cell.level, "horizon": cell.horizon,
                    "predictor_set": cell.predictor_set, **score(p, PRIMARY_YHAT),
                    "dm_stat_vs_zero": r["stat"], "p_vs_zero": r["p_value"], "n_months": r["n_months"]})
    return pd.DataFrame(out)


def primary_dm(scores: pd.DataFrame, rows: pd.Index) -> pd.DataFrame:
    out = []
    main = scores[(scores["sample"] == "common") & (scores["predictor_set"] == "all")]
    for (lv, h), grp in main.groupby(["level", "horizon"]):
        preds = {m: primary_cell(f"{m}_{lv}_h{h}_all_common", rows) for m in grp["model"]}
        for m1, m2 in combinations([m for m in MODEL_ORDER if preds.get(m) is not None], 2):
            a, b = preds[m1], preds[m2]
            out.append({"test": f"{m2} vs {m1}", "level": lv, "horizon": h, "predictor_set": "all",
                        **dm_test(a["y"], a[PRIMARY_YHAT], b[PRIMARY_YHAT], a["date"], h)})
    for (m, h), _ in main[main["level"] == "L2"].groupby(["model", "horizon"]):
        a = primary_cell(f"{m}_L2_h{h}_within_common", rows)
        b = primary_cell(f"{m}_L2_h{h}_all_common", rows)
        if a is None or b is None:
            continue
        out.append({"test": f"{m}: all vs within", "level": "L2", "horizon": h, "predictor_set": "all|within",
                    **dm_test(a["y"], a[PRIMARY_YHAT], b[PRIMARY_YHAT], a["date"], h)})
    return pd.DataFrame(out)


def primary_subperiods(scores: pd.DataFrame, rows: pd.Index) -> pd.DataFrame:
    parts = []
    for cell in scores[(scores["sample"] == "common") & (scores["predictor_set"] != "constant")].itertuples():
        p = primary_cell(cell.cell, rows)
        if p is None:
            continue
        by = score_by_period(p, SUBPERIODS, PRIMARY_YHAT)
        parts.append(by.assign(cell=cell.cell, model=cell.model, level=cell.level,
                               horizon=cell.horizon, predictor_set=cell.predictor_set))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def pipeline_replication(rows: pd.Index) -> pd.DataFrame:
    """
    Bali et al.'s check (does RF beat OLS one month ahead?) inside this pipeline: L0, h=1, on the
    h=1 common sample, on that sample cut at the primary sample's last month, and on the primary
    sample. The middle column separates the test-window effect from the bond-composition effect.
    """

    out = []
    preds = {m: load_cell(f"{m}_L0_h1_all_common") for m in MODEL_ORDER}
    preds = {m: p for m, p in preds.items() if p is not None}
    if not preds:
        return pd.DataFrame()
    any_p = next(iter(preds.values()))
    last = str(any_p.loc[rows, "date"].astype(str).str[:7].max())
    samples = {"h1_sample": any_p.index,
               f"h1_sample_to_{last}": any_p.index[any_p["date"].astype(str).str[:7].to_numpy() <= last],
               "primary": rows}
    for name, idx in samples.items():
        for m, p in preds.items():
            q = p.loc[idx]
            out.append({"sample": name, "test": f"{m} vs zero", "n": len(q),
                        "r2_zero": score(q, YHAT)["r2_zero"],
                        **dm_test(q["y"], np.zeros(len(q)), q[YHAT], q["date"], 1)})
        if "ols" in preds and "rf" in preds:
            a, b = preds["ols"].loc[idx], preds["rf"].loc[idx]
            out.append({"sample": name, "test": "rf vs ols", "n": len(a), "r2_zero": np.nan,
                        **dm_test(a["y"], a[YHAT], b[YHAT], a["date"], 1)})
    return pd.DataFrame(out)


def primary_tables(ps: pd.DataFrame, pdm: pd.DataFrame, psub: pd.DataFrame, rep: pd.DataFrame,
                   rows: pd.Index) -> str:
    months = load_cell(ps["cell"].iloc[0], cols=("row", "date")).loc[rows, "date"].astype(str).str[:7]
    lines = ["## Primary specification",
             f"All levels and horizons scored on the same {len(rows):,} bond-months "
             f"({months.min()} to {months.max()}, {months.nunique()} forecast-origin months): those present "
             "in every cell. Forecasts projected onto each target's constraint (L2 demeaned within "
             "issuer-month, IC averaged within it, L0/L1 unchanged). Significance vs a zero forecast "
             "(modified DM): * 10%, ** 5%, *** 1%.\n"]

    main = ps[ps["predictor_set"] == "all"].assign(
        cell_txt=lambda d: (d["r2_zero"] * 100).map(lambda v: f"{v:+.2f}%") + d["p_vs_zero"].map(stars))
    t = main.pivot_table(index=["model", "level"], columns="horizon", values="cell_txt", aggfunc="first")
    t = t.reindex([(m, lv) for m in MODEL_ORDER for lv in LEVEL_ORDER if (m, lv) in t.index])
    lines += ["\n### OOS R² vs zero forecast\n", t.to_markdown()]
    t = main.pivot_table(index=["model", "level"], columns="horizon", values="r2_hist")
    t = t.reindex([(m, lv) for m in MODEL_ORDER for lv in LEVEL_ORDER if (m, lv) in t.index])
    lines += ["\n### OOS R² vs prevailing training mean\n", t.to_markdown(floatfmt="+.4f")]

    show = pdm.assign(stat=pdm["stat"].map(lambda v: f"{v:+.2f}") + pdm["p_value"].map(stars),
                      p_value=pdm["p_value"].map(lambda v: f"{v:.3f}"))
    lines += ["\n### Modified Diebold–Mariano tests between forecasts\n",
              "Positive stat: the first-named model (for 'all vs within', the 341-predictor set) has lower MSE.\n",
              show[["test", "level", "horizon", "stat", "p_value", "n_months", "lags"]]
              .sort_values(["test", "level", "horizon"]).to_markdown(index=False)]

    if len(rep):
        t = rep[rep["test"].str.endswith("vs zero")].assign(model=lambda d: d["test"].str.split(" ").str[0])
        t = t.pivot_table(index="model", columns="sample", values="r2_zero").reindex(MODEL_ORDER)
        t = t[[c for c in rep["sample"].unique()]]
        n = rep.groupby("sample", sort=False)["n"].first()
        rv = rep[rep["test"] == "rf vs ols"].set_index("sample")
        lines += ["\n### Replication inside the pipeline: L0, h=1, OOS R² vs zero (raw forecasts)\n",
                  t.to_markdown(floatfmt="+.4f"),
                  "\nBond-months: " + ", ".join(f"{k} {v:,}" for k, v in n.items()) + ". RF vs OLS (DM, positive = RF "
                  "better): " + ", ".join(f"{k} {r.stat:+.2f} (p={r.p_value:.3f})" for k, r in rv.iterrows()) + "."]

    if len(psub):
        l2 = psub[(psub["level"] == "L2") & (psub["predictor_set"] == "all")]
        t = l2.pivot_table(index=["model", "period"], columns="horizon", values="r2_zero")
        periods = [name for name, _, _ in SUBPERIODS]
        t = t.reindex([(m, per) for m in MODEL_ORDER for per in periods if (m, per) in t.index])
        nm = l2.pivot_table(index="period", columns="horizon", values="n_months", aggfunc="first").reindex(periods)
        lines += ["\n### L2 OOS R² vs zero by sub-period\n", t.to_markdown(floatfmt="+.4f"),
                  "\nForecast-origin months per sub-period:\n", nm.to_markdown()]
    return "\n".join(lines)


# %% Run
def main() -> None:
    _start = time.time()
    scores = pd.read_csv(OUT / "scores.csv")
    report(f"{len(scores)} scored cells")
    starts = sorted(scores["oos_start"].astype(str).unique())
    check(starts == [OOS_START], f"every cell was fitted with config.OOS_START={OOS_START} (found {starts})")

    check_adding_up(scores)

    dm = dm_table(scores)
    dm.to_csv(OUT / "dm_tests.csv", index=False)

    dnr = dnr_benchmark()
    dnr.to_csv(OUT / "dnr_benchmark.csv", index=False)

    rows = common_rows(scores)
    ps, pdm, psub = primary_scores(scores, rows), primary_dm(scores, rows), primary_subperiods(scores, rows)
    for cell in ps.itertuples():
        if cell.predictor_set == "constant":
            check(abs(cell.r2_zero) < 1e-12,
                  f"{cell.cell}: issuer-constant predictors give a zero demeaned L2 forecast (R² {cell.r2_zero:.1e})")
    rep = pipeline_replication(rows)
    ps.to_csv(OUT / "primary_scores.csv", index=False)
    rep.to_csv(OUT / "primary_replication.csv", index=False)
    pdm.to_csv(OUT / "primary_dm_tests.csv", index=False)
    psub.to_csv(OUT / "primary_subperiods.csv", index=False)
    report(f"primary sample: {len(rows):,} bond-months common to every level and horizon")

    md = ["# Phase 2 results", f"_generated {time.strftime('%Y-%m-%d %H:%M')} from output/phase2/_",
          primary_tables(ps, pdm, psub, rep, rows),
          "\n## Appendix: each horizon's own common sample, raw forecasts", r2_tables(scores)]
    if len(dm):
        dm_show = dm.assign(
            stat=dm["stat"].map(lambda v: f"{v:+.2f}") + dm["p_value"].map(stars),
            p_value=dm["p_value"].map(lambda v: f"{v:.3f}"),
        )[["test", "level", "horizon", "predictor_set", "stat", "p_value", "n_months", "lags"]]
        md += ["\n### Modified Diebold–Mariano tests\n",
               "Positive stat: the first-named model (or, for 'vs zero', the model) has lower MSE. "
               "Newey–West, lags = max(h−1, ⌊4(T/100)^(2/9)⌋). * 10%, ** 5%, *** 1%.\n",
               dm_show.sort_values(["test", "level", "horizon"]).to_markdown(index=False)]
    if len(dnr):
        md += ["\n### DNR shipped forecasts on our h=1 test rows\n", dnr.to_markdown(index=False, floatfmt="+.4f")]

    sub = subperiod_scores(scores)
    sub.to_csv(OUT / "subperiods.csv", index=False)
    if len(sub):
        md.append(subperiod_tables(sub))
    (OUT / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"wrote {OUT / 'tables.md'}")

    finish(_start)


if __name__ == "__main__":
    main()
