"""
Phase 2 grid: {OLS, Elastic Net, Random Forest} x {L0, L1, IC, L2} x h in {1, 3, 12}, plus the
L2 predictor-set comparison (all 341 vs the 40 within-issuer-varying predictors).

Every cell is fitted fold by fold with the h-month embargo (src/splits.py) and, by default, on
the common sample: the same L2-eligible rows for every level at a given horizon, so moving
down the decomposition changes only the target.

Each finished cell writes its out-of-sample predictions to output/phase2/preds/ and appends a
row to output/phase2/scores.csv; a rerun skips cells whose predictions already exist, so a long
Random Forest run can be stopped and resumed. Inference (DM tests) and the tables are in
`exploratory/phase2_results.py`, which reads only these files.

RUN: DSE4101_DATA_DIR=... python -m exploratory.phase2_grid --models ols,enet,rf
     python -m exploratory.phase2_grid --models rf --levels L2 --predictor-set within
     python -m exploratory.phase2_grid --models enet --horizons 1 --levels L0 --max-folds 1   # smoke test
"""

# %% Setup
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.config import OOS_START as CONFIG_OOS_START, OUTPUT_DIR
from src.data import PREDICTOR_COLS, load_panel
from src.harness import LEVELS, build_labels, run_level, score
from src.models import TunedElasticNet, TunedRandomForest
from src.pipeline import ols_builder
from src.runlog import check, finish, log, report, rss_gb

OOS_START = pd.Period(CONFIG_OOS_START, freq="M")
REFIT_FREQ = 12
HORIZONS = (1, 3, 12)

OUT = OUTPUT_DIR / "phase2"
PREDS_DIR = OUT / "preds"
DIAG_DIR = OUT / "diagnostics"

BUILDERS = {
    "ols": ols_builder(),
    "enet": TunedElasticNet,
    "rf": TunedRandomForest,
}


def predictor_set(name: str) -> list[str]:
    """
    all      : the 341 panel predictors
    within   : the 40 that vary across an issuer's bonds within a month
    constant : the 301 that do not (a pipeline check: zero signal for L2 under linear models)

    The split is read from output/within_issuer_predictors.csv
    (exploratory/within_issuer_predictors.py).
    """

    if name == "all":
        return list(PREDICTOR_COLS)
    split = pd.read_csv(OUTPUT_DIR / "within_issuer_predictors.csv", index_col="predictor")
    varying = set(split.index[split["varying"]])
    if name == "within":
        return [p for p in PREDICTOR_COLS if p in varying]
    if name == "constant":
        return [p for p in PREDICTOR_COLS if p not in varying]
    raise ValueError(f"unknown predictor set {name!r}")


def cell_name(model: str, level: str, horizon: int, pset: str, common: bool) -> str:
    return f"{model}_{level}_h{horizon}_{pset}_{'common' if common else 'own'}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ols,enet,rf")
    ap.add_argument("--levels", default="L0,L1,IC,L2")
    ap.add_argument("--horizons", default="1,3,12")
    ap.add_argument("--predictor-set", default="all", choices=["all", "within", "constant"])
    ap.add_argument("--own-sample", action="store_true",
                    help="midterm samples (L0/L1 on all labelled rows) instead of the common sample")
    ap.add_argument("--max-folds", type=int, default=None, help="smoke test: stop after this many folds")
    ap.add_argument("--force", action="store_true", help="refit cells whose predictions already exist")
    return ap.parse_args()


# %% Run
def main() -> None:
    args = parse_args()
    _start = time.time()
    for d in (PREDS_DIR, DIAG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    models = args.models.split(",")
    levels = [lvl for lvl in LEVELS if lvl.name in args.levels.split(",")]
    horizons = [int(h) for h in args.horizons.split(",")]
    common = not args.own_sample
    pset = args.predictor_set
    predictors = predictor_set(pset)
    smoke = args.max_folds is not None

    todo = [
        (m, lvl, h) for m in models for h in horizons for lvl in levels
        if args.force or smoke or not (PREDS_DIR / f"{cell_name(m, lvl.name, h, pset, common)}.parquet").exists()
    ]

    # A saved cell is skipped on resume, so one fitted under an older OOS start would otherwise
    # sit in the tables next to cells fitted under the current one.
    scores_path = OUT / "scores.csv"
    if scores_path.exists() and not (args.force or smoke):
        saved = pd.read_csv(scores_path, dtype={"oos_start": str})
        stale = saved.loc[saved["oos_start"] != str(OOS_START), "cell"].tolist()
        if stale:
            raise SystemExit(
                f"{len(stale)} saved cell(s) in {scores_path} were fitted with a different OOS start "
                f"than config.OOS_START={OOS_START}, e.g. {stale[:3]}. Move output/phase2 aside and "
                f"rerun, or pass --force to refit."
            )
    log(f"{len(todo)} cell(s) to run; predictor set {pset!r} ({len(predictors)} predictors); "
        f"{'common' if common else 'own'} sample")
    if not todo:
        return

    df = load_panel(cols=["date", "cusip", "gvkey", *predictors], targets=("retx", "retxrf"))
    df[predictors] = df[predictors].astype("float32")
    build_labels(df, HORIZONS)
    report(f"panel loaded: {df.shape}, {df.memory_usage(deep=True).sum() / 1e9:.2f}GB, rss {rss_gb():.2f}GB")

    n_by_h: dict[int, set[int]] = {}
    for model, level, horizon in todo:
        name = cell_name(model, level.name, horizon, pset, common)
        log(f"=== {name} ===")
        t0 = time.time()

        if smoke:
            # Run only the last `max_folds` folds: the largest training windows, so timings are worst-case.
            import src.harness as H
            months = pd.PeriodIndex(pd.to_datetime(df["date"]), freq="M").unique().sort_values()
            folds = H._generate(months, OOS_START, horizon, REFIT_FREQ, embargo=horizon)[-args.max_folds:]
            eligible = H.eligible_rows(df, level, horizon, common)
            for fold in folds:
                tf = time.time()
                p, d = H.fit_predict_fold(df, fold, level, horizon, predictors, BUILDERS[model], eligible)
                report(f"smoke {name} fold {fold.fit_month}: {time.time() - tf:.1f}s, n_train={d['n_train']:,}, "
                       f"n_test={d['n_test']:,}, rss {rss_gb():.2f}GB, "
                       f"params={ {k: v for k, v in d.items() if k.startswith('param_')} }, "
                       f"fold r2_zero={score(p)['r2_zero']:.4f}")
            continue

        preds, diagnostics, folds, eligible = run_level(
            df, level, horizon, OOS_START, REFIT_FREQ, predictors, BUILDERS[model], common,
            on_fold=lambda d: log(f"  fold {d['fit_month']}: n_train={d['n_train']:,} "
                                  f"{ {k[6:]: v for k, v in d.items() if k.startswith('param_')} }"),
        )

        check(
            preds.index.is_unique and len(preds) == int(sum(d["n_test"] for d in diagnostics)),
            f"{name}: every test row predicted exactly once ({len(preds):,} rows, {len(folds)} folds)",
        )
        n_by_h.setdefault(horizon, set()).add(len(preds))

        out = preds.assign(date=preds["date"].astype(str), fit_month=preds["fit_month"].astype(str))
        out.reset_index(names="row").to_parquet(PREDS_DIR / f"{name}.parquet", index=False)
        pd.DataFrame(diagnostics).to_csv(DIAG_DIR / f"{name}.csv", index=False)

        s_raw, s_proj = score(preds, "yhat"), score(preds, "yhat_proj")
        row = {
            "cell": name, "model": model, "level": level.name, "horizon": horizon,
            "predictor_set": pset, "sample": "common" if common else "own",
            **{f"{k}": v for k, v in s_raw.items()},
            **{f"{k}_proj": v for k, v in s_proj.items() if k != "n"},
            "n_folds": len(folds), "oos_start": str(OOS_START), "refit_freq": REFIT_FREQ,
            "seconds": round(time.time() - t0, 1),
        }
        scores_path = OUT / "scores.csv"
        prev = pd.read_csv(scores_path) if scores_path.exists() else pd.DataFrame()
        if len(prev):
            prev = prev[prev["cell"] != name]
        pd.concat([prev, pd.DataFrame([row])], ignore_index=True).to_csv(scores_path, index=False)

        report(f"{name}: r2_zero={s_raw['r2_zero']:.4f} r2_hist={s_raw['r2_hist']:.4f} "
               f"r2_zero_proj={s_proj['r2_zero']:.4f} n={s_raw['n']:,} in {time.time() - t0:.0f}s, "
               f"rss {rss_gb():.2f}GB")
        del preds, out

    if common:
        for h, ns in n_by_h.items():
            check(len(ns) == 1, f"h={h}: every level in this run scored on the same number of rows ({sorted(ns)})")

    finish(_start)


if __name__ == "__main__":
    main()
