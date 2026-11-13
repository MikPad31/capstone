"""
Fold-by-fold fit/predict harness for the return decomposition.

Promoted from `exploratory/target_pipeline.py` (which stays as the record of the midterm run) and
extended for Phase 2:

    - a fourth level, IC (issuer-common component), alongside L0/L1/L2
    - a common-sample mask so every level can be scored on the L2-eligible rows
    - estimators that tune on an embargoed inner split receive the training signal months
    - predictions carry `ybar` (training-window mean of the label: the prevailing-mean benchmark)
      and `yhat_proj` (the forecast pushed through the level's own transform, see `project`)
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Sequence

import numpy as np
import pandas as pd

from src.decompose import _cast_keys, demean, issuer_month_stats
from src.metrics import winsorize_cross_section
from src.runlog import check, report
from src.splits import Fold, _generate, rows_for_months
from src.targets import forward_return

RET_COLS = ("retx_realized_return", "retxrf_realized_return")


# %% Levels

class Level(NamedTuple):
    """
    One rung of the decomposition.

    transform:
        "none"       : label used as is (L0, L1)
        "demean"     : label minus its issuer-month mean (L2)
        "group_mean" : issuer-month mean of the label, assigned to every bond in the group (IC)

    By construction L1 = IC + L2 row by row on the common sample.
    """

    name: str
    target_col: str
    transform: str
    key: str | None
    min_group_size: int = 2


L0 = Level(name="L0", target_col="retxrf_realized_return", transform="none", key=None)
L1 = Level(name="L1", target_col="retx_realized_return", transform="none", key=None)
IC = Level(name="IC", target_col="retx_realized_return", transform="group_mean", key="gvkey")
L2 = Level(name="L2", target_col="retx_realized_return", transform="demean", key="gvkey")

LEVELS = (L0, L1, IC, L2)


def label_col(level: Level, horizon: int) -> str:
    return f"{level.target_col}_fwd{horizon}"


# %% Loading

def build_labels(df: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    """
    Adds every `<ret_col>_fwd<h>` label to `df` in place: compounded within cusip, then
    winsorized at 1/99 within formation month. Same construction as the midterm run.
    """

    months = pd.PeriodIndex(pd.to_datetime(df["date"]), freq="M")
    for ret_col in RET_COLS:
        for horizon in horizons:
            col = f"{ret_col}_fwd{horizon}"
            raw = forward_return(df, ret_col, horizon)
            notna = raw.notna()
            temp = pd.DataFrame({col: raw.loc[notna], "_month": months[notna.to_numpy()]})
            df[col] = np.nan
            df.loc[notna, col] = winsorize_cross_section(temp, col, ["_month"], 0.01, 0.99)
    return df


# %% Eligibility

def _group_size(df: pd.DataFrame, mask: np.ndarray, key: str, date_col: str) -> np.ndarray:
    """Number of `mask` rows sharing each row's (key, month). NaN where the key is missing."""

    key_notna = df[key].notna().to_numpy()
    keys = pd.Series(np.nan, index=df.index, dtype="object")
    if key_notna.any():
        keys.loc[key_notna] = _cast_keys(df.loc[key_notna], [key])[0].to_numpy()

    months = pd.PeriodIndex(pd.to_datetime(df[date_col]), freq="M")
    use = mask & key_notna
    stats = issuer_month_stats(
        df=df.loc[use].assign(_month=months[use]),
        key=key,
        time="_month",
    ).set_index([key, "_month"])["n_bonds"]

    pair_index = pd.MultiIndex.from_arrays([keys, months], names=[key, "_month"])
    return stats.reindex(pair_index).to_numpy()


def eligible_rows(
        df: pd.DataFrame,
        level: Level,
        horizon: int,
        common_sample: bool,
        date_col: str = "date",
) -> np.ndarray:
    """
    Boolean mask of the rows a level is fitted and scored on at `horizon`.

    common_sample=False reproduces the midterm: L0/L1 use every labelled row, L2 drops
    issuer-months with fewer than `min_group_size` bonds.

    common_sample=True gives every level the same rows: both labels present, issuer key present,
    and at least `min_group_size` labelled bonds in the issuer-month. L0 -> L1 -> IC/L2 is then a
    pure change of target, not of sample.
    """

    lab = df[label_col(level, horizon)].notna().to_numpy()

    if common_sample:
        both = (
            df[f"retx_realized_return_fwd{horizon}"].notna().to_numpy()
            & df[f"retxrf_realized_return_fwd{horizon}"].notna().to_numpy()
        )
        n = _group_size(df, both, "gvkey", date_col)
        mask = both & (n >= L2.min_group_size)
    elif level.transform == "none":
        mask = lab
    else:
        if level.key is None:
            raise ValueError(f"Level {level.name} transforms but has no key set")
        n = _group_size(df, lab, level.key, date_col)
        mask = lab & (n >= level.min_group_size)

    if not mask.any():
        raise ValueError(f"eligible_rows: mask is all False for level {level.name} h={horizon}")
    return mask


# %% Fold frames and labels

def fold_masks(
        df: pd.DataFrame,
        fold: Fold,
        eligible: np.ndarray,
        level: Level,
        date_col: str = "date",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train and test row masks for one fold, restricted to `eligible` (and to a present issuer key
    for levels that group by one). Masks rather than frames, so the caller copies only the
    columns it needs: the predictor block is ~1.4GB per million rows.
    """

    keep = eligible
    if level.key is not None:
        keep = keep & df[level.key].notna().to_numpy()
    train_mask = rows_for_months(df, fold.train_months, date_col) & keep
    test_mask = rows_for_months(df, fold.test_months, date_col) & keep
    if not train_mask.any():
        raise ValueError(f"fold_masks: empty train slice for fold {fold.fit_month}")
    if not test_mask.any():
        raise ValueError(f"fold_masks: empty test slice for fold {fold.fit_month}")
    return train_mask, test_mask


def project(
        frame: pd.DataFrame,
        values: pd.Series,
        level: Level,
        date_col: str = "date",
) -> pd.Series:
    """
    Applies the level's transform to `values` within (level.key, month) of `frame`.

    Used for the label on each side of a fold, and for the forecast on the test side: an L2
    forecast is demeaned within issuer-month and an IC forecast averaged within it, so the
    forecast obeys the same adding-up constraint as the target. Uses only the rows of `frame`,
    never labels from the other side.
    """

    if level.transform == "none":
        return values.astype("float64")

    months = pd.PeriodIndex(pd.to_datetime(frame[date_col]), freq="M")
    tmp = frame[[level.key]].assign(_month=months, _v=values.to_numpy(dtype="float64"))

    if level.transform == "demean":
        return demean(tmp, "_v", [level.key, "_month"])
    if level.transform == "group_mean":
        return tmp["_v"].groupby(_cast_keys(tmp, [level.key, "_month"]), observed=True).transform("mean")
    raise ValueError(f"project: unknown transform {level.transform!r}")


def select_predictors(X_train: pd.DataFrame) -> list[str]:
    """Drops predictors with zero variance on the training rows only."""

    variances = X_train.var(axis=0, skipna=True)
    dropped = [p for p in X_train.columns if not (variances.get(p, 0.0) > 0.0)]
    if dropped:
        report(f"dropped {len(dropped)} zero-variance predictors: {dropped}")
    return [p for p in X_train.columns if p not in dropped]


# %% Fit / predict

def fit_predict_fold(
        df: pd.DataFrame,
        fold: Fold,
        level: Level,
        horizon: int,
        predictors: Sequence[str],
        estimator_builder: Callable[[], object],
        eligible: np.ndarray,
        date_col: str = "date",
) -> tuple[pd.DataFrame, dict]:
    """
    Fits one fold on its training months and predicts its test months.

    Estimators with `needs_months = True` get `months=` (the training rows' signal months) and
    `horizon=` in `fit`, so they can tune on an embargoed inner split of the training window.

    Returns predictions (date, cusip, gvkey, fit_month, y, yhat, yhat_proj, ybar) and a
    diagnostics dict.
    """

    col = label_col(level, horizon)
    train_mask, test_mask = fold_masks(df, fold, eligible, level, date_col)

    meta = list(dict.fromkeys([date_col, "cusip", "gvkey", col]))
    train = df.loc[train_mask, meta]
    test = df.loc[test_mask, meta]

    y_train = project(train, train[col], level, date_col)
    y_test = project(test, test[col], level, date_col)

    X_train = df.loc[train_mask, list(predictors)]
    cols = select_predictors(X_train)
    if len(cols) < len(predictors):
        X_train = X_train[cols]
    X_test = df.loc[test_mask, cols]

    n_nan = int(X_train.isna().sum().sum()) + int(X_test.isna().sum().sum())
    check(n_nan == 0, f"{level.name} h={horizon} fold {fold.fit_month}: predictors are finite ({n_nan} NaN)")

    train_months = pd.PeriodIndex(pd.to_datetime(train[date_col]), freq="M")

    model = estimator_builder()
    if getattr(model, "needs_months", False):
        model.fit(X_train, y_train, months=train_months, horizon=horizon)
    else:
        model.fit(X_train, y_train)

    yhat = pd.Series(np.asarray(model.predict(X_test), dtype="float64"), index=test.index)
    yhat_proj = project(test, yhat, level, date_col)

    predictions = pd.DataFrame({
        date_col: pd.PeriodIndex(pd.to_datetime(test[date_col]), freq="M"),
        "cusip": test["cusip"].astype(str),
        "gvkey": test["gvkey"],
        "fit_month": fold.fit_month,
        "y": y_test.to_numpy(),
        "yhat": yhat.to_numpy(),
        "yhat_proj": yhat_proj.to_numpy(),
        "ybar": float(y_train.mean()),
    }, index=test.index)

    diagnostics = {
        "fit_month": str(fold.fit_month),
        "n_train": len(train),
        "n_test": len(test),
        "n_predictors": len(cols),
        "train_start": str(train_months.min()),
        "train_end": str(train_months.max()),
        **{f"param_{k}": v for k, v in getattr(model, "best_params_", {}).items()},
    }
    return predictions, diagnostics


def run_level(
        df: pd.DataFrame,
        level: Level,
        horizon: int,
        oos_start: pd.Period,
        refit_freq: int,
        predictors: Sequence[str],
        estimator_builder: Callable[[], object],
        common_sample: bool,
        on_fold: Callable[[dict], None] | None = None,
        date_col: str = "date",
) -> tuple[pd.DataFrame, list[dict], list[Fold], np.ndarray]:
    """Runs one level at one horizon over the whole OOS range and concatenates the folds."""

    months = pd.PeriodIndex(pd.to_datetime(df[date_col]), freq="M").unique().sort_values()
    folds = _generate(months, oos_start, horizon, refit_freq, embargo=horizon)
    eligible = eligible_rows(df, level, horizon, common_sample, date_col)

    parts, diagnostics = [], []
    for fold in folds:
        p, d = fit_predict_fold(df, fold, level, horizon, predictors, estimator_builder, eligible, date_col)
        parts.append(p)
        diagnostics.append(d)
        if on_fold is not None:
            on_fold(d)

    return pd.concat(parts, axis=0), diagnostics, folds, eligible


# %% Scoring

def fold_sums(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    """Sufficient statistics for OOS R^2 under any constant benchmark."""

    y = np.asarray(y, dtype="float64")
    yhat = np.asarray(yhat, dtype="float64")
    if len(y) == 0:
        raise ValueError("fold_sums: empty input")
    if np.isnan(y).any() or np.isnan(yhat).any():
        raise ValueError("fold_sums: y or yhat contains NaN values")
    return {
        "n": float(len(y)),
        "sum_r": float(y.sum()),
        "sum_r2": float((y ** 2).sum()),
        "sse": float(((y - yhat) ** 2).sum()),
    }


def r2_from_sums(sums: dict[str, float], benchmark: float) -> float:
    """OOS R^2 = 1 - SSE / sum((r - benchmark)^2), from accumulated sums."""

    sst = sums["sum_r2"] - 2 * benchmark * sums["sum_r"] + sums["n"] * benchmark ** 2
    if sst <= 0:
        raise ValueError(f"r2_from_sums: SST is non-positive ({sst}) for benchmark {benchmark}")
    return 1.0 - sums["sse"] / sst


def score(preds: pd.DataFrame, yhat_col: str = "yhat") -> dict[str, float]:
    """
    OOS R^2 of one cell's predictions under three benchmarks:

        r2_zero : against a zero forecast (Gu-Kelly-Xiu convention; the primary number)
        r2_hist : against the prevailing training-window mean `ybar` -- a genuinely out-of-sample
                  competitor, unlike the midterm's demeaned convention, which used the OOS mean
        r2_oosmean : against the realized OOS mean (the midterm's "demeaned" convention; look-ahead,
                     reported only for continuity with the midterm table)
    """

    y = preds["y"].to_numpy(dtype="float64")
    yhat = preds[yhat_col].to_numpy(dtype="float64")
    sums = fold_sums(y, yhat)
    sse_hist = float(((y - preds["ybar"].to_numpy(dtype="float64")) ** 2).sum())
    return {
        "r2_zero": r2_from_sums(sums, 0.0),
        "r2_hist": 1.0 - sums["sse"] / sse_hist,
        "r2_oosmean": r2_from_sums(sums, sums["sum_r"] / sums["n"]),
        "n": int(sums["n"]),
    }


def score_by_period(
        preds: pd.DataFrame,
        periods: Sequence[tuple[str, str, str]],
        yhat_col: str = "yhat",
        date_col: str = "date",
) -> pd.DataFrame:
    """
    `score` within each (name, first, last) block of signal months, both ends inclusive.

    Rows are assigned by forecast origin, not by when the label is realised: at h=12 a 2019
    signal's label already contains 2020 returns. A period holding no rows scores NaN with n=0.
    Rows outside every period are not scored; callers check that the n column sums to len(preds).
    """

    months = pd.PeriodIndex(preds[date_col].astype(str), freq="M")
    rows = []
    for name, first, last in periods:
        mask = np.asarray((months >= pd.Period(first, freq="M")) & (months <= pd.Period(last, freq="M")))
        if mask.any():
            s = score(preds.loc[mask], yhat_col)
        else:
            s = {"r2_zero": np.nan, "r2_hist": np.nan, "r2_oosmean": np.nan, "n": 0}
        rows.append({"period": name, "first": first, "last": last,
                     "n_months": int(months[mask].nunique()), **s})
    return pd.DataFrame(rows)
