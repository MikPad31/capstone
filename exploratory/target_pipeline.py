"""
Checks whether one rung of the decomposition, fitted fold by fold with the embargo, produces an OOS R^2.
Checks embargo cost on a splitter that leaks.

RUN: DSE4101_DATA_DIR=... python3 -m exploratory.target_pipeline
"""
# %% Setup
from __future__ import annotations

import resource
import sys
import time
from typing import NamedTuple, Callable, Sequence
import numpy as np
import pandas as pd

from src.data import PREDICTOR_COLS, load_panel
from src.targets import forward_return, label_coverage
from src.decompose import _cast_keys, issuer_month_stats, demean
from src.splits import Fold, generate_folds, _generate, rows_for_months, leaked_months, label_months
from src.metrics import winsorize_cross_section, oos_r2
from src.pipeline import Estimator, ols_builder
from src.config import OUTPUT_DIR

# Constants
RET_COLS = ("retx_realized_return", "retxrf_realized_return")  # L0 vs L1/L2
HORIZONS = (1, 3, 12)
EXPECTED_LABELLED_SHARE = {1: 1.0, 3: 0.962, 12: 0.7985}
PROVISIONAL_OOS_START = pd.Period("2015-01", freq="M")

# %% Helper functions

_failures: list[str] = []

def log(
        msg: str
) -> None:
    """
    Prints a timestamped progress line.

    Parameters
    ----------
    msg : str
        Text to print.

    Returns
    -------
    None
    """

    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check(
        ok: bool,
        msg: str
) -> bool:
    """
    Records a pass/fail as a [verify] line without aborting the run.
    Failures accumulate in `_failures` and are turned into a non-zero exit by `finish`.

    Parameters
    ----------
    ok : bool
        Whether the check passed.
    msg : str
        What was checked, stated with the measured quantity in it.

    Returns
    -------
    bool
        `ok`, unchanged, for use in a conditional.
    """

    print(f"[verify] {'PASS' if ok else 'FAIL'} — {msg}", flush=True)
    if not ok:
        _failures.append(msg)
    return ok


def report(
        msg: str
) -> None:
    """
    Prints a measured quantity as a [report] line. Never asserted.

    Parameters
    ----------
    msg : str
        The measurement, including the unit it is denominated in.

    Returns
    -------
    None
    """

    print(f"[report] {msg}", flush=True)


def rss_gb() -> float:
    """
    Returns the peak resident set size of this process in gigabytes.

    Returns
    -------
    float
        Peak RSS in GB.
    """

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on Darwin, KB on Linux
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6


def finish(
        start: float
) -> None:
    """
    Closes out a run: prints elapsed time and peak memory, then exits non-zero if
    any `check()` failed.

    Under a kernel this reports and, on failure, raises `SystemExit` — which the
    kernel surfaces as a traceback and survives, so the loaded panel is not lost.

    Parameters
    ----------
    start : float
        The `time.time()` recorded when the run began.

    Returns
    -------
    None
    """

    log(f"done in {time.time() - start:.1f}s. peak rss={rss_gb():.2f}GB")
    if _failures:
        log(f"{len(_failures)} check(s) failed")
        sys.exit(1)


# %% Loading

def load_df_for_pipeline()-> pd.DataFrame:
    """
    Loads the panel with predictors and targets, casts predictors to float32, and adds the forward return label.

    Returns
    -------
    pd.DataFrame
        The loaded panel with predictors and forward return label.
    """

    df = load_panel(cols=["date", "cusip", "gvkey", *PREDICTOR_COLS],
                    targets=("retx", "retxrf"))

    before = df[PREDICTOR_COLS].memory_usage(deep=True).sum() / 1e6
    df[PREDICTOR_COLS] = df[PREDICTOR_COLS].astype("float32")
    after = df[PREDICTOR_COLS].memory_usage(deep=True).sum() / 1e6
    report(f"predictor block: {before:.1f}MB -> {after:.1f}MB (float32 cast)")

    months = pd.PeriodIndex(pd.to_datetime(df["date"]), freq="M")

    # Build every label once over the whole panel
    for ret_col in RET_COLS:
        for horizon in HORIZONS:
            label_col = f"{ret_col}_fwd{horizon}"
            raw = forward_return(df, ret_col, horizon)
            notna = raw.notna()
            temp = pd.DataFrame({
                label_col: raw.loc[notna],
                "_month": months[notna.to_numpy()]
            })
            winsorized = winsorize_cross_section(
                df = temp,
                value_col = label_col,
                key_cols = ["_month"],
                lower = 0.01,
                upper = 0.99
            )

            df[label_col] = np.nan
            df.loc[notna, label_col] = winsorized

    log(f"panel shape: {df.shape}")
    report(f"loaded {len(df)} rows, {len(df.columns)} columns, {df.memory_usage(deep=True).sum() / 1e6:.1f}MB")

    return df
# %% Levels

class Level(NamedTuple):
    name: str # "L0", "L1", "L2"
    target_col: str # "retx_realized_return", "retxrf_realized_return"
    demean_target: bool
    demean_predictors: bool
    key: str | None # "gvkey"; None for L0/L1
    min_group_size: int = 2


L0 = Level(name="L0", target_col="retxrf_realized_return", demean_target=False, demean_predictors=False, key=None)
L1 = Level(name="L1", target_col="retx_realized_return",   demean_target=False, demean_predictors=False, key=None)
L2 = Level(name="L2", target_col="retx_realized_return",   demean_target=True,  demean_predictors=False, key="gvkey")

# %% Harness

def eligible_rows(
        df: pd. DataFrame,
        level: Level,
        label_col: str,
        date_col: str = "date",
) -> np.ndarray:
    """
    Boolean mask of the rows a level can use.

    L0/L1: rows with a non-missing label.
    L2: drops issuer_months holding fewer than `level.min_group_size` bonds since a demeaned singleton is zero.

    Note: L0/L1/L2 are **not** evaluated on the same rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to filter.
    level : Level
        The level of the decomposition (L0, L1, or L2).
    label_col : str
        Column name of label to check.
    date_col : str, optional
        Column name of date to check (default is "date").
    
    Returns
    -------
    np.ndarray
        Boolean mask, shape (len(df),).

    Raises
    ------
    ValueError
        If level demeans but 'level.key' is None, or if the mask would be all False.
    """

    mask = df[label_col].notna()
    if level.demean_target:
        if level.key is None:
            raise ValueError(f"Level {level.name} demeans but has no key set")

        key_notna = df[level.key].notna().to_numpy()
        mask = mask & key_notna

        keys = pd.Series(np.nan, index=df.index, dtype="object")
        if len(df.loc[key_notna]) > 0:
            keys.loc[key_notna] = _cast_keys(df.loc[key_notna], [level.key])[0].to_numpy()

        months = pd.PeriodIndex(pd.to_datetime(df[date_col]), freq="M")

        # Give issuer_month_stats an explicit Period[M] column instead of falling back to the raw datetime64
        # (previous version returns all NaN)
        stats_input = df.loc[mask].assign(_month=months[mask.to_numpy()])
        group_size = issuer_month_stats(
            df=stats_input,
            key=level.key,
            time="_month",
        ).set_index([level.key, "_month"])["n_bonds"]

        pair_index = pd.MultiIndex.from_arrays([keys, months], names=[level.key, "_month"])
        n_bonds = group_size.reindex(pair_index).to_numpy() # strip MultiIndex to align with df.index

        mask = mask.to_numpy() & (n_bonds >= level.min_group_size)

    else:
        mask = mask.to_numpy()

    if not mask.any():
        raise ValueError(f"eligible_rows: mask is all False for level {level.name}")

    return mask

# %% Scoring
def split_fold_frames(
        df: pd.DataFrame,
        fold: Fold,
        level: Level,
        label_col: str,
        eligible: np.ndarray,
        date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train and test splits for a fold, restricted to the rows eligible for the level.

    Use `rows_for_months` to get the row indices for the train and test months, then intersect with the eligibility mask.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to split.
    fold : Fold
        Fold to split on.
    level : Level
        The level of the decomposition (L0, L1, or L2).
    label_col : str
        Column name of label to check.
    eligible : np.ndarray
        Boolean mask of eligible rows for the level.
    date_col : str, optional
        Column name of date to check (default is "date").

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train, test), each a row-subset of `df` with original index preserved.

    Raises
    ------
    ValueError
        If either slice is empty after intersecting with eligibility mask.
    """

    train_mask = rows_for_months(df, fold.train_months, date_col) & eligible
    test_mask = rows_for_months(df, fold.test_months, date_col) & eligible

    if not train_mask.any():
        raise ValueError(f"split_fold_frames: empty train slice for fold {fold})")

    if not test_mask.any():
        raise ValueError(f"split_fold_frames: empty test slice for fold {fold})")

    return df.loc[train_mask], df.loc[test_mask]

def demean_fold_labels(
        train: pd.DataFrame,
        test: pd.DataFrame,
        level: Level,
        label_col: str,
) -> tuple[pd.Series, pd.Series]:
    """
    `label_col`, within (level.key, month) groups, demeaned on each side of the fold
    if `level.demean_target` is True; otherwise returns untouched.

    Each side is demeaned using only its own rows, so the test side is not contaminated by the train side.

    Parameters
    ----------    
    train : pd.DataFrame
        Training dataframe.
    test : pd.DataFrame
        Testing dataframe.
    level : Level
        The level of the decomposition (L0, L1, or L2).
    label_col : str
        Column name of label to demean.
    
    Returns
    -------
    tuple[pd.Series, pd.Series]
        (y_train, y_test), indexed similarly to `train`/`test`
    """

    if not level.demean_target:
        return train[label_col], test[label_col]

    assert level.key is not None

    train = train.dropna(subset=[level.key])
    test = test.dropna(subset=[level.key])

    train_months = pd.PeriodIndex(pd.to_datetime(train["date"]), freq="M")
    test_months = pd.PeriodIndex(pd.to_datetime(test["date"]), freq="M")

    y_train = demean(
        df=train.assign(_month=train_months),
        value=label_col,
        by=[level.key, "_month"],
    )

    y_test = demean(
        df=test.assign(_month=test_months),
        value=label_col,
        by=[level.key, "_month"],
    )

    return y_train, y_test


def select_predictors(
        train: pd.DataFrame,
        predictors: Sequence[str],
) -> list[str]:
    """
    Drops predictors with zero-variance, measured on `train` set only, to avoid leaking information from the test set.
    Resulting list of predictors is used for both train and test sets.

    Parameters
    ----------
    train : pd.DataFrame
        Training dataframe.
    predictors : Sequence[str]
        List of predictor column names to check for zero variance.

    Returns
    -------
    list[str]
        Subset of non-zero-variance predictors, in the same order as `predictors`.
    """

    variances = train[list(predictors)].var(axis=0, skipna=True)
    dropped = [pred for pred in predictors if not (variances.get(pred, 0.0) > 0.0)]
    if dropped:
        report(f"dropped {len(dropped)} zero-variance predictors: {dropped}")

    return [pred for pred in predictors if pred not in dropped]

def fit_predict_fold(
        df: pd.DataFrame,
        fold: Fold,
        level: Level,
        label_col: str,
        predictors: Sequence[str],
        estimator_builder: Callable[[], Estimator],
        eligible: np.ndarray,
        date_col: str = "date",
) -> tuple[pd.DataFrame, dict]:
    """
    Fits one fold on its training months and predicts its test months.

    Composes in the following order:
        1. `split_fold_frames` to get train/test slices
        2. `demean_fold_labels` to demean the label if `level.demean_target` is True
        3. `select_predictors` to drop zero-variance predictors on the train set
        4. `estimator_builder().fit`/`.predict` to fit and prdict the fold.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    fold : Fold
        Fold to fit and predict.
    level : Level
        The level of the decomposition (L0, L1, or L2).
    label_col : str
        Column name of label to predict.
    predictors : Sequence[str]
        List of predictor column names to use for fitting.
    estimator_builder : Callable[[], Estimator], optional
        Function that returns a new estimator instance.
    eligible : np.ndarray
        Boolean mask of eligible rows for the level.
    date_col : str, optional
        Column name of date to check (default is "date").
    
    Returns
    -------
    tuple[pd.DataFrame, dict]
        A tuple containing:
        - predictions :pd.DataFrame
            One row per test row, with columns:
                - date (Period[M])
                - cusip
                - fit_month (Period[M], signal month per fold)
                - y (float64)
                - yhat (float64)
        - diagnostics : dict
            A dictionary containing additional information about the fold.
                - fit_month
                - n_train
                - train_start
                - train_end
        
        Long format so folds concatenate and scoring does not need fold count
    
    Raises
    ------
    ValueError
        if either train or test slice is empty after eligibility filtering.
    """

    train, test = split_fold_frames(df, fold, level, label_col, eligible, date_col)
    y_train, y_test = demean_fold_labels(train, test, level, label_col)

    # Restrict train/test to the same rows as y_train/y_test, which may have dropped some rows due to demeaning
    train = train.loc[y_train.index]
    test = test.loc[y_test.index]

    cols = select_predictors(train, predictors)

    X_train = train[cols]
    X_test = test[cols]

    n_nan_train = int(X_train.isna().sum().sum())
    n_nan_test = int(X_test.isna().sum().sum())
    check(
        n_nan_train == 0 and n_nan_test == 0,
        f"{level.name} {label_col} fold {fold.fit_month}: predictors are finite "
        f"({n_nan_train} NaN in train, {n_nan_test} in test)"
    )

    model = estimator_builder()
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)

    predictions = pd.DataFrame({
        date_col: pd.PeriodIndex(pd.to_datetime(test[date_col]), freq="M"),
        "cusip": test["cusip"],
        "fit_month": fold.fit_month,
        "y": y_test,
        "yhat": yhat,
    }, index=test.index)

    train_months = pd.PeriodIndex(pd.to_datetime(train[date_col]), freq="M")
    diagnostics = {
        "fit_month": fold.fit_month,
        "n_train": len(train),
        "train_start": train_months.min(),
        "train_end": train_months.max(),
    }

    return predictions, diagnostics

def run_level(
        df: pd.DataFrame,
        level: Level,
        horizon: int,
        oos_start: pd.Period,
        refit_freq: int,
        predictors: Sequence[str],
        estimator_builder: Callable[[], Estimator],
        embargo: int | None = None,
        date_col: str = "date",
) -> tuple[pd.DataFrame, list[dict], list[Fold], np.ndarray]:
    """
    Runs one level at one horizon over the whole OOS range and concatenates the folds.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    level : Level
        The level of the decomposition (L0, L1, or L2).
    horizon : int
        Horizon in months for the forward return label.
    oos_start : pd.Period
        The start of the out-of-sample period.
    refit_freq : int
        The frequency of refitting the model.
    predictors : Sequence[str]
        List of predictor column names to use for fitting.
    estimator_builder : Callable[[], Estimator], optional
        Function that returns a new estimator instance.
    eligible : np.ndarray
        Boolean mask of eligible rows for the level.
    embargo : int | None, optional
        The number of months to embargo after each refit (default is None).
        `embargo = None` means `horizon`; 
        `embargo = 1` reproduces textbook definition of expanding window which leaks the next month into the training set.
    date_col : str, optional
        Column name of date to check (default is "date").

    Returns
    -------
    tuple[pd.DataFrame, list[dict], list[Fold], np.ndarray]
        A tuple containing:
        - predictions : pd.DataFrame
        - diagnostics : list[dict]
            fold-level diagnostics, one dict per fold in fold order from `fit_predict_fold`.
        - folds : list[Fold]
            The list of folds used in the pipeline.
        - eligible : np.ndarray
            The boolean mask of eligible rows for the level.
    """

    months = pd.PeriodIndex(pd.to_datetime(df[date_col]), freq="M").unique().sort_values()
    folds = _generate(
        months,
        oos_start,
        horizon,
        refit_freq,
        embargo=horizon if embargo is None else embargo,
    )

    label_col = f"{level.target_col}_fwd{horizon}"
    eligible = eligible_rows(df, level, label_col, date_col)

    results = [
        fit_predict_fold(
            df=df,
            fold=fold,
            level=level,
            label_col=label_col,
            predictors=predictors,
            estimator_builder=estimator_builder,
            eligible=eligible,
            date_col=date_col,
        )
        for fold in folds
    ]
    parts, diagnostics = zip(*results)
    return pd.concat(parts, axis=0), list(diagnostics), folds, eligible


# %% Validation 1: check missingness in data

def v1_check_missingness(df: pd.DataFrame) -> None:
    """
    Checks for missing values in the predictor columns of the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to check for missing values.

    Returns
    -------
    None
    """
    # Cell-level checks
    n_cells = df[PREDICTOR_COLS].size
    n_missing = df[PREDICTOR_COLS].isna().sum().sum()
    n_zero = (df[PREDICTOR_COLS] == 0).to_numpy().sum()

    # Per-column check
    col_missing = df[PREDICTOR_COLS].isna().sum()
    n_col_missing = (col_missing > 0).sum()

    worst_10 = col_missing.sort_values(ascending=False).head(10)

    # Check missingness by month
    months = pd.PeriodIndex(pd.to_datetime(df["date"]), freq="M")
    mean_gaps = df[PREDICTOR_COLS].isna().any(axis=1).groupby(months).mean()

    # Report
    report(f"predictor missingness: {100 * n_missing / n_cells:.4f}% ({n_missing:,} of {n_cells:,} cells NaN); {n_col_missing} columns affected.")
    for col, n in worst_10.items():
        report(f"column {col!r} has {n:,} NaN values ({100 * n / len(df):.4f}% of rows)")

    for date, gap in mean_gaps.items():
        report(f"month {date} has {100 * gap:.4f}% missing values")

    report(f"Predictor exact zeros: {100 * n_zero / n_cells:.4f}% ({n_zero:,} of {n_cells:,} cells are exactly zero)")

    if n_missing == 0:
        log("No missing predictors; imputation (impute_fit/impute_apply) is not needed")
    else:
        log(f"Missing predictors found: {100 * n_missing / n_cells:.4f}% missing; imputation required")

def fold_sums(
        preds: pd.DataFrame,
) -> dict[str, float]:
    """ Sufficient statistics for a fold: n, sum_r, sum_r2, sum_p, sum_rp, sse.
    
    Parameters
    ----------
    preds : pd.DataFrame
        Dataframe with columns 'y' and 'yhat' for the fold.

    Returns
    -------
    dict[str, float]
        Dictionary containing the sufficient statistics for the fold.

    Raises
    ------
    ValueError
        If the input dataframe does not contain 'y' or 'yhat' columns
    """

    if len(preds) ==0:
        raise ValueError("fold_sums: input dataframe is empty")

    y = preds["y"].to_numpy(dtype="float64")
    yhat = preds["yhat"].to_numpy(dtype="float64")

    if np.isnan(y).any() or np.isnan(yhat).any():
        raise ValueError("fold_sums: y or yhat contains NaN values")

    residuals = y - yhat
    return {
        "n": float(len(preds)),
        "sum_r": float(y.sum()),
        "sum_r2": float((y ** 2).sum()),
        "sum_p": float(yhat.sum()),
        "sum_rp": float((y * yhat).sum()),
        "sse": float((residuals ** 2).sum()),
    }

def r2_from_sums(
    sums: dict[str, float],
    benchmark: float,
) -> float:
    """
    Computes OOS R^2 from accumulated sums for any benchmark mean.
    
    SSE = sum((r - p)^2)
    SST = sum_r2 - 2 * benchmark * sum_r + n * benchmark^2
    R^2 = 1 - SSE / SST

    benchmark = 0.0 (zero-benchmark as used by Gu-Kelly-Xiu)

    Parameters
    ----------
    sums : dict[str, float]
        Dictionary containing the sufficient statistics for the fold.
    benchmark : float
        The benchmark mean to compare against.

    Returns
    -------
    float
        The computed OOS R^2.

    Raises
    ------
    ValueError
        If SST <= 0, which indicates a degenerate y where zero variance around benchmark.
    """

    sst = sums["sum_r2"] - 2 * benchmark * sums["sum_r"] + sums["n"] * benchmark ** 2
    if sst <= 0:
        raise ValueError(f"r2_from_sums: SST is non-positive ({sst}) for benchmark {benchmark}")

    return 1.0 - sums["sse"] / sst

def check_r2_cross_reference(
        preds: pd.DataFrame,
        benchmark: float = 0.0,
) -> None:
    """
    Checks that the R^2 computed from fold_sums matches the R^2 computed directly from the shipped predictions.
    
    Parameters
    ----------
    preds : pd.DataFrame
        Dataframe with columns 'y' and 'yhat' for the fold.
    benchmark : float, optional
        The benchmark mean to compare against (default is 0.0).
    """

    sums = fold_sums(preds)
    r2_sums = r2_from_sums(sums, benchmark)

    r2_direct = oos_r2(preds["y"].to_numpy(), preds["yhat"].to_numpy(), benchmark)

    check(
        np.isclose(r2_sums, r2_direct, atol = 1e-10),
        f"r2_from_sums ({r2_sums:.10f}) matches oos_r2 ({r2_direct:.10f}) for benchmark {benchmark}"
    )   

# %% Validation 2
def v2_check_labels_and_eligibility(
        df: pd.DataFrame,
        levels: Sequence[Level] = None,
        horizons: Sequence[int] = HORIZONS,
        ret_cols: Sequence[str] = RET_COLS,
        date_col: str = "date",
) -> None:
    """
    Panel-level check on labels, winsorization, and eligibility.
    
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    levels : Sequence[Level], optional
        The levels of the decomposition (L0, L1, or L2).
    horizons : Sequence[int], optional
        The horizons in months for the forward return label.
    ret_cols : Sequence[str], optional
        The column names of the return columns.
    date_col : str, optional
        Column name of date to check (default is "date").

    Returns
    -------
    None
    """

    if levels is None:
        levels = [L0, L1, L2]

    months = pd.PeriodIndex(pd.to_datetime(df[date_col]), freq="M")
    n = len(df)

    raw_labels = {
        (ret_col, horizon): forward_return(df, ret_col, horizon)
        for ret_col in ret_cols
        for horizon in horizons
    }


    # label coverage check
    for ret_col in ret_cols:
        for horizon in horizons:
            label_col = f"{ret_col}_fwd{horizon}"
            notna = df[label_col].notna()

            by_month = notna.groupby(months).sum().sort_index()
            zero_months = set(by_month[by_month == 0].index)
            expected_tail = set(by_month.index[-(horizon - 1):]) if horizon > 1 else set()

            check(
                zero_months == expected_tail,
                f"{ret_col} h={horizon}: zero label months {zero_months} match expected tail {expected_tail} with {horizon - 1} panel months"
            )

            observed_share = notna.mean()
            expected_share = EXPECTED_LABELLED_SHARE[horizon] # empirical from previous runs
            check(
                abs(observed_share - expected_share) < 1e-3,
                f"{ret_col} h={horizon}: labelled share {observed_share:.4f} matches expected {expected_share:.4f}"
            )

            report(f"{ret_col} h={horizon}: labelled share = {observed_share:.4f}")

    # winsorization check
    for ret_col in ret_cols:
        for horizon in horizons:
            label_col = f"{ret_col}_fwd{horizon}"
            notna = df[label_col].notna()
            raw = raw_labels[(ret_col, horizon)]

            temp = pd.DataFrame({
                "raw": raw.loc[notna],
                "winsorized": df.loc[notna, label_col],
                "_month": months[notna.to_numpy()],
            })

            expected_winsorized = winsorize_cross_section(
                df=temp.rename(columns={"raw": label_col}),
                value_col=label_col, key_cols=["_month"],
                lower=0.01,
                upper=0.99,
            )

            check(
                np.array_equal(expected_winsorized.to_numpy(), temp["winsorized"].to_numpy()),
                f"{ret_col} h={horizon}: stored label equals the raw label winsorized by formation month at 1/99"
            )

            upper_bounds = temp.groupby("_month")["winsorized"].max()
            check(
                upper_bounds.nunique() > 1,
                f"{ret_col} h={horizon}: winsorization bound varies across months "
                f"({upper_bounds.nunique()} distinct)"
            )

            clipped_share = (temp["winsorized"] != temp["raw"]).groupby(temp["_month"]).mean()
            bad_months = clipped_share[(clipped_share < 0.005) | (clipped_share > 0.05)]
            check(
                bad_months.empty,
                f"{ret_col} h={horizon}: clipped share per month stays near 2% "
                f"({len(bad_months)} months outside [0.5%, 5%])"
            )
            report(f"{ret_col} h={horizon}: median clipped share = {clipped_share.median():.4f}")

    # eligibility check
    for level in levels:
        for horizon in horizons:
            label_col = f"{level.target_col}_fwd{horizon}"
            eligible = eligible_rows(df, level, label_col, date_col)
            notna = df[label_col].notna().to_numpy()

            n_eligible = int(eligible.sum())
            report(f"{level.name} h={horizon}: |E| = {n_eligible:,} bond-months "
                   f"({100 * n_eligible / n:.1f}% of {n:,})")

            if not level.demean_target:
                check(
                    np.array_equal(eligible, notna),
                    f"{level.name} h={horizon}: eligibility exactly equals the non-missing label mask"
                )
            else:
                subset_ok = not (eligible & ~notna).any()
                check(subset_ok, f"{level.name} h={horizon}: eligibility is a subset of the label mask")

                key_notna = df[level.key].notna().to_numpy()
                mask = notna & key_notna

                keys = pd.Series(np.nan, index=df.index, dtype="object")
                if len(df.loc[key_notna]) > 0:
                    keys.loc[key_notna] = _cast_keys(df.loc[key_notna], [level.key])[0].to_numpy()
                    
                stats_input = df.loc[mask].assign(_month=months[mask])
                group_size = issuer_month_stats(df=stats_input, key=level.key, time="_month") \
                    .set_index([level.key, "_month"])["n_bonds"]

                pair_index = pd.MultiIndex.from_arrays([keys, months], names=[level.key, "_month"])
                n_bonds = group_size.reindex(pair_index).to_numpy()

                check(
                    np.all(n_bonds[eligible] >= level.min_group_size),
                    f"{level.name} h={horizon}: every surviving (key, month) group has >= {level.min_group_size} bonds")

                excluded_but_labelled = notna & ~eligible & ~np.isnan(n_bonds)

                check(
                    np.all(n_bonds[excluded_but_labelled] == 1),
                    f"{level.name} h={horizon}: every excluded-but-labelled group has exactly 1 bond"
                )

                n_keyless = int((notna & ~key_notna).sum())
                report(f"{level.name} h={horizon}: {n_keyless:,} labelled rows excluded for a missing key")


            # winsorization precedees restriction
            needed = [date_col, label_col] + ([level.key] if level.key else [])
            raw_label = raw_labels[(level.target_col, horizon)]
            df_raw_slim = df[needed].copy()
            df_raw_slim[label_col] = raw_label
            eligible_raw = eligible_rows(df_raw_slim, level, label_col, date_col)
            check(
                np.array_equal(eligible, eligible_raw),
                f"{level.name} h={horizon}: eligibility from raw label matches eligibility from winsorized label"
            )

# %% Validation 3

def v3_check_harness_honours_fold(
        folds: Sequence[Fold],
        diagnostics: Sequence[dict],
        preds: pd.DataFrame,
        level: Level,
        horizon: int,
        date_col: str = "date",
) -> None:
    """
    Confirms the harness does not reintroduce excluded months by selecting rows another way.

    Parameters
    ----------
    folds : Sequence[Fold]
        The folds generated by `run_level`, in fold order.
    diagnostics : Sequence[dict]
        Per-fold diagnostics from `run_level`, in fold order.
    preds : pd.DataFrame
        The predictions from `run_level`, in fold order.
    level : Level
        The level of the decomposition (L0, L1, or L2).
    horizon : int
        The horizon in months for the forward return label.
    date_col : str, optional
        Column name of date to check (default is "date").

    Returns
    -------
    None
    """
    if len(folds) != len(diagnostics):
        raise ValueError(f"v3: {len(folds)} folds but {len(diagnostics)} diagnostics")

    prev_n_train, prev_train_end = -1, None

    for i, (fold, diag) in enumerate(zip(folds, diagnostics)):
        tag = f"{level.name} h={horizon} fold {i}"
        train_months, test_months = set(fold.train_months), set(fold.test_months)

        # training containment 
        check(
            min(fold.train_months) <= diag["train_start"] and diag["train_end"] <= max(fold.train_months),
            f"{tag}: fitted rows' [train_start, train_end] within fold's train_months range"
        )

        predicted_months = set(preds.loc[preds["fit_month"] == fold.fit_month, date_col])
        check(predicted_months <= test_months, f"{tag}: predicted months subset of test_months")

        check(train_months.isdisjoint(test_months), f"{tag}: train_months, test_months disjoint")

        fitted_months = pd.period_range(diag["train_start"], diag["train_end"], freq="M")
        fitted_labels = set().union(*(label_months(m, horizon) for m in fitted_months))
        predicted_labels = set().union(*(label_months(m, horizon) for m in predicted_months))
        check(
            fitted_labels.isdisjoint(predicted_labels),
            f"{tag}: no calendar month is labelled by both a fitted and a predicted signal month"
        )

        check(
            diag["train_end"] + horizon == min(fold.test_months),
            f"{tag}: train_end + horizon == first test month (index-level embargo)"
        )

        check(diag["n_train"] >= prev_n_train, f"{tag}: n_train non-decreasing from previous fold")
        if prev_train_end is not None:
            check(diag["train_end"] > prev_train_end, f"{tag}: train_end strictly increases")
        prev_n_train, prev_train_end = diag["n_train"], diag["train_end"]

# %% Validation 4

def v4_check_predictions_partition(
        preds: pd.DataFrame,
        folds: Sequence[Fold],
        eligible: np.ndarray,
        df: pd.DataFrame,
        level: Level,
        horizon: int,
        date_col: str = "date",
) -> None:
    """
    Confirms the concatenated predictions from one `run_level` call partition the OOS range.

    Parameters
    ----------
    preds : pd.DataFrame
        The predictions from `run_level`, in fold order.
    folds : Sequence[Fold]
        The folds generated by `run_level`, in fold order.
    eligible : np.ndarray
        Eligibility mask for this level/horizon, aligned to `df`'s index.
    df : pd.DataFrame
        The original dataframe used to generate `preds`.
    level : Level
        The level of the decomposition (L0, L1, or L2).
    horizon : int
        The horizon in months for the forward return label.
    date_col : str, optional
        Column name of date to check (default is "date").

    Returns
    -------
    None
    """
    tag = f"{level.name} h={horizon}"

    n_dupes = preds.duplicated(subset=["date", "cusip"]).sum()
    check(n_dupes == 0, f"{tag}: no duplicated (date, cusip) pairs ({n_dupes} found)")

    union_test_months = set().union(*(set(f.test_months) for f in folds))
    check(
        set(preds["date"]) == union_test_months,
        f"{tag}: predicted months exactly equal the union of folds' test_months"
    )

    months = pd.PeriodIndex(pd.to_datetime(df[date_col]), freq="M")
    in_oos_range = months.isin(union_test_months)
    expected_n = int((eligible & in_oos_range).sum())

    check(
        len(preds) == expected_n,
        f"{tag}: len(preds) ({len(preds):,}) == |E ∩ OOS-range| ({expected_n:,}) — holds at "
        f"L2 only because eligibility already excludes the demeaning-drop rows"
    )

    row_gap = (preds["date"] - preds["fit_month"]).map(lambda p: p.n)
    check((row_gap >= horizon).all(), f"{tag}: every predicted row has date - fit_month >= horizon")

    first_gap_ok = all(
        (preds.loc[preds["fit_month"] == f.fit_month, "date"].min() - f.fit_month).n == horizon
        for f in folds
    )
    check(first_gap_ok, f"{tag}: each fold's first test month attains date - fit_month == horizon")

    check(preds[["y", "yhat"]].notna().all().all(), f"{tag}: no NaN in y or yhat")

# %% Validation 5
def v5_negative_control(
        df: pd.DataFrame,
        level: Level,
        horizon: int,
        oos_start: pd.Period,
        refit_freq: int,
        predictors: Sequence[str],
        estimator_builder: Callable[[], Estimator],
        benchmark: float = 0.0,
        date_col: str = "date",
) -> None:
    """
    Runs `level` at `horizon` twice:
        1. With the embargo set to `horizon` (no leakage)
        2. With the embargo set to 1 (leakage of the next month into training set)
    Compares the OOS R^2 of the two runs plus training row counts.

    Naive path also trains on `horizon`-1 extra months per fold.
    Report both counts to check if naive split is actually training on more rows than the embargoed split.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    level : Level
        The level of the decomposition (L0, L1, or L2).
    horizon : int
        Horizon in months for the forward return label.
    oos_start : pd.Period
        The start of the out-of-sample period.
    refit_freq : int
        The frequency of refitting the model.
    predictors : Sequence[str]
        List of predictor column names to use for fitting.
    estimator_builder : Callable[[], Estimator]
        Function that returns a new estimator instance.
    benchmark : float, optional
        The benchmark mean to compare against (default is 0.0).
    date_col : str, optional
        Column name of date to check (default is "date").

    Returns
    -------
    None
    """

    embargoed, embargoed_diagnostics, _, _ = run_level(
        df = df,
        level = level,
        horizon = horizon,
        oos_start = oos_start,
        refit_freq = refit_freq,
        predictors = predictors,
        estimator_builder = estimator_builder,
        embargo = None,
        date_col = date_col
    )

    naive, naive_diagnostics, _, _ = run_level(
        df = df,
        level = level,
        horizon = horizon,
        oos_start = oos_start,
        refit_freq = refit_freq,
        predictors = predictors,
        estimator_builder = estimator_builder,
        embargo = 1,
        date_col = date_col
    )

    r2_embargoed = r2_from_sums(fold_sums(embargoed), benchmark)
    r2_naive = r2_from_sums(fold_sums(naive), benchmark)

    n_train_embargoed = sum(d["n_train"] for d in embargoed_diagnostics)
    n_train_naive = sum(d["n_train"] for d in naive_diagnostics)

    report(f"Negative control results for level {level.name}, horizon {horizon} months: "
           f"Embargoed R^2 = {r2_embargoed:.4f}, Embargoed train rows = {n_train_embargoed:,}")
    report(f"Negative control results for level {level.name}, horizon {horizon} months: "
           f"Naive R^2 = {r2_naive:.4f}, Naive train rows = {n_train_naive:,}")

    if r2_naive < r2_embargoed:
        log(f"Negative control: Naive R^2 ({r2_naive:.4f}) does not exceed Embargoed R^2 "
            f"({r2_embargoed:.4f}). Either leak is not exploitable or the harness is unable "
            f"to measure it. Check the harness and data for issues.")

# %% Trace

def trace_first_fold(
        df: pd.DataFrame,
        level: Level,
        horizon: int,
        oos_start: pd.Period,
        refit_freq: int,
        predictors: Sequence[str],
        estimator_builders: dict[Callable[[], Estimator]],
        date_col: str = "date",
        n_forecasts: int = 5,
) -> None:
    """
    Traces the first fold of a level at a given horizon.
    Prints the following:
        1. training window: first and last signal month, bond-months, disctinct bonds
        2. forecast window and calendar months covered by its labels
        3. embargoed months
        4. predictor counter reaching the fit after dropping zero-variance predictors
        5. per estimator: hyperparameter grid (only for EN/RF), inner split formation, selected value
        6. first `n_forecasts` rows and their realized outcomes in the order that the pipeline produces them

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    level : Level
        The level of the decomposition (L0, L1, or L2).
    horizon : int
        Horizon in months for the forward return label.
    oos_start : pd.Period
        The start of the out-of-sample period.
    refit_freq : int
        The frequency of refitting the model.
    predictors : Sequence[str]
        List of predictor column names to use for fitting.
    estimator_builders : dict[Callable[[], Estimator]]
        Dictionary of functions that return new estimator instances.
    date_col : str, optional
        Column name of date to check (default is "date").
    n_forecasts : int, optional
        Number of forecast rows to print (default is 5).
    
    Returns
    -------
    None    
    """

    label_col = f"{level.target_col}_fwd{horizon}"
    eligible = eligible_rows(df, level, label_col, date_col)
    months = pd.PeriodIndex(pd.to_datetime(df[date_col]), freq="M").unique().sort_values()

    folds = _generate(months, oos_start, horizon, refit_freq, embargo=horizon)
    fold = folds[0]

    train, test = split_fold_frames(df, fold, level, label_col, eligible, date_col)
    y_train, y_test = demean_fold_labels(train, test, level, label_col)
    train = train.loc[y_train.index]
    test = test.loc[y_test.index]

    # Training window
    train_months_sorted = pd.PeriodIndex(pd.to_datetime(train[date_col]), freq="M")
    log(f"[trace] training window: {train_months_sorted.min()} to {train_months_sorted.max()}, {len(train):,} bond-months, {train['cusip'].nunique():,} distinct bonds")

    # Forecast window and label months
    log(f"[trace] fit_month (forecast origin): {fold.fit_month}")
    log(f"[trace] test_months (label coverage): {list(fold.test_months)}")

    # Embargoed months
    all_months_between = pd.period_range(min(fold.train_months), max(fold.test_months), freq="M")
    embargoed_months = sorted(set(all_months_between) - set(fold.train_months) - set(fold.test_months))
    log(f"[trace] embargoed months: {embargoed_months}")

    # Predictor count after zero-variance drop
    cols = select_predictors(train, predictors)
    log(f"[trace] predictors: {len(cols)} / {len(predictors)} predictors dropped for zero variance")

    X_train = train[cols]
    X_test = test[cols]

    # Per estimator:
    for name, builder in estimator_builders.items():
        log(f"[trace] estimator: {name}")
        model = builder()

        if hasattr(model, "param_grid"):
            log(f"[trace] hyperparameter grid: {model.param_grid}")
            inner_months = pd.PeriodIndex(pd.to_datetime(train[date_col]), freq="M").unique().sort_values()
            inner_folds = _generate(
                inner_months,
                oos_start=train_months_sorted.min(),
                horizon=horizon,
                refit_freq=refit_freq,
                embargo=horizon,
            )
            log(f"[trace] inner splits: {len(inner_folds)} folds")

        model.fit(X_train, y_train)

        if hasattr(model, "best_params_"):
            log(f"[trace] selected hyperparameters: {model.best_params_}")

        yhat = model.predict(X_test)

        # Print first n_forecasts rows and their realized outcomes
        for i in range(min(n_forecasts, len(test))):
            log(f"[trace]   forecast: cusip={test['cusip'].iloc[i]} yhat={yhat[i]:.6f}")
        for i in range(min(n_forecasts, len(test))):
            log(f"[trace]   realized: cusip={test['cusip'].iloc[i]} y={y_test.iloc[i]:.6f}")

# %% RUN - load once
_start = time.time()
df = load_df_for_pipeline()

# %% RUN Verifications
v1_check_missingness(df)
v2_check_labels_and_eligibility(df)

# %% RUN - one cell: L0, h=1, OLS
# REFIT_FREQ = 12  # settled choice, log it per decision-point #5 once this runs

# preds, diagnostics, folds, eligible = run_level(
#     df=df,
#     level=L0,
#     horizon=1,
#     oos_start=PROVISIONAL_OOS_START,
#     refit_freq=REFIT_FREQ,
#     predictors=PREDICTOR_COLS,
#     estimator_builder=ols_builder(),
# )

# v3_check_harness_honours_fold(
#     folds=folds, diagnostics=diagnostics, preds=preds, level=L0, horizon=1,
# )

# v4_check_predictions_partition(
#     preds=preds, folds=folds, eligible=eligible, df=df, level=L0, horizon=1,
# )


# sums = fold_sums(preds)
# r2 = r2_from_sums(sums, benchmark=0.0)
# report(f"L0 h=1 OOS R^2 (zero benchmark) = {r2:.4f}, n = {int(sums['n']):,}, folds = {len(diagnostics)}")

# %% Run 3x3 grid: L0/L1/L2 x h=1/3/12
# Run across all horizons and levels
REFIT_FREQ = 12
LEVELS = (L0, L1, L2)
BENCHMARK = 0.0
grid_rows: list[dict] = []

for horizon in HORIZONS:
    for level in LEVELS:
        log(f"=== {level.name} h={horizon} ===")
        preds, diagnostics, folds, eligible = run_level(
            df=df,
            level=level,
            horizon=horizon,
            oos_start=PROVISIONAL_OOS_START,
            refit_freq=REFIT_FREQ,
            predictors=PREDICTOR_COLS,
            estimator_builder=ols_builder(),
        )
        v3_check_harness_honours_fold(
            folds=folds,
            diagnostics=diagnostics,
            preds=preds,
            level=level,
            horizon=horizon,
        )

        v4_check_predictions_partition(
            preds=preds,
            folds=folds,
            eligible=eligible,
            df=df,
            level=level,
            horizon=horizon
        )

        sums = fold_sums(preds)
        r2 = r2_from_sums(sums, benchmark=BENCHMARK)

        if level is L0 and horizon == 1:
            check(np.isclose(r2, -0.0237, atol=5e-5) and int(sums["n"]) == 527_436,
                  f"L0 h=1 reproduces the single-cell run (R^2={r2:.4f}, n={int(sums['n']):,})")
        if level is L0 and horizon == 12:
            check(np.isclose(r2, -0.1390, atol=5e-5),
                  f"L0 h=12 reproduces the negative control's embargoed arm (R^2={r2:.4f})")

        report(f"{level.name} h={horizon} OOS R^2 (zero benchmark) = {r2:.4f}, n = {int(sums['n']):,}, folds = {len(diagnostics)}")

        check_r2_cross_reference(preds, benchmark=BENCHMARK)

        grid_rows.append({
            "level": level.name,
            "horizon": horizon,
            "r2": r2,
            "benchmark": BENCHMARK,
            **sums,
            "n": int(sums['n']),
            "n_folds": len(folds),
            "n_train": sum(d["n_train"] for d in diagnostics),
            "oos_start": str(PROVISIONAL_OOS_START),
            "refit_freq": REFIT_FREQ,
        })

        output_df = pd.DataFrame(grid_rows)
        output_df.to_csv(OUTPUT_DIR / "grid_sums.csv", index=False)

        del preds, diagnostics, folds, eligible  # free memory

grid = pd.DataFrame(grid_rows)
r2_table = grid.pivot(index="level", columns="horizon", values="r2")
n_table = grid.pivot(index="level", columns="horizon", values="n")
log("OOS R^2 (zero benchmark), levels x horizons:")
print(r2_table.to_string(float_format=lambda v: f"{v:.4f}"), flush=True)
log("n (scored bond-months):")
print(n_table.to_string(), flush=True)


v5_negative_control(
    df=df,
    level=L0,
    horizon=12,
    oos_start=PROVISIONAL_OOS_START,
    refit_freq=REFIT_FREQ,
    predictors=PREDICTOR_COLS,
    estimator_builder=ols_builder(),
    benchmark=BENCHMARK,
)

# %% RUN - close out
finish(_start)