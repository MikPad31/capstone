"""
Explores whether compounding `t+1 ... t+h` within `cusip` reproduces the shipped realized return at `h=1`.
Checks the cost in sample size at `h=3` and `h=12`.

Run: DSE4101_DATA_DIR=... python3 -m exploratory.targets_prototype
"""

# %% Setup
from __future__ import annotations

import resource
import sys
import time

import numpy as np
import pandas as pd

from src.config import HORIZONS
from src.data import load_panel, load_predictions

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

# %% Loading the returns

RETURN_COLS = ["date", "cusip"]

def load_returns() -> pd.DataFrame:
    """
    Loads the return slice of the panel, joined to the shipped realized returns.
    
    Returns
    -------
    pd.DataFrame
        One row per bond-month: `RETURN_COLS`, `retx_realized_return` and `retxrf_realized_return`.
    """

    log("loading return slice (date, cusip, retx, retxrf)")
    df = load_panel(cols=RETURN_COLS, targets=("retx", "retxrf"))
    log(f"  rows={len(df):,}  rss={rss_gb():.2f}GB")

    report(f"return slice: {len(df):,} rows")
    report(
        f"dates: {df['date'].min()} .. {df['date'].max()}  "
        f"({df['date'].nunique():,} distinct months)"
    )
    return df


# %% Compounding

def _gross_grid(
        df: pd.DataFrame,
        ret_col: str,
        by: str = "cusip",
        date_col: str = "date",
) -> pd.DataFrame:
    """
    Dense bond x month grid of gross returns, with NaN for missing months.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with `by`, `date_col` and `ret_col`.
    ret_col : str
        One-month forward return column to compound.
    by : str, optional
        Column to group by (default is "cusip").
    date_col : str, optional
        Signal-month column (default is "date").

    Returns
    -------
    pd.DataFrame : float64
        Dense grid of gross returns, indexed by `by` and `date_col`, with NaN for missing months.

    Raises
    ------
    KeyError
        If `by`, `date_col` or `ret_col` are not in `df`.
    ValueError
        If any (`by`, `date_col`) pair is duplicated in `df`.
    """

    if by not in df.columns:
        raise KeyError(f"_gross_grid: {by} not in df.columns")
    if date_col not in df.columns:
        raise KeyError(f"_gross_grid: {date_col} not in df.columns")
    if ret_col not in df.columns:
        raise KeyError(f"_gross_grid: {ret_col} not in df.columns")

    months = pd.PeriodIndex(pd.to_datetime(df[date_col]), freq="M")

    idx = pd.MultiIndex.from_arrays([df[by], months], names=[by, date_col])

    mask_dup = idx.duplicated()
    if mask_dup.any():
        raise ValueError(
            f"_gross_grid: {by} and {date_col} are not unique in df; "
            f"{mask_dup.sum():,} duplicates"
        )

    gross = 1 + df[ret_col].astype("float64")

    grid = pd.Series(gross.values, index = idx).unstack(date_col)

    full_months = pd.period_range(months.min(), months.max(), freq = "M")
    grid = grid.reindex(columns=full_months)

    return grid

def forward_return(
        df: pd.DataFrame,
        ret_col: str,
        horizon: int,
        by: str = "cusip",
        date_col: str = "date",
) -> pd.Series:
    """
    Cumulative return over `t, t+1, ..., t+h-1` aligned to signal month.
    `ret_col` is assumed to be one-month forward returns thus the window starts at `h=1`.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with `by`, `date_col` and `ret_col`.
    ret_col : str
        One-month forward return column to compound.
    horizon : int
        Number of months to compound forward. Must be >= 1.
    by : str, optional
        Column to group by (default is "cusip").
    date_col : str, optional
        Signal-month column (default is "date").

    Returns
    -------
    pd.Series : float64
        Cumulative return over `horizon` months, aligned to signal month, indexed like `df`.
        NaN where window crosses a gap or goes past the last month of the grid.

    Raises
    ------
    ValueError
        If `horizon` < 1.
    """

    if horizon < 1:
        raise ValueError(f"forward_return: horizon must be >= 1, got {horizon}")

    months = pd.PeriodIndex(pd.to_datetime(df[date_col]), freq="M")
    idx = pd.MultiIndex.from_arrays([df[by], months], names=[by, date_col])

    grid = _gross_grid(df, ret_col, by=by, date_col=date_col)

    cum_prod = grid.shift(0, axis=1)
    for k in range(1, horizon):
        cum_prod = cum_prod * grid.shift(-k, axis=1)

    label = cum_prod - 1.0

    res = label.stack().reindex(idx)
    res = pd.Series(res.values, index=df.index, name=f"{ret_col}_fwd{horizon}")

    return res
    
def check_h1(
        df: pd.DataFrame,
        ret_col: str,
        by: str = "cusip",
        date_col: str = "date",
) -> dict[str, int | float]:
    """
    Checks that `forward_return(..., horizon=1)` reproduces its own input exactly.
    Catches a leading shift(-1) which would make every label one month late.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with `by`, `date_col` and `ret_col`.
    ret_col : str
        One-month forward return column to compound.
    by : str, optional
        Column to group by (default is "cusip").
    date_col : str, optional
        Signal-month column (default is "date").
    
    Returns
    -------
    dict[str, int | float]
        n_compared   : rows where both sides are non-missing
        n_exact      : compared rows agreeing bit for bit
        max_abs_diff : largest absolute deviation over compared rows; expect 0.0
        n_ours_only  : forward_return present, input missing; expect 0.0
        n_theirs_only: input present, forward_return missing; expect 0.0 

    Never raises on a mismatch: the caller decides what a pass is, so a prototype can
    print the size of a gap and a test can assert on a deliberately shifted column.
    """
    fwd = forward_return(df, ret_col, 1, by=by, date_col=date_col)
    shipped = df[ret_col]

    fwd_valid = fwd.notna()
    shipped_valid = shipped.notna()
    both = fwd_valid & shipped_valid

    n_compared = int(both.sum())
    n_exact = int(
        np.isclose(
            fwd[both].to_numpy(), shipped[both].to_numpy(), atol=1e-12, rtol=0.0
        ).sum()
    )
    max_abs_diff = round(
        float((fwd[both] - shipped[both]).abs().max()) if n_compared else 0.0, 12
    )
    n_fwd_only = int((fwd_valid & ~shipped_valid).sum())
    n_shipped_only = int((~fwd_valid & shipped_valid).sum())

    check(
        max_abs_diff == 0.0,
        f"h=1 forward_return({ret_col}) reproduces its input exactly on "
        f"{n_compared} rows (max abs diff {max_abs_diff})",
    )
    check(
        n_fwd_only == 0 and n_shipped_only == 0,
        f"h=1 one-sided rows: {n_fwd_only} fwd-only, {n_shipped_only} shipped-only",
    )

    return {
        "n_compared": n_compared,
        "n_exact": n_exact,
        "max_abs_diff": max_abs_diff,
        "n_ours_only": n_fwd_only,
        "n_theirs_only": n_shipped_only,
    }


def check_grid_round_trip(
        df: pd.DataFrame,
        ret_col: str,
        by: str = "cusip",
        date_col: str = "date",
) -> dict[str, int]:
    """
    Checks that `_gross_grid` pivots and preserves every input row exactly once.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with `by`, `date_col` and `ret_col`.
    ret_col : str
        Return column to compound.
    by : str, optional
        Column to group by (default is "cusip").
    date_col : str, optional
        Signal-month column (default is "date").

    Returns
    -------
    dict[str, int]
        n_input         : len(df)
        n_grid          : non-null cells in the pivoted grid
        n_missing_key   : rows with a missing `by` or `date_col` value, which `_gross_grid` drops
    """

    n_input = len(df)
    grid = _gross_grid(df, ret_col, by=by, date_col=date_col)
    n_grid = int(grid.notna().sum().sum())
    n_missing_key = int(df[by].isna().sum() + pd.to_datetime(df[date_col]).isna().sum())

    check(n_grid == n_input, f"grid round-trip: {n_grid:,} non-null cells vs {n_input:,} input rows")

    return {
        "n_input": n_input,
        "n_grid": n_grid,
        "n_missing_key": n_missing_key,
    }

def label_coverage(
        fwd: pd.Series,
        df: pd.DataFrame,
        date_col: str = "date",
) -> pd.DataFrame:
    """
    Labelled and unlabelled bond-months per signal month.

    A rung's sample size is a function of the horizon, quietly: at h=12 every bond
    leaving the panel within a year of a month is unlabelled there. This is what makes
    "L0 at h=1 versus L0 at h=12" a comparison of two samples rather than an unexplained
    drop in n.

    Parameters
    ----------
    fwd : pd.Series
        Output of forward_return, indexed like `df`.
    df : pd.DataFrame
        The frame `fwd` was computed from.
    date_col : str, optional
        Signal-month column, by default "date".

    Returns
    -------
    pd.DataFrame
        One row per signal month, ascending. 
        Columns: 
            month (period[M]) : signal month
            n_rows (int64) : total rows in the month
            n_labelled (int64) : labelled rows in the month
            share_labelled (float64) : proportion of labelled rows in the month. Units are bond-months.
    """

    months = pd.PeriodIndex(pd.to_datetime(df[date_col]), freq="M")

    temp = pd.DataFrame({"month": months, "labelled": fwd.notna().to_numpy()})
    agg = temp.groupby("month", observed=True)["labelled"].agg(
        n_rows="size", n_labelled="sum"
    )
    agg["share_labelled"] = agg["n_labelled"] / agg["n_rows"]

    agg = agg.reset_index().sort_values("month").reset_index(drop=True)
    agg["n_rows"] = agg["n_rows"].astype("int64")
    agg["n_labelled"] = agg["n_labelled"].astype("int64")

    return agg

# %% RUN — load once

_start = time.time()
df = load_returns()

# %% RUN — V1a: grid round-trip

for ret_col in ("retx_realized_return", "retxrf_realized_return"):
    check_grid_round_trip(df, ret_col)

# %% RUN — V1b: check_h1

for ret_col in ("retx_realized_return", "retxrf_realized_return"):
    check_h1(df, ret_col)

# %% RUN — V3: coverage

for h in HORIZONS:
    for ret_col in ("retx_realized_return", "retxrf_realized_return"):
        fwd = forward_return(df, ret_col, h)
        cov = label_coverage(fwd, df)
        report(f"{ret_col} h={h}: {fwd.notna().mean():.1%} labelled overall")
        n_missing = int(df[ret_col].isna().sum())
        report(f"{ret_col}: {n_missing:,} missing after join ({n_missing/len(df):.4%})")

# %% RUN — close out

finish(_start)
