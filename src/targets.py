"""
Forward cumulative returns over t, t+1, ..., t+h-1, and the checks that validate them.
Input for `forward_return` is assumed to be one-month-forward returns, so the window starts at `h=1`.

    - `_gross_grid` is private: an implementation detail of `forward_return`, tested
      directly only from `test/test_targets.py`
    - `check_h1` never raises on a mismatch
"""


from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import load_panel

RETURN_COLS = ["date", "cusip"]

def load_returns() -> pd.DataFrame:
    """
    Loads the return slice of the panel, joined to the shipped realized returns.
    
    Returns
    -------
    pd.DataFrame
        One row per bond-month: `RETURN_COLS`, `retx_realized_return` and `retxrf_realized_return`.
    """

    df = load_panel(cols=RETURN_COLS, targets=("retx", "retxrf"))

    return df

def _gross_grid(
        df: pd.DataFrame,
        ret_col: str,
        by: str = "cusip",
        date_col: str = "date",
) -> pd.DataFrame:
    """
    Dense bond x month grid of gross returns, with NaN for missing months.
    The grid is the gap handling mechanism for absent bonds.
    `by` must be the bond identifier, not the issuer key.
    
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

    At horizon=1 this is an identity: label(t,1) == ret_col[t], no shift. 
    Compounding is a direct product and output is named f"{ret_col}_fwd{horizon}".
    Note: does not use log1p/expm1 because the log form turns values below 0.0 into NaN and sums it across the bond
    
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

    return {
        "n_compared": n_compared,
        "n_exact": n_exact,
        "max_abs_diff": max_abs_diff,
        "n_ours_only": n_fwd_only,
        "n_theirs_only": n_shipped_only,
    }


def label_coverage(
        fwd: pd.Series,
        df: pd.DataFrame,
        date_col: str = "date",
) -> pd.DataFrame:
    """
    Labelled and unlabelled bond-months per signal month.
    Enables clear comparisons of label coverage across horizons, models, and targets.

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