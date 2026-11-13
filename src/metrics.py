"""
Holds cross-sectional winsorization and OOS R^2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

def winsorize_cross_section(
        df: pd.DataFrame,
        value_col: str,
        key_cols: list[str],
        lower: float = 0.01,
        upper: float = 0.99,
) -> pd.Series:
    """
    Clips a value column to its own cross-section's quantiles, group by group. 
    Not a global clip.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    value_col : str
        Name of the column to winsorize.
    key_cols : list[str]
        List of columns that define the cross-section (e.g., ['signal_date']).
    lower : float, optional
        Lower quantile for winsorization (default is 0.01).
    upper : float, optional
        Upper quantile for winsorization (default is 0.99).

    Returns
    -------
    pd.Series
        float64 series of winsorized values, with the same index as `df`.

    Raises
    ------
    ValueError
        If `value_col` has missing entries, or if `lower` is not below `upper`.
    """

    x = df[value_col].astype("float64")

    if x.isna().any():
        raise ValueError(
            f"winsorize_cross_section: {value_col} has {int(x.isna().sum()):,} missing entries"
        )

    if lower >= upper:
        raise ValueError(
            f"winsorize_cross_section: lower ({lower}) must be below upper ({upper})"
        )

    grouped = x.groupby([df[c] for c in key_cols], observed=True)
    lower_q = grouped.transform("quantile", lower)
    upper_q = grouped.transform("quantile", upper)

    return x.clip(lower=lower_q, upper=upper_q)

def oos_r2(
        r: pd.Series,
        p: pd.Series,
        benchmark: float,
) -> float:
    """
    Computes the out-of-sample R^2, taking benchmark as a parameter supplied by caller function.

    Parameters
    ----------
    r : pd.Series
        Realized returns.
    p : pd.Series
        Predicted returns, indexed like `r`.
    benchmark : float
        Benchmark return: 0.0 for the zero-benchmark convention 
        or mean of realized returns for the demeaned convention.

    Returns
    -------
    float
        The out-of-sample R^2 value.

    Raises
    ------
    ValueError
        If `r` and `p` differ in length or index, if either has missing entries, or if the
        benchmark leaves a zero denominator.
    """

    if len(r) != len(p):
        raise ValueError(f"oos_r2: r has {len(r):,} rows, p has {len(p):,}")

    if isinstance(r, pd.Series) and isinstance(p, pd.Series) and not r.index.equals(p.index):
        # Handle misaligned Series to ensure rows are compared correctly
        raise ValueError("oos_r2: r and p are not aligned on the same index")

    # Assert to numpy to handle NaN values
    r = np.asarray(r, dtype="float64")
    p = np.asarray(p, dtype="float64")

    if np.isnan(r).any() or np.isnan(p).any():
        raise ValueError(
            f"oos_r2: {int(np.isnan(r).sum()):,} missing in r, {int(np.isnan(p).sum()):,} in p"
        )

    sst = ((r - benchmark) ** 2).sum()

    if sst == 0.0:
        raise ValueError(f"oos_r2: every realized return equals the benchmark ({benchmark})")

    return float(1 - ((r - p) ** 2).sum() / sst)