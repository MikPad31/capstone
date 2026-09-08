"""
Decomposition over a paramerised group key (either `gvkey` or `issuer_cusip`).
L2 demeaning of `retx` within issuer-month will take in group key as an argument.
"""

from __future__ import annotations

import pandas as pd

def _cast_keys(
        df: pd.DataFrame, 
        by: list[str],
) -> pd.DataFrame:
    """
    Casts any float group key to int64.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    by : list[str]
        List of columns to group by.

    Returns
    -------
    pd.DataFrame
        `df` unchanged if no key needs casting, 
        otherwise returns a shared copy of `df` with float group keys cast to int64.
    """

    return [
        df[c].astype("int64") if pd.api.types.is_float_dtype(df[c]) else df[c] for c in by
    ]


def issuer_month_stats(
        df: pd.DataFrame,
        key: str,
        entity: str | None = None,
        time: str = "date",
) -> pd.DataFrame:
    """
    Counts bonds, and optionally distinct entitites in each key-month.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    key : str
        Column to group by.
    entity : str | None, optional
        Column representing distinct entities, by default None.
    time : str, optional
        Column representing time, by default "date".

    Returns
    -------
    pd.DataFrame
        Dataframe with one row per (`time`, `key`) 
        and optionally `n_entities` when `entity` argument is provided.
    """

    grouped = df.groupby(_cast_keys(df, [time, key]), observed=True)

    stats = grouped.size().rename("n_bonds").to_frame()

    if entity is not None:
        stats["n_entities"] = grouped[entity].nunique()

    return stats.reset_index()

def demean(
        df: pd.DataFrame,
        value: str,
        by: list[str],
) -> pd.Series:
    """
    Demeans `value` within each `by` group.
    Note: restrict the frame to the subset of rows with non-missing `value` before calling this function.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    value : str
        Column to demean.
    by : list[str]
        Columns to group by.

    Returns
    -------
    pd.Series
        float64 series of demeaned values, with the same index as `df`.

    Raises
    ------
    ValueError
        If `value` has missing entries within any `by` group.
    """

    x = df[value].astype("float64")

    if x.isna().any():
        raise ValueError(f"demean: {value} has {int(x.isna().sum()):,} missing entries")

    return x - x.groupby(_cast_keys(df, by), observed=True).transform("mean")

def variance_decomp(
        df: pd.DataFrame,
        value: str,
        by: list[str],
) -> dict[str, float]:
    """
    Splits the total sum of squares of `value` into between- and within- group compnents.
    Each component is computed independently rather than by subtraction for a genuine check on the group means.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    value : str
        Column to decompose.
    by : list[str]
        Columns to group by.

    Returns
    -------
    dict[str, float]
        Dictionary with the following keys:
            - "ss_total"
            - "ss_between"
            - "ss_within"
            - "between_share"
            - "n_obs"
            - "n_groups"
            - "x_bar"
    """

    x = df[value].astype("float64")
    grouped = x.groupby(_cast_keys(df, by), observed=True)

    x_bar = x.mean()
    group_mean = grouped.transform("mean")

    ss_total = float(((x - x_bar) ** 2).sum())
    ss_between = float(((group_mean - x_bar) ** 2).sum())
    ss_within = float(((x - group_mean) ** 2).sum())

    return{
        "ss_total": ss_total,
        "ss_between": ss_between,
        "ss_within": ss_within,
        "between_share": ss_between / ss_total,
        "n_obs": int(len(x)),
        "n_groups": int(grouped.ngroups),
        "x_bar": float(x_bar),
    }