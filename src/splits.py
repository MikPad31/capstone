"""
Expanding-window train/test folds with an h-month embargo.

A signal dated `t` is labelled by returns over `t+1 … t+h`, so a training signal `s` is
admissible for a test block starting at `T` only if `s + h <= T`. Without that, training
labels overlap test labels by `h-1` months.

    - only `rows_for_months` touches the panel
    - other functions are just a function of synthetic list of months.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd


class Fold(NamedTuple):
    """
    One refitting fold of the months it learns from and predicts on.

    `fit_month` is the last training *signal* month, not the last calendar month whose
    returns the fit consumed where the two differ by `horizon`.
    Holding it in signal units keeps every field of this tuple in the same units,
    and makes the embargo readable as `fold.fit_month + horizon == fold.test_months[0]`.
    """

    fit_month: pd.Period
    train_months: list[pd.Period]
    test_months: list[pd.Period]


def label_months(
        signal: pd.Period,
        horizon: int,
) -> set[pd.Period]:
    """
    Returns the months that a signal dated `signal` is labelled by, given a horizon.
    Follows the verified convention that a signal dated `t` is labelled by the cumulative return
    over `t+1 … t+h`.

    Parameters
    ----------
    signal : pd.Period
        Month the signal is dated.
    horizon : int
        Number of months ahead that the signal is predicting.

    Returns
    -------
    set[pd.Period]
        Months that the signal is labelled by.
    """

    return {signal + i for i in range(1, horizon + 1)}


def leaked_months(
        fold: Fold,
        horizon: int,
) -> set[pd.Period]:
    """
    Union of label months over all train months in the fold, intersected with the union over test months.

    Parameters
    ----------
    fold : Fold
        Fold to check for leakage.
    horizon : int
        Number of months ahead that the signal is predicting.

    Returns
    -------
    set[pd.Period]
        Months that are both in the training window and labelled by the test window.
    """

    train_labels = set().union(*(label_months(m, horizon) for m in fold.train_months))
    test_labels = set().union(*(label_months(m, horizon) for m in fold.test_months))
    return train_labels.intersection(test_labels)


def _as_month_list(
        months: Sequence[pd.Period]
) -> list[pd.Period]:
    """
    Validates that the input is a non-empty, ascending, contiguous list of months and returns it as a list.

    Contiguity is a precondition rather than a relaxed invariant: on a calendar with
    holes the largest admissible training month can precede `T - h`, so the embargo gap
    would silently exceed `horizon`. The panel is gapless -- 245 months is exactly the
    inclusive span 2002-07 … 2022-11.

    Parameters
    ----------
    months : Sequence[pd.Period]
        Sequence of months to validate.

    Returns
    -------
    list[pd.Period]
        Validated list of months.

    Raises
    ------
    ValueError
        If the input is empty, not ascending, or not contiguous.
    """

    # coerced to a list to prevent raising TypeError
    # due to ambiguous handling of PeriodIndex in boolean contexts
    month_list = [pd.Period(m, freq="M") for m in months]

    if not month_list:
        raise ValueError("Input months sequence is empty.")

    if any(month_list[i] >= month_list[i + 1] for i in range(len(month_list) - 1)):
        raise ValueError("Input months sequence is not strictly ascending.")

    if any((month_list[i + 1] - month_list[i]).n != 1 for i in range(len(month_list) - 1)):
        raise ValueError("Input months sequence is not contiguous.")

    return month_list


def _generate(
        months: Sequence[pd.Period],
        oos_start: pd.Period,
        horizon: int,
        refit_freq: int,
        embargo: int,
) -> list[Fold]:
    """
    Generates a list of fold for the given months, out-of-sample start, horizon, refit frequency, and embargo.

    Private function: callers use `generate_folds()`. The `embargo=1` path exists only so the test
    suite can build a splitter that leaks and triggers the contamination guard.
    The `embargo=horizon` path is the only one used in production.

    Parameters
    ----------
    months : Sequence[pd.Period]
        Sequence of months to generate folds from.
    oos_start : pd.Period
        Month to start the out-of-sample period.
    horizon : int
        Number of months ahead that the signal is predicting.
    refit_freq : int
        Number of months between refits.
    embargo : int
        Months of separation enforced between the last training signal and the first test
        signal of each block. `horizon` is the correct value; 1 is the naive control.

    Returns
    -------
    list[Fold]
        List of generated folds. The final block may be shorter than `refit_freq`; it is
        kept, so the test blocks cover the whole usable OOS range.

    Raises
    ------
    ValueError
        If the input months sequence is invalid or if the parameters are inconsistent.
    """

    folds = []

    months = _as_month_list(months)
    if horizon < 1 or refit_freq < 1:
        raise ValueError("Horizon and refit frequency must be positive integers.")

    oos_start = pd.Period(oos_start, freq='M')
    if oos_start not in set(months):
        raise ValueError(f"oos_start {oos_start} is not in the provided months.")

    last_test = months[-1] - horizon
    if oos_start > last_test:
        raise ValueError(f"oos_start {oos_start} is after the last test month {last_test}.")

    oos = [m for m in months if oos_start <= m <= last_test]

    # anchor on first test month since it carries the earliest test label
    for i in range(0, len(oos), refit_freq):
        block = oos[i:i + refit_freq]
        first_test = block[0]
        train = [m for m in months if m <= first_test - embargo]
        if not train:
            raise ValueError(f"No training months available for test month {first_test} with embargo {embargo}.\n {oos_start} is too early for the given horizon and embargo.")

        folds.append(Fold(fit_month=train[-1], train_months=train, test_months=block))

    return folds


def generate_folds(
        months: Sequence[pd.Period],
        oos_start: pd.Period,
        horizon: int,
        refit_freq: int,
) -> list[Fold]:
    """
    Wrapper that generates folds with an embargo equal to the horizon.

    Invariants, asserted in `test/test_splits.py`:
      - every fold: max(train_months) + horizon == min(test_months)
      - every fold: max(train label month) < min(test label month)
      - every fold: leaked_months(fold, horizon) is empty
      - train windows expand: folds[i].train_months superset of folds[i-1].train_months
      - test blocks partition the usable OOS range: no gaps, no overlaps
      - every test signal T satisfies T + horizon <= months[-1]

    Parameters
    ----------
    months : Sequence[pd.Period]
        Sequence of months to generate folds from.
    oos_start : pd.Period
        Month to start the out-of-sample period. Required, with no default: the group has
        not settled a value, and nothing should inherit an undecided split by accident.
    horizon : int
        Number of months ahead that the signal is predicting.
    refit_freq : int
        Number of months between refits.

    Returns
    -------
    list[Fold]
        List of generated folds.

    Raises
    ------
    ValueError
        If the input months sequence is invalid or if the parameters are inconsistent.
    """

    return _generate(months, oos_start, horizon, refit_freq, embargo=horizon)


def rows_for_months(
        df: pd.DataFrame,
        months: Sequence[pd.Period],
        date_col: str = "date",
) -> np.ndarray:
    """
    Boolean row mask selecting the rows of `df` whose signal month is in `months`.
    The only function in this module that touches the panel.

    Parameters
    ----------
    df : pd.DataFrame
        Panel of bond-months, carrying a signal-month column.
    months : Sequence[pd.Period]
        Months to select, typically a fold's `train_months` or `test_months`.
    date_col : str, optional
        Name of the signal-month column, by default "date".

    Returns
    -------
    np.ndarray
        Boolean mask, one element per row of `df`.

    Raises
    ------
    KeyError
        If `date_col` is not a column of `df`.
    ValueError
        If no row matches. Almost always a dtype mismatch between the panel's timestamp
        `date` and the Period months, which would otherwise pass silently and fit a fold
        on zero rows.
    """

    if date_col not in df.columns:
        raise KeyError(f"{date_col!r} is not a column of df")

    wanted = {pd.Period(m, freq="M") for m in months}
    # via to_datetime so the column may be timestamps (the panel) or "YYYY-MM" strings
    mask = np.asarray(pd.PeriodIndex(pd.to_datetime(df[date_col]), freq="M").isin(wanted))

    if not mask.any():
        raise ValueError(
            f"no rows matched {len(wanted)} months ({min(wanted)} … {max(wanted)}); "
            f"check the dtype of df[{date_col!r}]"
        )

    return mask
