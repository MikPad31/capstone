"""
Modified Diebold-Mariano test for panel forecasts (Gu, Kelly & Xiu, 2020).

Comparing two forecasts of the same target on the same rows, GKX collapse the panel to one
loss differential per month,

    d_t = mean over bonds i in month t of (e1_it^2 - e2_it^2),

and test mean(d_t) = 0 with a Newey-West standard error. Averaging within the month absorbs
the strong cross-sectional correlation of errors; the HAC correction absorbs the serial
correlation that overlapping 3- and 12-month labels induce across months.

Sign convention: a positive statistic means model 2 has the lower loss (model 2 is better).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def dm_lags(horizon: int, n_months: int) -> int:
    """
    Newey-West bandwidth, fixed in advance: max(h - 1, floor(4 (T/100)^(2/9))).

    h - 1 is the MA order that overlapping h-month labels induce; the Newey-West (1994) rule of
    thumb is the floor for h = 1 and h = 3, where residual serial correlation is still possible.
    """

    return max(horizon - 1, int(math.floor(4 * (n_months / 100) ** (2 / 9))))


def newey_west_se(d: np.ndarray, lags: int) -> float:
    """Newey-West (Bartlett kernel) standard error of the mean of `d`."""

    d = np.asarray(d, dtype="float64")
    T = len(d)
    u = d - d.mean()
    lrv = float(u @ u) / T
    for k in range(1, min(lags, T - 1) + 1):
        w = 1.0 - k / (lags + 1)
        lrv += 2.0 * w * float(u[k:] @ u[:-k]) / T
    return math.sqrt(max(lrv, 0.0) / T)


def monthly_loss_diff(
        y: np.ndarray,
        yhat1: np.ndarray,
        yhat2: np.ndarray,
        months: pd.PeriodIndex | np.ndarray,
) -> pd.Series:
    """d_t: per-month cross-sectional mean of squared-error differences, model 1 minus model 2."""

    y = np.asarray(y, dtype="float64")
    diff = (y - np.asarray(yhat1, dtype="float64")) ** 2 - (y - np.asarray(yhat2, dtype="float64")) ** 2
    if np.isnan(diff).any():
        raise ValueError("monthly_loss_diff: NaN in errors; compare on a common sample")
    return pd.Series(diff).groupby(np.asarray(months)).mean().sort_index()


def dm_test(
        y: np.ndarray,
        yhat1: np.ndarray,
        yhat2: np.ndarray,
        months: pd.PeriodIndex | np.ndarray,
        horizon: int,
        lags: int | None = None,
) -> dict[str, float]:
    """
    Modified DM test of equal predictive accuracy.

    Returns
    -------
    dict with
        stat     : mean(d_t) / NW se; positive means model 2 is more accurate
        p_value  : two-sided, standard normal
        mean_d   : mean monthly loss differential
        n_months : T
        lags     : the bandwidth used
    """

    d = monthly_loss_diff(y, yhat1, yhat2, months).to_numpy()
    T = len(d)
    if lags is None:
        lags = dm_lags(horizon, T)
    se = newey_west_se(d, lags)
    stat = float(d.mean() / se) if se > 0 else float("nan")
    p = float(math.erfc(abs(stat) / math.sqrt(2))) if se > 0 else float("nan")
    return {"stat": stat, "p_value": p, "mean_d": float(d.mean()), "n_months": T, "lags": lags}
