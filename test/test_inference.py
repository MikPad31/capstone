"""
Tests for src/inference.py: Newey-West against statsmodels, the monthly collapse, and the sign
convention of the modified Diebold-Mariano statistic.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.inference import dm_lags, dm_test, monthly_loss_diff, newey_west_se


@pytest.mark.parametrize("lags", [0, 2, 11])
def test_newey_west_matches_statsmodels(lags):
    rng = np.random.default_rng(1)
    d = np.convolve(rng.normal(size=200), np.ones(4) / 4, mode="valid")  # serially correlated
    fit = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False})
    assert np.isclose(newey_west_se(d, lags), float(fit.bse[0]), rtol=1e-10)


def test_dm_lags_floor_and_overlap():
    assert dm_lags(12, 95) == 11       # overlap dominates at h=12
    assert dm_lags(1, 95) == 3         # rule-of-thumb floor at h=1
    assert dm_lags(3, 95) == 3


def test_monthly_loss_diff_is_a_per_month_mean():
    months = pd.PeriodIndex(["2015-01", "2015-01", "2015-02"], freq="M")
    y = np.array([1.0, 0.0, 2.0])
    d = monthly_loss_diff(y, yhat1=np.zeros(3), yhat2=y, months=months)
    # model 2 is perfect: d_t = mean of squared model-1 errors in each month
    assert d.tolist() == [0.5, 4.0]


def test_dm_sign_positive_when_model_two_better():
    rng = np.random.default_rng(2)
    months = np.repeat(pd.period_range("2015-01", periods=96, freq="M"), 50)
    signal = rng.normal(size=len(months))
    y = signal + rng.normal(size=len(months))
    out = dm_test(y, yhat1=np.zeros(len(y)), yhat2=signal, months=months, horizon=1)
    assert out["stat"] > 5 and out["p_value"] < 1e-6
    flipped = dm_test(y, yhat1=signal, yhat2=np.zeros(len(y)), months=months, horizon=1)
    assert np.isclose(flipped["stat"], -out["stat"])


def test_dm_refuses_nan():
    months = pd.PeriodIndex(["2015-01", "2015-02"], freq="M")
    with pytest.raises(ValueError):
        dm_test(np.array([1.0, np.nan]), np.zeros(2), np.zeros(2), months, horizon=1)
