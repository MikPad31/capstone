"""
Tests for src/harness.py: the level transforms add up (L1 = IC + L2), the common sample is the
same mask for every level, and the three R^2 benchmarks are what they claim to be.
"""

import numpy as np
import pandas as pd
import pytest

from src.harness import IC, L0, L1, L2, LEVELS, eligible_rows, project, score


def _frame():
    # month 1: issuer 1 has 2 bonds, issuer 2 has 1 bond, one bond has no issuer key
    # month 2: issuer 1 has 3 bonds, issuer 2 has 2
    df = pd.DataFrame({
        "date": pd.to_datetime(["2015-01-31"] * 4 + ["2015-02-28"] * 5),
        "gvkey": [1.0, 1.0, 2.0, np.nan, 1.0, 1.0, 1.0, 2.0, 2.0],
        "retx_realized_return_fwd1": [0.01, 0.03, 0.02, 0.05, -0.01, 0.02, 0.05, 0.00, 0.04],
        "retxrf_realized_return_fwd1": [0.02, 0.04, 0.01, 0.06, 0.00, 0.03, 0.04, 0.01, 0.05],
    })
    return df


def test_l1_equals_ic_plus_l2():
    df = _frame().dropna(subset=["gvkey"])
    y = df["retx_realized_return_fwd1"]
    ic = project(df, y, IC)
    l2 = project(df, y, L2)
    np.testing.assert_allclose((ic + l2).to_numpy(), y.to_numpy(), atol=1e-15)
    # L2 sums to zero within issuer-month; IC is constant within it
    months = df["date"].dt.to_period("M")
    assert np.allclose(l2.groupby([df["gvkey"], months]).sum(), 0.0)
    assert (ic.groupby([df["gvkey"], months]).nunique() == 1).all()


def test_project_none_is_identity():
    df = _frame()
    y = df["retx_realized_return_fwd1"]
    np.testing.assert_array_equal(project(df, y, L0).to_numpy(), y.to_numpy())


def test_midterm_eligibility():
    df = _frame()
    assert eligible_rows(df, L0, 1, common_sample=False).sum() == 9
    # L2 drops the singleton issuer-month and the missing key
    np.testing.assert_array_equal(
        eligible_rows(df, L2, 1, common_sample=False),
        [True, True, False, False, True, True, True, True, True],
    )


def test_common_sample_is_identical_across_levels():
    df = _frame()
    masks = [eligible_rows(df, lvl, 1, common_sample=True) for lvl in LEVELS]
    for m in masks[1:]:
        np.testing.assert_array_equal(m, masks[0])
    np.testing.assert_array_equal(masks[0], eligible_rows(df, L2, 1, common_sample=False))


def test_common_sample_requires_both_labels():
    df = _frame()
    df.loc[0, "retxrf_realized_return_fwd1"] = np.nan
    mask = eligible_rows(df, L1, 1, common_sample=True)
    # row 0 loses its label, which also leaves row 1 a singleton issuer-month
    assert not mask[0] and not mask[1]


def test_score_benchmarks():
    preds = pd.DataFrame({
        "y": [1.0, -1.0, 2.0, 0.0],
        "yhat": [0.5, -0.5, 1.0, 0.0],
        "ybar": [0.25] * 4,
    })
    out = score(preds)
    sse = 0.25 + 0.25 + 1.0
    assert np.isclose(out["r2_zero"], 1 - sse / 6.0)
    assert np.isclose(out["r2_hist"], 1 - sse / sum((v - 0.25) ** 2 for v in [1.0, -1.0, 2.0, 0.0]))
    assert np.isclose(out["r2_oosmean"], 1 - sse / sum((v - 0.5) ** 2 for v in [1.0, -1.0, 2.0, 0.0]))
    assert out["n"] == 4


def test_unknown_transform_raises():
    df = _frame().dropna(subset=["gvkey"])
    bad = L2._replace(transform="median")
    with pytest.raises(ValueError):
        project(df, df["retx_realized_return_fwd1"], bad)
