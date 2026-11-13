"""
Tests for src/models.py: the Gram-based elastic net against scikit-learn, the inner split's
embargo, and that both tuned estimators fit, tune and predict on a small synthetic panel.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import ElasticNet

from src.models import (
    TunedElasticNet,
    TunedRandomForest,
    _cd_gram,
    _centred,
    _moments,
    _moments_masked,
    inner_split,
)


def _panel(n_months=80, n_per_month=60, p=8, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.period_range("2005-01", periods=n_months, freq="M")
    m = np.repeat(months, n_per_month)
    X = rng.uniform(-1, 1, size=(len(m), p))
    beta = np.zeros(p)
    beta[:3] = [0.5, -0.3, 0.2]
    y = 0.01 + X @ beta + rng.normal(scale=0.5, size=len(m))
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(p)]), pd.Series(y), pd.PeriodIndex(m)


@pytest.mark.parametrize("alpha,l1_ratio", [(0.01, 0.5), (0.05, 0.9), (0.001, 0.1)])
def test_cd_gram_matches_sklearn(alpha, l1_ratio):
    X, y, _ = _panel()
    Xv, yv = X.to_numpy(), y.to_numpy()
    gram, xy, mx, my = _centred(_moments(Xv, yv))
    coef = _cd_gram(gram, xy, float(len(yv)), alpha, l1_ratio, np.zeros(Xv.shape[1]), tol=1e-12)

    ref = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, tol=1e-12, max_iter=100_000).fit(Xv, yv)
    np.testing.assert_allclose(coef, ref.coef_, atol=1e-6)
    np.testing.assert_allclose(my - mx @ coef, ref.intercept_, atol=1e-6)


def test_moments_add_up_to_full_sample():
    X, y, _ = _panel()
    Xv, yv = X.to_numpy(), y.to_numpy()
    a, b = _moments(Xv[:1000], yv[:1000]), _moments(Xv[1000:], yv[1000:])
    full = _moments(Xv, yv)
    for k in full:
        np.testing.assert_allclose(np.asarray(a[k]) + np.asarray(b[k]), full[k])


def test_chunked_masked_moments_match_direct():
    X, y, _ = _panel()
    Xv, yv = X.to_numpy(), y.to_numpy()
    mask = np.random.default_rng(3).random(len(yv)) < 0.4
    direct = _moments(Xv[mask], yv[mask])
    chunked = _moments_masked(Xv.astype(np.float32), yv, mask, chunk=777)
    for k in direct:
        np.testing.assert_allclose(chunked[k], direct[k], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("horizon", [1, 3, 12])
def test_inner_split_embargo(horizon):
    _, _, months = _panel()
    inner, val = inner_split(months, horizon, n_val_months=24)
    val_months = months[val].unique()
    inner_months = months[inner].unique()

    assert len(val_months) == 24
    assert val_months.max() == months.max()
    # exactly h months separate the last inner-train signal from the first validation signal
    assert (val_months.min() - inner_months.max()).n == horizon
    assert not (inner & val).any()


def test_inner_split_refuses_short_window():
    months = pd.PeriodIndex(pd.period_range("2010-01", periods=30, freq="M"))
    with pytest.raises(ValueError):
        inner_split(months, horizon=12, n_val_months=24)


def test_tuned_elastic_net_recovers_signal():
    X, y, months = _panel(n_months=120, n_per_month=200)
    model = TunedElasticNet(n_alphas=10, n_val_months=24)
    model.fit(X, y, months=months, horizon=1)
    assert {"alpha", "l1_ratio", "val_mse", "n_nonzero"} <= set(model.best_params_)
    # the three true signals are kept, with the right signs
    assert np.sign(model.coef_[:3]).tolist() == [1.0, -1.0, 1.0]
    assert model.predict(X).shape == (len(X),)


def test_tuned_random_forest_fits_and_predicts():
    X, y, months = _panel()
    model = TunedRandomForest(
        grid={"max_depth": (2, 3), "min_data_in_leaf": (20,)},
        n_trees_tune=10, n_trees_final=20, n_threads=1, n_val_months=24,
    )
    model.fit(X, y, months=months, horizon=3)
    assert model.best_params_["max_depth"] in (2, 3)
    pred = model.predict(X)
    assert pred.shape == (len(X),) and np.isfinite(pred).all()
    imp = model.feature_importance(X.columns)
    assert np.isclose(imp.sum(), 1.0)
    # the strongest true signal ranks among the top predictors
    assert "x0" in imp.index[:3]
