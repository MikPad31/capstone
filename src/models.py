"""
Tuned estimators for the Phase 2 grid: Elastic Net and Random Forest (LightGBM in RF mode).

Both tune on an embargoed inner split of their own training window and then refit on the
whole window with the chosen hyperparameters:

    inner validation = the last `n_val_months` training signal months
    inner train      = signal months s with s + horizon <= first validation month

which is `src/splits.py`'s embargo applied once more inside the fold, so no inner-train label
overlaps a validation label. Nothing here sees the outer fold's test months.

Both expose `needs_months = True` so `src/harness.py` passes the training signal months and the
horizon to `fit`, and `best_params_` so the chosen hyperparameters land in the fold diagnostics.
"""

from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np
import pandas as pd

from src.config import SEED

N_VAL_MONTHS = 36


def inner_split(
        months: pd.PeriodIndex,
        horizon: int,
        n_val_months: int = N_VAL_MONTHS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Boolean masks (inner_train, validation) over the rows of a training window.

    Parameters
    ----------
    months : pd.PeriodIndex
        Signal month of every training row.
    horizon : int
        Forecast horizon; the gap between the last inner-train and the first validation signal.
    n_val_months : int
        Number of trailing signal months held out for validation.

    Raises
    ------
    ValueError
        If the window is too short to leave both an inner-train and a validation block.
    """

    unique = months.unique().sort_values()
    if len(unique) < n_val_months + horizon + 12:
        raise ValueError(
            f"inner_split: {len(unique)} training months cannot hold {n_val_months} validation "
            f"months, a {horizon}-month embargo and a year of inner training"
        )
    first_val = unique[-n_val_months]
    val = np.asarray(months >= first_val)
    inner = np.asarray(months <= first_val - horizon)
    return inner, val


# %% Elastic Net

def _moments(X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray | float]:
    """Raw cross-products, so centred Grams of a union of row blocks can be assembled without refitting."""

    return {
        "n": float(X.shape[0]),
        "sx": X.sum(axis=0),
        "sy": float(y.sum()),
        "xtx": X.T @ X,
        "xty": X.T @ y,
        "yty": float(y @ y),
    }


def _moments_masked(X: np.ndarray, y: np.ndarray, mask: np.ndarray, chunk: int = 200_000) -> dict:
    """`_moments` of the `mask` rows, accumulated in float64 over row chunks of a float32 `X`."""

    p = X.shape[1]
    out = {"n": 0.0, "sx": np.zeros(p), "sy": 0.0, "xtx": np.zeros((p, p)), "xty": np.zeros(p), "yty": 0.0}
    for start in range(0, len(y), chunk):
        m = mask[start:start + chunk]
        if m.any():
            part = _moments(X[start:start + chunk][m].astype(np.float64), y[start:start + chunk][m])
            out = _add(out, part)
    return out


def _centred(m: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Centred Gram, centred X'y, and the means of X and y, from raw moments."""

    mx = m["sx"] / m["n"]
    my = m["sy"] / m["n"]
    gram = m["xtx"] - m["n"] * np.outer(mx, mx)
    xy = m["xty"] - m["n"] * mx * my
    return gram, xy, mx, my


def _add(a: dict, b: dict) -> dict:
    return {k: a[k] + b[k] for k in a}


def _cd_gram(
        gram: np.ndarray,
        xy: np.ndarray,
        n: float,
        alpha: float,
        l1_ratio: float,
        coef: np.ndarray,
        tol: float = 1e-7,
        max_iter: int = 5000,
) -> np.ndarray:
    """
    Coordinate descent for the elastic net, on sufficient statistics only.

    Minimises (1 / 2n) ||y - Xb||^2 + alpha * l1_ratio * ||b||_1 + 0.5 * alpha * (1 - l1_ratio) * ||b||^2
    (scikit-learn's ElasticNet objective) over centred data, using the Gram matrix and X'y only:
    the 1M-row design is never touched, so each fit costs O(p^2) per sweep regardless of n.
    `coef` is the warm start and is updated in place.

    Agreement with sklearn.linear_model.ElasticNet is asserted in test/test_models.py.
    """

    l1 = alpha * l1_ratio * n
    l2 = alpha * (1.0 - l1_ratio) * n
    diag = np.diag(gram)
    # grad[j] = xy[j] - (gram @ coef)[j], maintained incrementally
    resid_corr = xy - gram @ coef
    scale = max(float(np.abs(xy).max()), 1e-12)

    for _ in range(max_iter):
        max_step = 0.0
        for j in range(len(coef)):
            if diag[j] == 0.0:
                continue
            old = coef[j]
            rho = resid_corr[j] + diag[j] * old
            new = np.sign(rho) * max(abs(rho) - l1, 0.0) / (diag[j] + l2)
            if new != old:
                resid_corr -= gram[j] * (new - old)  # gram is symmetric; the row is contiguous
                coef[j] = new
                max_step = max(max_step, abs(new - old) * diag[j])
        if max_step <= tol * scale:
            break
    return coef


class TunedElasticNet:
    """
    Elastic Net with (alpha, l1_ratio) chosen on the embargoed inner validation split.

    For each l1_ratio, walks a descending alpha path with warm starts, scores every point on the
    validation block by MSE, and refits the winner on the full training window. Predictors are
    centred, not rescaled: the panel's predictors are already rank-rescaled to [-1, 1], so they
    share a scale and the penalty treats them alike.
    """

    needs_months = True

    def __init__(
            self,
            l1_ratios: Sequence[float] = (0.1, 0.5, 0.9),
            n_alphas: int = 25,
            alpha_min_ratio: float = 1e-4,
            n_val_months: int = N_VAL_MONTHS,
    ):
        self.l1_ratios = tuple(l1_ratios)
        self.n_alphas = n_alphas
        self.alpha_min_ratio = alpha_min_ratio
        self.n_val_months = n_val_months
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.best_params_: dict = {}
        self.param_grid = {"l1_ratio": self.l1_ratios, "n_alphas": n_alphas}

    def fit(self, X: pd.DataFrame, y: pd.Series, months: pd.PeriodIndex, horizon: int) -> None:
        Xv = X.to_numpy(dtype=np.float32)
        yv = y.to_numpy(dtype=np.float64)
        inner, val = inner_split(months, horizon, self.n_val_months)
        gap = ~(inner | val)

        m_inner = _moments_masked(Xv, yv, inner)
        gram, xy, mx, my = _centred(m_inner)
        X_val, y_val = Xv[val].astype(np.float64), yv[val]

        best = (np.inf, None, None)
        for l1_ratio in self.l1_ratios:
            alpha_max = np.abs(xy).max() / (m_inner["n"] * l1_ratio)
            alphas = np.geomspace(alpha_max, alpha_max * self.alpha_min_ratio, self.n_alphas)
            coef = np.zeros(Xv.shape[1])
            for alpha in alphas:
                coef = _cd_gram(gram, xy, m_inner["n"], alpha, l1_ratio, coef)
                pred = my + (X_val - mx) @ coef
                mse = float(np.mean((y_val - pred) ** 2))
                if mse < best[0]:
                    best = (mse, alpha, l1_ratio)

        _, alpha, l1_ratio = best
        # Refit on inner + validation + the embargo gap: every training row.
        m_full = _add(_add(m_inner, _moments_masked(Xv, yv, val)), _moments_masked(Xv, yv, gap))
        gram, xy, mx, my = _centred(m_full)
        coef = _cd_gram(gram, xy, m_full["n"], alpha, l1_ratio, np.zeros(Xv.shape[1]))

        self.coef_ = coef
        self.intercept_ = float(my - mx @ coef)
        self.best_params_ = {
            "alpha": float(alpha),
            "l1_ratio": float(l1_ratio),
            "val_mse": float(best[0]),
            "n_nonzero": int((coef != 0).sum()),
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("TunedElasticNet.predict called before fit")
        return self.intercept_ + X.to_numpy(dtype=np.float64) @ self.coef_


# %% Random Forest

# Depth 1 is included because tuning on the largest fold, and the midterm replication, both
# chose the shallowest depth offered: an optimum on the edge of the grid is not an optimum.
# Depth 6 is in turn the grid's upper edge, and L2 at h=1 picks it in 6 of 8 folds; see
# exploratory/README.md before reading L2 results as evidence about nonlinearity.
RF_GRID = {
    "max_depth": (1, 2, 4, 6),
    "min_data_in_leaf": (200, 2000),
}


class TunedRandomForest:
    """
    Random forest (LightGBM, boosting="rf") with depth and leaf size chosen on the embargoed
    inner validation split, then refit on the full training window with more trees.

    boosting="rf" averages independently grown trees on histogram-binned features. Two
    differences from Breiman's random forest: each tree is fitted on a `bagging_fraction`
    subsample of rows drawn without replacement (not a bootstrap sample), and `feature_fraction`
    samples the predictors once per tree, not at every split (LightGBM's per-split option is
    `feature_fraction_bynode`, not used here).
    """

    needs_months = True

    def __init__(
            self,
            grid: dict | None = None,
            n_trees_tune: int = 100,
            n_trees_final: int = 300,
            bagging_fraction: float = 0.5,
            feature_fraction: float = 0.33,
            max_bin: int = 63,
            n_threads: int = 6,
            n_val_months: int = N_VAL_MONTHS,
            tune: bool = True,
    ):
        self.grid = grid if grid is not None else RF_GRID
        self.n_trees_tune = n_trees_tune
        self.n_trees_final = n_trees_final
        self.base = {
            "objective": "regression",
            "boosting": "rf",
            "bagging_fraction": bagging_fraction,
            "bagging_freq": 1,
            "feature_fraction": feature_fraction,
            "max_bin": max_bin,
            "num_threads": n_threads,
            "seed": SEED,
            "deterministic": True,
            "force_row_wise": True,
            "verbose": -1,
        }
        self.n_val_months = n_val_months
        self.tune = tune
        self.model_ = None
        self.best_params_: dict = {}
        self.param_grid = self.grid

    def _params(self, max_depth: int, min_data_in_leaf: int) -> dict:
        return {
            **self.base,
            "max_depth": max_depth,
            "num_leaves": 2 ** max_depth,
            "min_data_in_leaf": min_data_in_leaf,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series, months: pd.PeriodIndex, horizon: int) -> None:
        import lightgbm as lgb

        Xv = X.to_numpy(dtype=np.float32)
        yv = y.to_numpy(dtype=np.float64)
        configs = list(product(self.grid["max_depth"], self.grid["min_data_in_leaf"]))

        if self.tune and len(configs) > 1:
            inner, val = inner_split(months, horizon, self.n_val_months)
            ds = lgb.Dataset(Xv[inner], yv[inner], free_raw_data=True)
            X_val, y_val = Xv[val], yv[val]
            scores = {}
            for depth, leaf in configs:
                booster = lgb.train(self._params(depth, leaf), ds, num_boost_round=self.n_trees_tune)
                scores[(depth, leaf)] = float(np.mean((y_val - booster.predict(X_val)) ** 2))
            depth, leaf = min(scores, key=scores.get)
            val_mse = scores[(depth, leaf)]
        else:
            depth, leaf = configs[0]
            val_mse = float("nan")

        ds_full = lgb.Dataset(Xv, yv, free_raw_data=True)
        self.model_ = lgb.train(self._params(depth, leaf), ds_full, num_boost_round=self.n_trees_final)
        self.best_params_ = {"max_depth": depth, "min_data_in_leaf": leaf, "val_mse": val_mse}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("TunedRandomForest.predict called before fit")
        return self.model_.predict(X.to_numpy(dtype=np.float32))

    def feature_importance(self, names: Sequence[str]) -> pd.Series:
        """Total split gain per predictor, normalised to sum to one."""

        gain = self.model_.feature_importance(importance_type="gain")
        return pd.Series(gain / max(gain.sum(), 1e-12), index=list(names)).sort_values(ascending=False)
