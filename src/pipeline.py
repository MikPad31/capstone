"""
OLS estimator for the target pipeline harness.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import pandas as pd


class Estimator(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class OLS:
    """
    Ordinary least squares with an intercept appended as a ones column.
    Zero-variance predictors are assumed already dropped upstream.


    `solver` is deliberately explicit, not auto-selected: the choice is made once, from a
    coefficient comparison on the first real fold (see `compare_solvers`), not guessed at
    write time.
    """

    def __init__(self, solver: str = "cholesky"):
        if solver not in ("cholesky", "lstsq"):
            raise ValueError(f"OLS: unknown solver {solver!r}")
        self.solver = solver
        self.coef_: np.ndarray | None = None
        self.cond_: float | None = None  # condition number of X'X, cholesky path only

    @staticmethod
    def _design(X: pd.DataFrame) -> np.ndarray:
        # Cross-products accumulate in float64 regardless of X's dtype (float32 upstream).
        Xv = X.to_numpy(dtype=np.float64)
        ones = np.ones((Xv.shape[0], 1), dtype=np.float64)
        return np.hstack([ones, Xv])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        Xd = self._design(X)
        yv = y.to_numpy(dtype=np.float64)

        if self.solver == "lstsq":
            coef, *_ = np.linalg.lstsq(Xd, yv, rcond=None)
        else:
            xtx = Xd.T @ Xd
            xty = Xd.T @ yv
            self.cond_ = float(np.linalg.cond(xtx))
            L = np.linalg.cholesky(xtx) # xtx = L L'
            z = np.linalg.solve(L, xty) # forward solve
            coef = np.linalg.solve(L.T, z) # back solve

        self.coef_ = coef

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("OLS.predict called before fit")
        return self._design(X) @ self.coef_


def ols_builder(solver: str = "cholesky") -> Callable[[], OLS]:
    """
    Zero-argument builder: `ols_builder()` returns the builder
    Each call to the returned function returns a fresh OLS.
    """
    def _build() -> OLS:
        return OLS(solver=solver)
    return _build


def compare_solvers(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Fits both solvers on the same (X, y) and reports their relative coefficient
    difference and the condition number of X'X. 
    Run once, on the first real fold to decide `solver` for the grid and not a per-fold check.

    Returns
    -------
    dict with `max_rel_diff`, `cond`, `coef_cholesky`, `coef_lstsq`.
    """
    chol = OLS(solver="cholesky")
    chol.fit(X, y)
    ls = OLS(solver="lstsq")
    ls.fit(X, y)

    denom = np.maximum(np.abs(ls.coef_), 1e-12)
    max_rel_diff = float(np.max(np.abs(chol.coef_ - ls.coef_) / denom))

    return {
        "max_rel_diff": max_rel_diff,
        "cond": chol.cond_,
        "coef_cholesky": chol.coef_,
        "coef_lstsq": ls.coef_,
    }