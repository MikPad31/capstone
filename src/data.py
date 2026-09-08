"""
Data-loading entry point for the OSBAP panel and DNR predictions.

- OSBAP_ML_Panel_Oct_2024.pkl is 1,102,569 rows x 381 columns (~5.1GB)
- this module loads a column-restricted slice instead of the full file, by hooking
  pandas' pickle format to harvest only the requested columns out of each block
"""

from __future__ import annotations

import gc
import pathlib
import pickle

import numpy as np
import pandas as pd

from src.config import (
    DNR_PREDICTIONS_PATH as _PARQUET_PATH,
    OSBAP_COLUMNS_PATH as _COLUMNS_TXT,
    OSBAP_PANEL_PATH as _PANEL_PATH,
)

_CATEGORICAL_COLS = {"cusip", "issuer_cusip"}
_VALID_TARGETS = {"retd", "retx", "retxrf"}


def _require(path: pathlib.Path, what: str) -> pathlib.Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found at {path}. Set DSE4101_DATA_DIR to the directory "
            f"holding the data files (see src/config.py)."
        )
    return path


def _load_column_order() -> list[str]:
    """
    Parses the column order from the OSBAP_ML_Panel_columns.txt file into a position-ordered name list.
    Allows for picking of which block of columns to load.
    """

    txt = _require(_COLUMNS_TXT, "OSBAP_ML_Panel_columns.txt")
    lines = txt.read_text().splitlines()[2:]  # ignore 2 header lines
    names: list[str] = [None] * len(lines)  # type: ignore[list-item]
    for line in lines:
        if not line:
            continue
        pos_str, name = line.split("\t", 1)
        names[int(pos_str)] = name
    assert all(n is not None for n in names), "columns.txt did not cover every position"
    return names


_COLUMN_CACHE: list[str] | None = None


def all_columns() -> list[str]:
    """
    381 names, index = column position. 
    Read once, on first use.
    """
    
    global _COLUMN_CACHE
    if _COLUMN_CACHE is None:
        _COLUMN_CACHE = _load_column_order()
    return _COLUMN_CACHE


# Resolved on attribute access rather than at import, so importing this module
# does not require the data directory to be present.
_LAZY = {
    "ALL_COLUMNS": lambda cols: cols,
    "PREDICTOR_COLS": lambda cols: cols[13:354],  # 341 rank-rescaled-to-[-1,1] predictors
    "FORECAST_COLS": lambda cols: cols[354:381],  # 27 DNR ML forecasts, out of scope here
}


def __getattr__(name: str):
    if name in _LAZY:
        return _LAZY[name](all_columns())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _DoneEarly(Exception):
    """
    Raised once every requested column is harvested, to stop the unpickler
    from also reading blocks (e.g. the 27-column forecast block) nobody asked for.
    """


class _ColumnSelectiveUnpickler(pickle.Unpickler):
    """
    Hooks pandas' block-reconstruction step to copy out only the wanted
    columns from each block as it streams by, instead of materializing the
    whole DataFrame.
    """

    def __init__(self, file, wanted_positions: set[int]):
        super().__init__(file)
        self._wanted_positions = wanted_positions
        self.harvested: dict[str, np.ndarray] = {}

    def find_class(self, module, name):
        real = super().find_class(module, name)
        if module == "pandas._libs.internals" and name == "_unpickle_block":
            return self._hook(real)
        return real

    def _hook(self, real_unpickle_block):
        def hook(values, placement, ndim): # handle both BlockPlacement and slice placements in pandas' builds
            if isinstance(placement, slice):
                positions = np.arange(placement.start, placement.stop, placement.step or 1)
            elif hasattr(placement, "as_array"):
                positions = np.asarray(placement.as_array).ravel()
            else:
                positions = np.asarray(placement).ravel()

            hit = [p for p in positions.tolist() if p in self._wanted_positions]
            if hit:
                arr2d = values if values.ndim == 2 else values[np.newaxis, :]
                for pos in hit:
                    local_row = int(np.where(positions == pos)[0][0])
                    self.harvested[all_columns()[pos]] = arr2d[local_row].copy()
            del values  # drop the only strong ref so the block is collectible

            if len(self.harvested) == len(self._wanted_positions):
                raise _DoneEarly()
            return None  # never need the reconstructed Block itself, only the raw arrays above

        return hook


def _read_columns(names: list[str]) -> dict[str, np.ndarray]:
    """
    Streams the panel pickle and harvests just `names`, stopping early once all are found.
    """

    name_to_pos = {name: pos for pos, name in enumerate(all_columns())}
    positions = {name_to_pos[n] for n in names}
    with open(_require(_PANEL_PATH, "OSBAP panel pickle"), "rb") as fh:
        unpickler = _ColumnSelectiveUnpickler(fh, positions)
        try:
            unpickler.load()
        except _DoneEarly:
            pass
    missing = set(names) - unpickler.harvested.keys()
    if missing:
        raise RuntimeError(
            f"load_panel: failed to harvest columns {sorted(missing)} -- "
            "pandas' internal pickle block format may have changed."
        )
    return unpickler.harvested


def _join_realized_returns(df: pd.DataFrame, targets: tuple[str, ...]) -> pd.DataFrame:
    """
    Left-joins `<target>_realized_return` columns from predictions.parquet onto `df`.
    """

    bad = set(targets) - _VALID_TARGETS
    if bad:
        raise ValueError(f"unknown target(s) {sorted(bad)}, expected subset of {_VALID_TARGETS}")

    # realized_return doesn't depend on model_key, so read only the 4 needed
    # columns and dedup instead of also reading model_key to filter by it
    preds = pd.read_parquet(
        _require(_PARQUET_PATH, "predictions.parquet"),
        columns=["signal_date", "cusip", "target", "realized_return"],
        filters=[("target", "in", list(targets))],
    )
    preds = preds.drop_duplicates(subset=["signal_date", "cusip", "target"])
    preds["signal_date"] = preds["signal_date"].astype(df["date"].dtype)

    wide = preds.pivot(index=["signal_date", "cusip"], columns="target", values="realized_return")
    wide.columns = [f"{t}_realized_return" for t in wide.columns]
    wide = wide.reset_index().rename(columns={"signal_date": "date"})
    del preds
    gc.collect()

    merged = df.merge(wide, on=["date", "cusip"], how="left")
    del wide
    gc.collect()
    return merged


def load_panel(
    cols: list[str] | None = None,
    targets: tuple[str, ...] = ("retx", "retxrf"),
) -> pd.DataFrame:
    """
    Loads a column-restricted slice of the OSBAP panel, optionally joined to realized returns.

    cols : list[str] 
        column names to pull (see ALL_COLUMNS / PREDICTOR_COLS). `date` is always included. 
        Defaults to a small identifier set; predictors are never loaded unless explicitly requested.
    targets : tuple[str, ...] 
        realized-return targets (subset of retd/retx/retxrf) to join from predictions.parquet,
        matched on date == signal_date, cusip == cusip with no extra lag.
        Pass () to skip the join.
    """
    if cols is None:
        cols = ["date", "ID", "cusip", "gvkey", "issuer_cusip", "RATING_NUM"]

    wanted = list(dict.fromkeys(["date", *cols]))
    if targets and "cusip" not in wanted:
        wanted.append("cusip")  # required as a join key even if not requested

    raw = _read_columns(wanted)
    df = pd.DataFrame({name: raw[name] for name in wanted})
    del raw
    gc.collect()

    for name in wanted:
        if name in _CATEGORICAL_COLS:
            df[name] = df[name].astype("category")

    if targets:
        df = _join_realized_returns(df, targets) # recast the joined realized-return columns to float32 to save memory
        for name in wanted:
            if name in _CATEGORICAL_COLS:
                df[name] = df[name].astype("category")

    return df
