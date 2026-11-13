"""
Which predictors vary across the bonds of one issuer in the same month?

For every predictor, the share of multi-bond (gvkey, month) groups in which it is not constant.
An issuer-level variable (equity, fundamentals) should sit at ~0%; a bond-level one (coupon,
age, duration, ...) well above. Writes output/within_issuer_predictors.csv, which
`exploratory/phase2_grid.py` reads to build the within-issuer-varying predictor subset.

RUN: DSE4101_DATA_DIR=... python -m exploratory.within_issuer_predictors
"""

# %% Setup
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR
from src.data import PREDICTOR_COLS, load_panel
from src.runlog import finish, log, report

# A predictor counts as within-issuer varying if it differs across an issuer's bonds in at
# least this share of multi-bond issuer-months. Chosen from the gap in the printed distribution.
VARYING_THRESHOLD = 0.05
TOL = 1e-9

# %% Load
_start = time.time()
df = load_panel(cols=["date", "gvkey", *PREDICTOR_COLS], targets=())
df[PREDICTOR_COLS] = df[PREDICTOR_COLS].astype("float32")
df = df[df["gvkey"].notna()]
df["gvkey"] = df["gvkey"].astype("int64")
df["_month"] = df["date"].dt.to_period("M")

size = df.groupby(["gvkey", "_month"])["date"].transform("size")
multi = df[size >= 2]
report(f"{len(multi):,} bond-months in {multi.groupby(['gvkey', '_month']).ngroups:,} multi-bond issuer-months")

# %% Within-group range per predictor
g = multi.groupby(["gvkey", "_month"])[PREDICTOR_COLS]
spread = (g.max() - g.min()) > TOL
share = spread.mean().rename("share_groups_varying")

out = share.to_frame()
out["varying"] = out["share_groups_varying"] >= VARYING_THRESHOLD
out = out.sort_values("share_groups_varying", ascending=False)
out.index.name = "predictor"
out.to_csv(OUTPUT_DIR / "within_issuer_predictors.csv")

# %% Report
bins = [0, 1e-12, 0.001, 0.01, 0.05, 0.2, 0.5, 1.0001]
report("distribution of share-of-groups-varying across predictors:")
print(pd.cut(out["share_groups_varying"], bins, right=False).value_counts().sort_index().to_string(), flush=True)
report(f"within-issuer varying (>= {VARYING_THRESHOLD:.0%} of groups): {int(out['varying'].sum())} of {len(out)}")
print(out[out["varying"]].to_string(float_format=lambda v: f"{v:.3f}"), flush=True)
report("largest shares among those classed issuer-constant:")
print(out[~out["varying"]].head(10).to_string(float_format=lambda v: f"{v:.4f}"), flush=True)

finish(_start)
