"""
Set file paths, keys, and seeds.

- Do not hardcode paths outside this module.
- Set DSE4101_DATA_DIR to point at an existing copy of the data instead of downloading a second one.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("DSE4101_DATA_DIR", REPO_ROOT / "data" / "raw"))
OUTPUT_DIR = REPO_ROOT / "output"

OSBAP_PANEL_PATH = DATA_DIR / "OSBAP_ML_Panel_Oct_2024.pkl"
OSBAP_COLUMNS_PATH = DATA_DIR / "OSBAP_ML_Panel_columns.txt"

DNR_PREDICTIONS_DIR = DATA_DIR / "dnr_ml_predictions"
DNR_PREDICTIONS_PATH = DNR_PREDICTIONS_DIR / "predictions.parquet"
DNR_README_PATH = DNR_PREDICTIONS_DIR / "README.txt"

PANEL_START = "2002-07"
PANEL_END = "2022-11"

HORIZONS = (1, 3, 12)

# First out-of-sample signal month. Every saved prediction in output/phase2/ was fitted with it,
# so changing it means refitting the whole grid (~3 hours of compute on a 16 GB machine);
# phase2_grid.py refuses to mix starts.
OOS_START = "2015-01"

# Robustness sub-periods over forecast-origin (signal) months: (name, first, last), inclusive.
# Must tile OOS_START..PANEL_END with no gap or overlap (asserted in test/test_config.py).
SUBPERIODS = (
    ("pre_covid", "2015-01", "2019-12"),
    ("covid", "2020-01", "2020-12"),
    ("post_covid", "2021-01", "2022-11"),
)

SEED = 0
