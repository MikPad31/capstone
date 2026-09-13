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

# Declared provisionally, may change after exploratory analysis.
OOS_START = None

SEED = 0
