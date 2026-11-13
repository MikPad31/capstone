"""
Downloads both data sources into the data directory and verifies them.

The data directory is data/raw/ by default, or $DSE4101_DATA_DIR if set (see src/config.py).
Files already present are not downloaded again, so this is safe to rerun.

    1. OSBAP ML panel: OSBAP_ML_Panel_Oct_2024.zip (886 MB) -> OSBAP_ML_Panel_Oct_2024.pkl (3.3 GB)
    2. DNR ML predictions: dnr_ml_predictions.zip (201 MB) -> dnr_ml_predictions/{predictions.parquet, README.txt}
    3. OSBAP_ML_Panel_columns.txt, the column-position sidecar src/data.py needs. It does not
       ship with the panel, so it is copied from scripts/, where it was generated from the pickle.
    4. Verification: exact file sizes, then a column-restricted load checked against the known
       shape of the panel. A truncated download fails here instead of giving wrong numbers later.

Needs ~5 GB of free disk while unpacking (the zips are deleted afterwards unless --keep-archives).

RUN: python scripts/get_data.py            (download what is missing, then verify)
     python scripts/get_data.py --verify   (verify an existing copy only)
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # run as a script from anywhere, not only as a module

from src.config import (  # noqa: E402
    DATA_DIR, DNR_PREDICTIONS_DIR, DNR_PREDICTIONS_PATH, DNR_README_PATH,
    OSBAP_COLUMNS_PATH, OSBAP_PANEL_PATH,
)
from src.runlog import check, finish, log, report  # noqa: E402

OSBAP_URL = "https://openbondassetpricing.com/wp-content/uploads/2024/10/OSBAP_ML_Panel_Oct_2024.zip"
DNR_URL = "https://openbondassetpricing.com/wp-content/uploads/2026/04/dnr_ml_predictions.zip"
COLUMNS_SOURCE = REPO_ROOT / "scripts" / "OSBAP_ML_Panel_columns.txt"

# Sizes of the files as published (checked against the archives on 2026-11-01).
EXPECTED_BYTES = {
    OSBAP_PANEL_PATH: 3_344_235_193,
    DNR_PREDICTIONS_PATH: 250_153_299,
}
EXPECTED_ROWS = 1_102_569
EXPECTED_BONDS = 19_768
EXPECTED_MONTHS = ("2002-07", "2022-11", 245)


# %% Download helpers
def download(url: str, dest: Path) -> None:
    """Streams `url` to `dest` via a .part file, so an interrupted download never looks complete."""

    import requests

    part = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done, next_pct = 0, 10
        with open(part, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total and done * 100 >= next_pct * total:
                    log(f"  {dest.name}: {next_pct}% of {total / 1e6:,.0f} MB")
                    next_pct += 10
    part.rename(dest)


def get_osbap(keep_archives: bool) -> None:
    if OSBAP_PANEL_PATH.exists():
        log(f"{OSBAP_PANEL_PATH.name} already present, skipping download")
        return
    archive = DATA_DIR / "OSBAP_ML_Panel_Oct_2024.zip"
    if not archive.exists():
        log(f"downloading {OSBAP_URL}")
        download(OSBAP_URL, archive)
    log("unpacking the panel (the zip holds a gzipped pickle)")
    tmp = OSBAP_PANEL_PATH.with_name(OSBAP_PANEL_PATH.name + ".part")
    with zipfile.ZipFile(archive) as z:
        member = next(n for n in z.namelist() if n.endswith(".pkl.gz"))
        with z.open(member) as raw, gzip.open(raw) as gz, open(tmp, "wb") as out:
            shutil.copyfileobj(gz, out, length=1 << 22)
    tmp.rename(OSBAP_PANEL_PATH)
    if not keep_archives:
        archive.unlink()


def get_dnr(keep_archives: bool) -> None:
    if DNR_PREDICTIONS_PATH.exists() and DNR_README_PATH.exists():
        log(f"{DNR_PREDICTIONS_DIR.name}/ already present, skipping download")
        return
    archive = DATA_DIR / "dnr_ml_predictions.zip"
    if not archive.exists():
        log(f"downloading {DNR_URL}")
        download(DNR_URL, archive)
    DNR_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for target in (DNR_PREDICTIONS_PATH, DNR_README_PATH):
            member = next(n for n in z.namelist() if n.rsplit("/", 1)[-1] == target.name)
            with z.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, length=1 << 22)
    if not keep_archives:
        archive.unlink()


def get_columns() -> None:
    if OSBAP_COLUMNS_PATH.exists():
        log(f"{OSBAP_COLUMNS_PATH.name} already present")
        return
    shutil.copyfile(COLUMNS_SOURCE, OSBAP_COLUMNS_PATH)
    log(f"copied {OSBAP_COLUMNS_PATH.name} from scripts/")


# %% Verification
def verify() -> None:
    sizes_ok = True
    for path, size in EXPECTED_BYTES.items():
        got = path.stat().st_size if path.exists() else None
        sizes_ok &= check(got == size, f"{path.name}: {'missing' if got is None else f'{got:,} bytes'}, expected {size:,}")
    columns_ok = check(OSBAP_COLUMNS_PATH.exists(), f"{OSBAP_COLUMNS_PATH.name} present")
    check(DNR_README_PATH.exists(), f"{DNR_README_PATH.name} present")
    if not (sizes_ok and columns_ok):
        log("skipping the load checks: delete any incomplete file, then rerun without --verify to fetch it again")
        return

    import pandas as pd
    from src.data import PREDICTOR_COLS, load_panel

    df = load_panel(cols=["date", "ID", "cusip", "gvkey"], targets=("retx", "retxrf"))
    months = pd.PeriodIndex(pd.to_datetime(df["date"]), freq="M")
    first, last, n_months = EXPECTED_MONTHS
    check(len(df) == EXPECTED_ROWS, f"panel has {len(df):,} rows, expected {EXPECTED_ROWS:,}")
    check(df["ID"].nunique() == EXPECTED_BONDS, f"{df['ID'].nunique():,} unique bonds, expected {EXPECTED_BONDS:,}")
    check((str(months.min()), str(months.max()), months.nunique()) == EXPECTED_MONTHS,
          f"months {months.min()} to {months.max()} ({months.nunique()}), expected {first} to {last} ({n_months})")
    for t in ("retx", "retxrf"):
        n_missing = int(df[f"{t}_realized_return"].isna().sum())
        check(n_missing == 0, f"{t} realized return joined onto every row ({n_missing} missing)")
    check(len(PREDICTOR_COLS) == 341, f"{len(PREDICTOR_COLS)} predictor columns, expected 341")


# %% Run
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--verify", action="store_true", help="only verify an existing copy")
    parser.add_argument("--keep-archives", action="store_true", help="keep the downloaded zips")
    args = parser.parse_args()

    start = time.time()
    report(f"data directory: {DATA_DIR}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not args.verify:
        get_osbap(args.keep_archives)
        get_dnr(args.keep_archives)
        get_columns()
    verify()
    finish(start)


if __name__ == "__main__":
    main()
