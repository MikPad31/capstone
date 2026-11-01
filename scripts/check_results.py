"""
Compares a rerun's output/ against the numbers reported in results/.

results/ holds the CSVs behind every number in the report and slides. After rerunning the
pipeline (scripts/run_phase2.sh, and the replication scripts if wanted), this checks each
regenerated file against its reported copy: same rows, and every numeric column equal to within
TOL. Files not regenerated yet are listed as skipped, not failed.

RUN: python scripts/check_results.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import OUTPUT_DIR  # noqa: E402
from src.runlog import check, finish, log, report  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
TOL = 1e-6  # numbers are reported to 2-4 decimals; multithreaded tree fits may differ in the last bits

# reported file -> regenerated file, relative to results/ and output/
FILES = {
    "phase2/scores.csv": "phase2/scores.csv",
    "phase2/dm_tests.csv": "phase2/dm_tests.csv",
    "phase2/subperiods.csv": "phase2/subperiods.csv",
    "phase2/dnr_benchmark.csv": "phase2/dnr_benchmark.csv",
    "within_issuer_predictors.csv": "within_issuer_predictors.csv",
    "data_summary/panel.csv": "data_summary/panel.csv",
    "data_summary/targets.csv": "data_summary/targets.csv",
    "data_summary/monthly_h1.csv": "data_summary/monthly_h1.csv",
    "replication/bali_final_results.csv": "replication/bali_final_results.csv",
    "replication/validation_results.csv": "replication/validation_results.csv",
}


def compare(reported: pd.DataFrame, rerun: pd.DataFrame) -> tuple[bool, str]:
    if list(reported.columns) != list(rerun.columns) or len(reported) != len(rerun):
        return False, f"shape or columns differ: {reported.shape} vs {rerun.shape}"
    num = [c for c in reported.columns if pd.api.types.is_numeric_dtype(reported[c])
           and pd.api.types.is_numeric_dtype(rerun[c])]
    other = [c for c in reported.columns if c not in num]
    keys = reported[other].astype(str).reset_index(drop=True)
    if not keys.equals(rerun[other].astype(str).reset_index(drop=True)):
        return False, "non-numeric columns differ"
    a, b = reported[num].to_numpy(dtype=float), rerun[num].to_numpy(dtype=float)
    same_nan = np.array_equal(np.isnan(a), np.isnan(b))
    diff = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
    return same_nan and diff <= TOL, f"max abs diff {diff:.1e} over {a.size:,} numbers"


def main() -> None:
    start = time.time()
    for rel_reported, rel_rerun in FILES.items():
        reported, rerun = RESULTS_DIR / rel_reported, OUTPUT_DIR / rel_rerun
        if not rerun.exists():
            report(f"skipped {rel_reported}: output/{rel_rerun} not regenerated yet")
            continue
        ok, detail = compare(pd.read_csv(reported), pd.read_csv(rerun))
        check(ok, f"{rel_reported}: {detail}")
    log(f"tolerance {TOL:g}")
    finish(start)


if __name__ == "__main__":
    main()
