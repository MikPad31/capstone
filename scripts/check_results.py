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
IGNORE = {"seconds", "rss_gb"}  # run-time bookkeeping, not results
# The separate Bali replication fits scikit-learn random forests, which are not bit-identical across
# library versions (exploratory/README.md: OOS RF 0.5503% vs 0.5519% on a full rerun, same chosen
# parameters). A difference here is reported, not failed.
KNOWN_DIFF = {"replication/bali_final_results.csv", "replication/validation_results.csv"}

# reported file -> regenerated file, relative to results/ and output/
FILES = {
    "phase2/primary_scores.csv": "phase2/primary_scores.csv",
    "phase2/primary_dm_tests.csv": "phase2/primary_dm_tests.csv",
    "phase2/primary_subperiods.csv": "phase2/primary_subperiods.csv",
    "phase2/primary_replication.csv": "phase2/primary_replication.csv",
    "phase2/scores.csv": "phase2/scores.csv",
    "phase2/dm_tests.csv": "phase2/dm_tests.csv",
    "phase2/subperiods.csv": "phase2/subperiods.csv",
    "phase2/dnr_benchmark.csv": "phase2/dnr_benchmark.csv",
    "phase2/dm_weighting_check.csv": "phase2/dm_weighting_check.csv",
    "within_issuer_predictors.csv": "within_issuer_predictors.csv",
    "data_summary/panel.csv": "data_summary/panel.csv",
    "data_summary/targets.csv": "data_summary/targets.csv",
    "data_summary/monthly_h1.csv": "data_summary/monthly_h1.csv",
    "replication/bali_final_results.csv": "replication/bali_final_results.csv",
    "replication/validation_results.csv": "replication/validation_results.csv",
}


def compare(reported: pd.DataFrame, rerun: pd.DataFrame) -> tuple[bool, str]:
    reported = reported.drop(columns=[c for c in reported.columns if c in IGNORE])
    rerun = rerun.drop(columns=[c for c in rerun.columns if c in IGNORE])
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
    if not a.size:
        return True, f"all {len(keys):,} values identical"
    return same_nan and diff <= TOL, f"max abs diff {diff:.1e} over {a.size:,} numbers"


def main() -> None:
    start = time.time()
    for rel_reported, rel_rerun in FILES.items():
        reported, rerun = RESULTS_DIR / rel_reported, OUTPUT_DIR / rel_rerun
        if not rerun.exists():
            report(f"skipped {rel_reported}: output/{rel_rerun} not regenerated yet")
            continue
        rep, new = pd.read_csv(reported), pd.read_csv(rerun)
        scope = ""
        if "cell" in rep.columns and "cell" in new.columns and len(new) < len(rep):
            # a partial rerun (some grid cells only): compare just the cells it produced
            keyed = rep.set_index(["cell", *[c for c in ("period",) if c in rep.columns]])
            idx = pd.MultiIndex.from_frame(new[keyed.index.names]) if keyed.index.nlevels > 1 else new["cell"]
            rep = keyed.loc[idx].reset_index()[rep.columns]
            scope = f" ({new['cell'].nunique()} of {pd.read_csv(reported)['cell'].nunique()} cells rerun)"
        ok, detail = compare(rep, new)
        if not ok and rel_reported in KNOWN_DIFF:
            report(f"{rel_reported}: {detail} (known: scikit-learn RF not bit-identical across versions; "
                   f"see exploratory/README.md)")
            continue
        check(ok, f"{rel_reported}: {detail}{scope}")
    log(f"tolerance {TOL:g}")
    finish(start)


if __name__ == "__main__":
    main()
