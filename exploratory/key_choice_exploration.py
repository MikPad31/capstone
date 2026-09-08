"""
Checks whether `gvkey` or `issuer_cusip` is the better issuer key for the OSBAP panel.

L2 demeans `retx` within issuer-month, and "issuer" is either `gvkey` (Compustat's
consolidated parent firm) or `issuer_cusip` (the bond's own CUSIP-6 prefix, i.e. the
legal issuing entity).

    - reads identifier columns only, not the 341-predictor block so it is rerunnable
    - covers exposure, decomposition, splits, and the power/link-stability costs;
    - [verify] lines are internal identity checks and exit non-zero on failure;
    - [report] lines are measured and never asserted

Run:  DSE4101_DATA_DIR=... python3 -m exploratory.key_choice_evidence
"""

from __future__ import annotations

import resource
import sys
import time

import numpy as np
import pandas as pd

from src.data import load_panel

IDENTIFIER_COLS = ["date", "cusip", "issuer_cusip", "gvkey", "RATING_NUM"]

# The share of rows where issuer_cusip must equal cusip[:6] for us to keep calling
# it "the legal issuing entity" rather than a merged field of unknown provenance.
CUSIP6_MATCH_MIN = 0.999

_failures: list[str] = []


def log(msg: str) -> None:
    """
    Prints a timestamped progress line.

    Parameters
    ----------
    msg : str
        Text to print.

    Returns
    -------
    None
    """

    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check(ok: bool, msg: str) -> bool:
    """
    Records a pass/fail as a [verify] line without aborting the run.
    Failures accumulate in `_failures` and are turned into a non-zero exit by `main`

    Parameters
    ----------
    ok : bool
        Whether the check passed.
    msg : str
        What was checked, stated with the measured quantity in it.

    Returns
    -------
    bool
        `ok`, unchanged, for use in a conditional.
    """

    print(f"[verify] {'PASS' if ok else 'FAIL'} — {msg}", flush=True)
    if not ok:
        _failures.append(msg)
    return ok


def report(msg: str) -> None:
    """
    Prints a measured quantity as a [report] line. Never asserted.

    Parameters
    ----------
    msg : str
        The measurement, including the unit it is denominated in.

    Returns
    -------
    None
    """

    print(f"[report] {msg}", flush=True)


def rss_gb() -> float:
    """
    Returns the peak resident set size of this process in gigabytes.

    Returns
    -------
    float
        Peak RSS in GB.
    """

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on Darwin, KB on Linux
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6


def load_identifiers() -> pd.DataFrame:
    """
    Loads the identifier slice of the panel, joined to realized `retx`.

    Returns
    -------
    pd.DataFrame
        One row per bond-month, carrying `IDENTIFIER_COLS` and
        `retx_realized_return`. The 341-predictor block is never read.
    """

    log("loading panel identifiers (date, cusip, issuer_cusip, gvkey, RATING_NUM)")
    df = load_panel(cols=IDENTIFIER_COLS, targets=("retx",))
    log(f"  rows={len(df):,}  rss={rss_gb():.2f}GB")
    return df


def check_keys(df: pd.DataFrame) -> None:
    """
    Establishes what the two candidate issuer keys actually are: `gvkey` and `issuer_cusip`.
    Checks that `issuer_cusip` is the CUSIP-6 prefix of `cusip` for most rows, and reports
    the cardinality of each key and the nesting ratios.

    Parameters
    ----------
    df : pd.DataFrame
        The identifier slice of the OSBAP panel, loaded by `load_identifiers()`.

    Returns
    -------
    None
    """

    n = len(df)
    report(f"panel: {n:,} bond-month rows")
    report(
        f"dates: {df['date'].min()} .. {df['date'].max()}  "
        f"({df['date'].nunique():,} distinct months)"
    )

    # Prefix the ~20k categories and index back to rows, not 1.1M object strings twice.
    cusip_cat = df["cusip"].astype("category")
    codes = cusip_cat.cat.codes.to_numpy()
    prefix_by_code = cusip_cat.cat.categories.astype(str).str[:6].to_numpy()
    row_prefix = np.where(codes >= 0, prefix_by_code[codes], None)  # code -1 is a missing cusip

    # A rate against a threshold, not equality: one bad CUSIP should report, not abort.
    match = row_prefix == df["issuer_cusip"].astype(str).to_numpy()
    n_mismatch = int((~match).sum())
    match_share = float(match.mean())
    check(
        match_share >= CUSIP6_MATCH_MIN,
        f"issuer_cusip == cusip[:6] for {100 * match_share:.4f}% of {n:,} rows "
        f"({n_mismatch:,} mismatches; threshold {100 * CUSIP6_MATCH_MIN:.1f}%)",
    )

    # A duplicate inflates every group size and SS downstream, silently.
    n_dupes = int(df.duplicated(["date", "cusip"]).sum())
    check(n_dupes == 0, f"(date, cusip) is the unit of observation — {n_dupes:,} duplicate rows")

    # Entities per firm bounds how much the key choice can matter; ~1 would settle it.
    # An upper bound only -- Q1 needs firm-month counts, not distinct-key counts.
    n_gvkey = df["gvkey"].nunique()
    n_issuer_cusip = df["issuer_cusip"].nunique()
    n_cusip = df["cusip"].nunique()
    report(
        f"cardinality: gvkey={n_gvkey:,}  issuer_cusip={n_issuer_cusip:,}  cusip={n_cusip:,}"
    )
    report(
        f"nesting ratios: {n_issuer_cusip / n_gvkey:.2f} entities per firm, "
        f"{n_cusip / n_issuer_cusip:.2f} bonds per entity, "
        f"{n_cusip / n_gvkey:.2f} bonds per firm  [unit: distinct keys, not observations]"
    )

    # 0 on the Oct 2024 panel, so the guide's drop-before-grouping is a no-op here;
    # non-zero means the vintage changed. Dropping belongs in Part 1, where it counts.
    n_gvkey_null = int(df["gvkey"].isna().sum())
    report(
        f"gvkey missing on {n_gvkey_null:,} / {n:,} rows "
        f"({100 * n_gvkey_null / n:.2f}% of bond-months); dtype={df['gvkey'].dtype}"
    )
    n_issuer_null = int(df["issuer_cusip"].isna().sum())
    report(f"issuer_cusip missing on {n_issuer_null:,} / {n:,} rows ({100 * n_issuer_null / n:.2f}%)")


def main() -> None:
    """
    Runs each part in turn, then exits non-zero if any check failed.

    The ~2GB identifier slice is loaded once and passed down; every part reads the
    same columns, so re-loading per part would dominate the runtime.

    Returns
    -------
    None
    """

    start = time.time()
    df = load_identifiers()

    check_keys(df)

    log(f"done in {time.time() - start:.1f}s. peak rss={rss_gb():.2f}GB")
    if _failures:
        log(f"{len(_failures)} check(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
