"""
Checks whether `gvkey` or `issuer_cusip` is the better issuer key for the OSBAP panel.

L2 demeans `retx` within issuer-month, and "issuer" is either `gvkey` (Compustat's
consolidated parent firm) or `issuer_cusip` (the bond's own CUSIP-6 prefix, i.e. the
legal issuing entity).

    - reads identifier columns only, not the 341-predictor block so it is rerunnable
    - covers exposure, decomposition, splits, and the power/link-stability costs;
    - [verify] lines are internal identity checks and exit non-zero on failure;
    - [report] lines are measured and never asserted

The `# %%` markers make this runnable two ways off the same source:

    - as cells (VS Code / PyCharm), against a live kernel, so the ~2GB identifier
      slice is loaded once and every later cell reuses it in memory;
    - as a script, top to bottom, exiting non-zero if any [verify] line failed.

The RUN cells at the bottom are the script's execution path — there is no `main()`,
so the two ways cannot drift apart.

Run:  DSE4101_DATA_DIR=... python3 -m exploratory.key_choice_exploration
"""

# %% setup

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


# %% helpers — logging, checks, memory


def log(
        msg: str
) -> None:
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


def check(
        ok: bool,
        msg: str
) -> bool:
    """
    Records a pass/fail as a [verify] line without aborting the run.
    Failures accumulate in `_failures` and are turned into a non-zero exit by `finish`.

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


def report(
        msg: str
) -> None:
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


def finish(
        start: float
) -> None:
    """
    Closes out a run: prints elapsed time and peak memory, then exits non-zero if
    any `check()` failed.

    Under a kernel this reports and, on failure, raises `SystemExit` — which the
    kernel surfaces as a traceback and survives, so the loaded panel is not lost.

    Parameters
    ----------
    start : float
        The `time.time()` recorded when the run began.

    Returns
    -------
    None
    """

    log(f"done in {time.time() - start:.1f}s. peak rss={rss_gb():.2f}GB")
    if _failures:
        log(f"{len(_failures)} check(s) failed")
        sys.exit(1)


# %% Loading


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


# %% part 0 — what the two candidate keys actually are


def check_keys(
    df: pd.DataFrame
) -> None:
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

# %% group sizes per key


def group_stats(
        df: pd.DataFrame, 
        key: str
) -> pd.DataFrame:
    """
    Computes group-level statistics for a given key.

    Parameters
    ----------
    df : pd.DataFrame
        The identifier slice of the OSBAP panel.
    key : str
        The column name to group by.

    Returns
    -------
    pd.DataFrame
        A DataFrame with group-level statistics.
    """
    # assign a copy of the DataFrame to avoid SettingWithCopyWarning
    keyed = df if key != "gvkey" else df.assign(gvkey=df["gvkey"].astype("int64"))

    # safe to use size() == nunique("cusip") since (date, cusip) is unique in the panel; no duplicates.
    grouped = keyed.groupby(["date", key], observed=True)
    stats = grouped.size().rename("n_bonds").to_frame()

    if key != "issuer_cusip":
        stats["n_entities"] = grouped["issuer_cusip"].nunique()

    return stats.reset_index()


# %% Checking exposure


def check_exposure(
        df: pd.DataFrame
) -> None:
    """
    Checks how much of the panel has multiple issuer_cusip entities per gvkey firm.
        - check eligibility pass against the invariants `group_stats` relies on
        - report exposure:
            - as a share of firm-months
            - as a share of bond-month observations

    Parameters
    ----------
    df : pd.DataFrame
        The identifier slice of the OSBAP panel, loaded by `load_identifiers()`.

    Returns
    -------
    None
    """

    grp_stat = group_stats(df, "gvkey")
    # check if entities outnumber bonds
    check((grp_stat["n_entities"] <= grp_stat["n_bonds"]).all(), "n_entities <= n_bonds in every gvkey-month")

    # check that single-bond gvkey-months have exactly 1 entity
    check(
        (grp_stat.loc[grp_stat["n_bonds"] == 1, "n_entities"] == 1).all(),
        "single-bond gvkey-months have exactly 1 entity",
    )

    #check that the sum of n_bonds equals the total number of rows in df
    check(int(grp_stat["n_bonds"].sum()) == len(df), "group_stats n_bonds sums to the panel row count")

    # Single-bond firm-months can't be multi-entity
    # including them would measure how many firms have >=2 bonds, not contamination.
    eligible = grp_stat[grp_stat["n_bonds"] >= 2]
    n_eligible_obs = int(eligible["n_bonds"].sum())
    report(
        f"gvkey coverage: {len(eligible):,} / {len(grp_stat):,} firm-months eligible (>=2 bonds), "
        f"{n_eligible_obs:,} / {len(df):,} bond-months"
    )

    multi = eligible[eligible["n_entities"] >= 2]
    n_multi_obs = int(multi["n_bonds"].sum())
    report(
        f"exposure (firm-months): {100 * len(multi) / len(eligible):.1f}% "
        f"({len(multi):,} / {len(eligible):,} eligible firm-months hold >=2 distinct issuer_cusip)"
    )
    report(
        f"exposure (bond-months): {100 * n_multi_obs / n_eligible_obs:.1f}% "
        f"({n_multi_obs:,} / {n_eligible_obs:,} eligible bond-months sit in a multi-entity firm-month)"
    )


# %% RUN — load the panel once per kernel
#
# The ~2GB identifier slice is loaded here and reused by every cell below; every
# part reads the same columns, so re-loading per part would dominate the runtime.
# Rerun this cell only to pick up a new data vintage.

_start = time.time()
df = load_identifiers()

# %% RUN — part 0: What the two candidate keys actually are

check_keys(df)

# %% RUN — part 1: Checking exposure

check_exposure(df)

# %% RUN — close out

finish(_start)
