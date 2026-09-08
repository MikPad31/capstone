# %% Setup
"""
Builds and verifies the expanding-window splitter with an h-month embargo.


At h=12 a signal dated `t` is labelled by returns realised through `t+12`.
A training window ending at `t-1` overlaps its own test label by eleven months.
This script verifies whether the embargo rule `s <= T - h` removes overlap.
    - takes no data as input, fold are pure functions of `months`, `oos_start`, `horizon`, `refit_freq`

Run: python3 -m exploratory.splits_prototype
"""

from __future__ import annotations

import sys
import time
from typing import Callable, NamedTuple, Sequence

import pandas as pd

from src.config import PANEL_START, PANEL_END, HORIZONS, OOS_START, OUTPUT_DIR
from src.splits import Fold, _generate, generate_folds, label_months, leaked_months

# The group has not settled the real OOS_START (`src.config.OOS_START` is deliberately
# None), so every number this script prints for the real calendar is provisional. Kept
# local: putting it in config.py is how a value nobody chose becomes the value everybody
# used.
PROVISIONAL_OOS_START = pd.Period("2015-01", freq="M")

_failures: list[str] = []

# %% Helper functions (copied from exploratory/key_choice_exploration.py)

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

def finish(
        start: float
) -> None:
    """
    Closes out a run: prints elapsed time, then exits non-zero if any `check()` failed.

    No peak-memory line, unlike `key_choice_exploration.py` since nothing here loads the
    panel.
    
    Under a kernel this reports and, on failure, raises `SystemExit` which the
    kernel surfaces as a traceback and survives.

    Parameters
    ----------
    start : float
        The `time.time()` recorded when the run began.

    Returns
    -------
    None
    """

    log(f"done in {time.time() - start:.1f}s.")
    if _failures:
        log(f"{len(_failures)} check(s) failed")
        sys.exit(1)

# %% Splitter — promoted to src/splits.py

# `Fold`, `label_months`, `leaked_months` and `generate_folds` now live in src/splits.py
# and are imported above, so the [verify] blocks below check the module the pipeline
# actually uses rather than a second copy that could drift from it.


def generate_folds_naive(
        months: Sequence[pd.Period],
        oos_start: pd.Period,
        horizon: int,
        refit_freq: int,
) -> list[Fold]:
    """
    Wrapper that generates folds with an embargo equal to 1.
    Intended as a counterfactual to the embargo rule, to verify that it is actually removing overlap.

    Deliberately wrong for h > 1, and deliberately not in src/. It reaches for the private
    `_generate` rather than reimplementing the chunking so that it differs from
    `generate_folds` in exactly one number -- a control that differed in some second
    detail would measure the difference instead of the embargo.

    Parameters
    ----------
    months : Sequence[pd.Period]
        Sequence of months to generate folds from.
    oos_start : pd.Period
        Month to start the out-of-sample period.
    horizon : int
        Number of months ahead that the signal is predicting. Used only to decide which
        test labels are complete -- pointedly not to set the embargo. That is the bug.
    refit_freq : int
        Number of months between refits.

    Returns
    -------
    list[Fold]
        Folds whose training labels overlap their test labels by `horizon - 1` months.

    Raises
    ------
    ValueError
        If the input months sequence is invalid or if the parameters are inconsistent.
    """

    return _generate(months, oos_start, horizon, refit_freq, embargo=1)


def panel_months() -> list[pd.Period]:
    """
    The real signal calendar: 245 contiguous months, 2002-07 … 2022-11.
    """
    return list(pd.period_range(PANEL_START, PANEL_END, freq="M"))



# %% V1 — hand-computed fold

def v1_hand_computed_fold() -> None:
    """
    Checks one fold against arithmetic done on paper, asserting the full month lists.

    The arithmetic:

        months     = 2000-01 … 2001-12                              (24 months)
        oos_start  = 2001-01,  h = 3,  refit_freq = 12
        last test  = months[-1] - h = 2001-12 - 3        = 2001-09
        oos range  = 2001-01 … 2001-09                              (9 months)
        n_folds    = ceil(9 / 12)                        = 1
        last train = T - h = 2001-01 - 3                 = 2000-10
        train      = 2000-01 … 2000-10                              (10 months)
        train labels end at 2000-10 + 3                  = 2001-01
        test labels begin at 2001-01 + 1                 = 2001-02

    The training labels ending on the first test *signal* month reads as an off-by-one
    but is the rule being tight: nothing is wasted and nothing overlaps.

    Returns
    -------
    None
    """

    months = pd.period_range("2000-01", "2001-12", freq="M")
    folds = generate_folds(months, pd.Period("2001-01", freq="M"), horizon=3, refit_freq=12)

    # Guarded: every line below indexes folds[0], so a wrong count must FAIL here rather
    # than IndexError three lines later.
    if not check(len(folds) == 1, f"V1 produces exactly 1 fold (got {len(folds)})"):
        return

    fold = folds[0]
    expected_train = list(pd.period_range("2000-01", "2000-10", freq="M"))
    expected_test = list(pd.period_range("2001-01", "2001-09", freq="M"))

    check(fold.train_months == expected_train,
          f"V1 train months are 2000-01 … 2000-10 ({len(expected_train)} months)")
    check(fold.test_months == expected_test,
          f"V1 test months are 2001-01 … 2001-09 ({len(expected_test)} months)")
    check(fold.fit_month == pd.Period("2000-10", freq="M"),
          f"V1 fit_month is the last train signal 2000-10 (got {fold.fit_month})")
    check(max(label_months(fold.fit_month, 3)) == pd.Period("2001-01", freq="M"),
          "V1 last train label month is 2001-01")
    check(min(label_months(fold.test_months[0], 3)) == pd.Period("2001-02", freq="M"),
          "V1 first test label month is 2001-02")
    check(leaked_months(fold, 3) == set(),
          "V1 fold leaks 0 calendar months")


# %% V2 — negative control

def v2_negative_control(
        oos_start: pd.Period = PROVISIONAL_OOS_START,
        refit_freq: int = 12,
) -> None:
    """
    Runs the leakage guard against a splitter known to leak, then against ours.

    At horizon `h` the naive window's last training signal is `T-1`, whose label runs
    through `T+h-1`, while the test labels start at `T+1`. The intersection is
    `{T+1 … T+h-1}`, exactly `h-1` calendar months per fold, independent of block
    length. The count is asserted rather than mere non-emptiness: a guard that fires
    for the wrong reason is no better than one that never fires.

    At h=1 the naive splitter is correct and leaks nothing, which is why the bug stays
    invisible to anyone who only ever runs one-month horizons.

    Parameters
    ----------
    oos_start : pd.Period, optional
        First test signal month, by default the provisional 2015-01.
    refit_freq : int, optional
        Months between refits, by default 12.

    Returns
    -------
    None
    """

    months = panel_months()
    summary: list[str] = []

    for h in HORIZONS:
        naive = generate_folds_naive(months, oos_start, h, refit_freq)
        real = generate_folds(months, oos_start, h, refit_freq)

        naive_counts = {len(leaked_months(f, h)) for f in naive}
        real_counts = {len(leaked_months(f, h)) for f in real}

        check(naive_counts == {h - 1},
              f"naive splitter at h={h} leaks exactly h-1={h - 1} calendar months in every "
              f"one of its {len(naive)} folds (got {sorted(naive_counts)})")
        if h > 1:
            check(all(leaked_months(f, h) for f in naive),
                  f"naive splitter at h={h} leaks in every fold — the guard fires")
        check(real_counts == {0},
              f"embargoed splitter at h={h} leaks 0 calendar months in all {len(real)} folds")

        summary.append(f"h={h}: {h - 1} → 0")

    report("leaked calendar months per fold, naive → embargoed — " + ", ".join(summary))


# %% V3 — invariants

def v3_invariants(
        refit_freqs: Sequence[int] = (1, 3, 12),
) -> None:
    """
    Asserts every invariant from the `generate_folds` contract across a sweep of
    horizons, refit frequencies, and OOS start months.

    Violations accumulate into one [verify] line rather than one per fold: a check per
    fold would print thousands of lines and bury the failure that matters. Each
    violation string names its case, so a FAIL points at a parameter combination.

    The start months stress the edges rather than the middle — the earliest legal start
    for each horizon (one month of training history) and a start close enough to the
    panel end that the whole OOS range is a single short block are the two that will
    actually find something.

    Parameters
    ----------
    refit_freqs : Sequence[int], optional
        Refit frequencies to sweep, by default (1, 3, 12).

    Returns
    -------
    None
    """

    months = panel_months()
    n_cases = 0
    n_folds_total = 0
    failures: list[str] = []

    for h in HORIZONS:
        starts = [
            months[0] + h,              # earliest legal: exactly one training month
            pd.Period("2010-01", freq="M"),
            PROVISIONAL_OOS_START,
            months[-1] - h - 4,         # 5 usable test months: one short final block
        ]
        for oos_start in starts:
            for refit_freq in refit_freqs:
                n_cases += 1
                folds = generate_folds(months, oos_start, h, refit_freq)
                n_folds_total += len(folds)
                case = f"h={h}, refit_freq={refit_freq}, oos_start={oos_start}"

                if not folds:
                    failures.append(f"{case}: no folds generated")
                    continue

                for i, f in enumerate(folds):
                    if max(f.train_months) + h != min(f.test_months):
                        failures.append(f"{case} fold {i}: embargo gap is not exactly {h}")
                    if max(label_months(max(f.train_months), h)) >= min(
                            label_months(min(f.test_months), h)):
                        failures.append(f"{case} fold {i}: train and test labels overlap")
                    if leaked_months(f, h):
                        failures.append(f"{case} fold {i}: leaked months non-empty")
                    if max(f.test_months) + h > months[-1]:
                        failures.append(f"{case} fold {i}: test label runs past the panel end")
                    if f.fit_month != max(f.train_months):
                        failures.append(f"{case} fold {i}: fit_month is not the last train month")
                    if i and not set(folds[i - 1].train_months) <= set(f.train_months):
                        failures.append(f"{case} fold {i}: train window did not expand")
                    if i and f.test_months[0] != folds[i - 1].test_months[-1] + 1:
                        failures.append(f"{case} fold {i}: test blocks are not adjacent")

                # Outside the fold loop: these two are what make "the test blocks
                # partition the OOS range" mean the whole range and not a piece of it.
                if folds[0].test_months[0] != oos_start:
                    failures.append(f"{case}: first test month is not oos_start")
                if folds[-1].test_months[-1] != months[-1] - h:
                    failures.append(f"{case}: last test month is not panel_end - h")

    check(not failures,
          f"V3 invariants hold across {n_cases} parameter combinations "
          f"({n_folds_total} folds); {len(failures)} violation(s)")
    for f in failures[:10]:
        report(f"V3 violation: {f}")


# %% V4 — degenerate case

def v4_degenerate_case() -> None:
    """
    Checks that h=1, refit_freq=1 reduces to the textbook expanding window.

    The expected folds are built here by hand rather than by comparing against
    `generate_folds_naive`: the two splitters share `_generate`, so at h=1 that
    comparison would be a tautology. This is what shows the embargo is a
    generalisation of the standard splitter and not a replacement for it.

    Returns
    -------
    None
    """

    months = list(pd.period_range("2000-01", "2002-12", freq="M"))
    oos_start = pd.Period("2001-01", freq="M")

    first = months.index(oos_start)
    last = months.index(months[-1] - 1)      # last test signal with a complete label
    expected = [
        Fold(fit_month=months[i - 1], train_months=months[:i], test_months=[months[i]])
        for i in range(first, last + 1)
    ]

    check(generate_folds(months, oos_start, 1, 1) == expected,
          f"V4 h=1, refit_freq=1 reproduces the textbook expanding window "
          f"({len(expected)} single-month folds, 2001-01 … 2002-11)")


# %% V5 — real calendar

def v5_real_calendar(
        oos_start: pd.Period = PROVISIONAL_OOS_START,
        refit_freq: int = 12,
) -> None:
    """
    Runs the splitter on the real 245-month panel calendar and writes the fold tables.

    Expected shape, worked on paper before running (all provisional on `oos_start`):

        h   last test   n_oos   folds@12   fold-0 train
        1   2022-10     94      8          150  (2002-07 … 2014-12)
        3   2022-08     92      8          148  (2002-07 … 2014-10)
        12  2021-11     83      7          139  (2002-07 … 2014-01)

    The h=12 row is the cost of the embargo, and it is two distinct numbers: `h` = 12
    signal months unusable at the tail (a signal needs `t+h <= panel_end`), and `h-1` =
    11 fewer training months at the head than h=1. Neither is the 11 leaked months V2
    reports — that is a third quantity that happens to share a value with the second.

    `n_leaked` is written as a CSV column rather than only asserted, so the fold table
    is self-describing evidence a reviewer can open without rerunning anything.

    Parameters
    ----------
    oos_start : pd.Period, optional
        First test signal month, by default the provisional 2015-01.
    refit_freq : int, optional
        Months between refits, by default 12.

    Returns
    -------
    None
    """

    months = panel_months()
    check(len(months) == 245,
          f"panel calendar is 245 contiguous months, {months[0]} … {months[-1]} "
          f"(got {len(months)})")
    check(OOS_START is None,
          "config.OOS_START is still unset — no module can inherit an undecided split")
    report(f"oos_start={oos_start} is PROVISIONAL: every count below moves if the group "
           f"chooses another start")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for h in HORIZONS:
        folds = generate_folds(months, oos_start, h, refit_freq)
        last_test = months[-1] - h
        n_oos = len([m for m in months if oos_start <= m <= last_test])
        expected_n = -(-n_oos // refit_freq)      # ceil

        check(len(folds) == expected_n,
              f"h={h}: {len(folds)} folds == ceil({n_oos} usable OOS months / {refit_freq})")
        check(len([m for m in months if m > last_test]) == h,
              f"h={h}: the final {h} panel months are unusable, last test signal is {last_test}")

        first, last = folds[0], folds[-1]
        report(f"h={h} fold 0:  train {first.train_months[0]} … {first.train_months[-1]} "
               f"({len(first.train_months)} months) | test {first.test_months[0]} … "
               f"{first.test_months[-1]} ({len(first.test_months)} months)")
        report(f"h={h} fold {len(folds) - 1}:  train {last.train_months[0]} … "
               f"{last.train_months[-1]} ({len(last.train_months)} months) | test "
               f"{last.test_months[0]} … {last.test_months[-1]} "
               f"({len(last.test_months)} months)")

        path = OUTPUT_DIR / f"folds_h{h}.csv"
        pd.DataFrame([
            {
                "fold": i,
                "fit_month": str(f.fit_month),
                "n_train": len(f.train_months),
                "train_start": str(f.train_months[0]),
                "train_end": str(f.train_months[-1]),
                "test_start": str(f.test_months[0]),
                "test_end": str(f.test_months[-1]),
                "n_test": len(f.test_months),
                "n_leaked": len(leaked_months(f, h)),
            }
            for i, f in enumerate(folds)
        ]).to_csv(path, index=False)
        log(f"wrote {path.relative_to(OUTPUT_DIR.parent)} ({len(folds)} folds)")


# %% V6 — preconditions

def _raises(
        fn: Callable[[], object],
        exc: type[BaseException],
) -> bool:
    """
    Returns whether calling `fn` raises `exc`.

    Parameters
    ----------
    fn : Callable[[], object]
        Zero-argument callable to invoke.
    exc : type[BaseException]
        The exception type expected.

    Returns
    -------
    bool
        True if `fn()` raised `exc`.
    """

    try:
        fn()
    except exc:
        return True
    return False


def v6_preconditions() -> None:
    """
    Checks that the four ways of asking for an impossible split raise rather than
    quietly returning something usable-looking.

    Each of these would otherwise surface much later as a fit on zero rows, an empty
    results table, or an embargo silently wider than the horizon.

    Returns
    -------
    None
    """

    months = panel_months()
    gapped = months[:100] + months[101:]

    check(_raises(lambda: generate_folds(gapped, PROVISIONAL_OOS_START, 12, 12), ValueError),
          "a month list with a gap is rejected")
    check(_raises(lambda: generate_folds(months, months[-1], 12, 12), ValueError),
          "an oos_start leaving no test signal with a complete label is rejected")
    check(_raises(lambda: generate_folds(months, months[0], 12, 12), ValueError),
          "an oos_start leaving no training history is rejected")
    check(_raises(lambda: generate_folds(months, pd.Period("1999-01", freq="M"), 1, 1), ValueError),
          "an oos_start outside the panel calendar is rejected")


# %% RUN — V1: the hand-computed fold

_start = time.time()
v1_hand_computed_fold()

# %% RUN — V2: the negative control

v2_negative_control()

# %% RUN — V3: invariants across the sweep

v3_invariants()

# %% RUN — V4: degenerate case

v4_degenerate_case()

# %% RUN — V5: the real calendar, and the fold tables

v5_real_calendar()

# %% RUN — V6: preconditions

v6_preconditions()

# %% RUN — close out

finish(_start)
