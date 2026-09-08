"""
Unit tests for src/splits.py.

The fold arithmetic asserted here is computed by hand on a 24-month calendar,
2000-01 … 2001-12, with oos_start 2001-01, h=3 and refit_freq=12:

    - the last test signal with a complete label is 2001-12 - 3 = 2001-09, so the usable
      OOS range is 2001-01 … 2001-09 (9 months) and ceil(9/12) = 1 fold;
    - a training signal is admissible iff its label is realised by the first test month,
      s <= 2001-01 - 3 = 2000-10, so the fold trains on 2000-01 … 2000-10;
    - the training labels then end at 2000-10 + 3 = 2001-01 and the test labels begin at
      2001-01 + 1 = 2001-02. Adjacent, no overlap, nothing wasted.

The training labels ending on the first test *signal* month reads as an off-by-one and is
not: the rule is tight rather than conservative.

`naive_folds` below is the negative control -- the textbook `s <= T-1` window, which is
correct at h=1 and leaks h-1 calendar months per fold at every longer horizon. Without it
the leakage assertions are a guard nobody has watched fire.
"""

import pandas as pd
import pytest

from src.config import HORIZONS, OOS_START, PANEL_END, PANEL_START
from src.splits import (
    Fold,
    _generate,
    generate_folds,
    label_months,
    leaked_months,
    rows_for_months,
)

# The group has not settled config.OOS_START (it is deliberately None), so the tests carry
# their own provisional start. A test must never read the real one: that would couple the
# suite to a decision nobody has made.
OOS_START_TEST = pd.Period("2015-01", freq="M")

REFIT_FREQS = (1, 3, 12)


@pytest.fixture
def toy_months() -> list[pd.Period]:
    """24 months, 2000-01 … 2001-12 -- the calendar hand-computed in the module docstring."""

    return list(pd.period_range("2000-01", "2001-12", freq="M"))


@pytest.fixture
def panel_months() -> list[pd.Period]:
    """The real signal calendar. Still no data: months only."""

    return list(pd.period_range(PANEL_START, PANEL_END, freq="M"))


def naive_folds(months, oos_start, horizon, refit_freq) -> list[Fold]:
    """
    Deliberately wrong `s <= T-1` splitter. Lives here, never in src/ -- it exists only so
    the leakage guard can be seen firing.

    Calls the private `_generate` with `embargo=1` rather than reimplementing the chunking,
    so the control differs from the real splitter in exactly one number. A hand-rolled copy
    could drift in some second detail, and then the leakage test would be measuring the
    drift instead of the embargo.
    """

    return _generate(months, oos_start, horizon, refit_freq, embargo=1)


# --- the hand-computed fold -------------------------------------------------------------


def test_v1_hand_computed_fold_matches_paper(toy_months):
    folds = generate_folds(toy_months, pd.Period("2001-01", freq="M"), horizon=3, refit_freq=12)

    assert len(folds) == 1
    assert folds[0].train_months == list(pd.period_range("2000-01", "2000-10", freq="M"))
    assert folds[0].test_months == list(pd.period_range("2001-01", "2001-09", freq="M"))
    assert folds[0].fit_month == pd.Period("2000-10", freq="M")


def test_train_labels_end_where_test_labels_begin(toy_months):
    # The tightness of the rule, stated in label months rather than signal months: one more
    # training month would overlap, one fewer would waste data.
    fold = generate_folds(toy_months, pd.Period("2001-01", freq="M"), 3, 12)[0]

    assert max(label_months(fold.fit_month, 3)) == pd.Period("2001-01", freq="M")
    assert min(label_months(fold.test_months[0], 3)) == pd.Period("2001-02", freq="M")


def test_label_months_spans_t_plus_one_through_t_plus_h():
    # The convention the whole embargo is derived from: verified against
    # dnr_ml_predictions/README.txt, where realized_return_date is t+1 for signal_date t.
    assert label_months(pd.Period("2020-01", freq="M"), 3) == {
        pd.Period("2020-02", freq="M"),
        pd.Period("2020-03", freq="M"),
        pd.Period("2020-04", freq="M"),
    }


# --- the negative control ---------------------------------------------------------------


@pytest.mark.parametrize("horizon", HORIZONS)
def test_naive_splitter_leaks_h_minus_one_months(panel_months, horizon):
    # Asserting the count, not merely non-emptiness: a guard that fires for the wrong reason
    # is no better than one that never fires. At h=1 the naive window is correct and leaks
    # nothing, which is why the bug stays invisible to anyone who only runs one-month
    # horizons.
    folds = naive_folds(panel_months, OOS_START_TEST, horizon, 12)

    assert {len(leaked_months(f, horizon)) for f in folds} == {horizon - 1}


@pytest.mark.parametrize("horizon", [h for h in HORIZONS if h > 1])
def test_naive_splitter_leaks_in_every_fold(panel_months, horizon):
    folds = naive_folds(panel_months, OOS_START_TEST, horizon, 12)

    assert all(leaked_months(f, horizon) for f in folds)


@pytest.mark.parametrize("horizon", HORIZONS)
@pytest.mark.parametrize("refit_freq", REFIT_FREQS)
def test_embargoed_splitter_leaks_nothing(panel_months, horizon, refit_freq):
    folds = generate_folds(panel_months, OOS_START_TEST, horizon, refit_freq)

    assert all(leaked_months(f, horizon) == set() for f in folds)


# --- invariants -------------------------------------------------------------------------


@pytest.mark.parametrize("horizon", HORIZONS)
@pytest.mark.parametrize("refit_freq", REFIT_FREQS)
def test_embargo_gap_is_exactly_horizon(panel_months, horizon, refit_freq):
    # Equality, not `<=`: a conservative splitter would also pass an inequality while
    # throwing away training data, and on a gapped calendar this is what would fail.
    folds = generate_folds(panel_months, OOS_START_TEST, horizon, refit_freq)

    assert all(max(f.train_months) + horizon == min(f.test_months) for f in folds)
    assert all(f.fit_month == max(f.train_months) for f in folds)


@pytest.mark.parametrize("horizon", HORIZONS)
@pytest.mark.parametrize("refit_freq", REFIT_FREQS)
def test_train_windows_expand(panel_months, horizon, refit_freq):
    folds = generate_folds(panel_months, OOS_START_TEST, horizon, refit_freq)

    assert all(
        set(prev.train_months) <= set(cur.train_months)
        for prev, cur in zip(folds, folds[1:])
    )


@pytest.mark.parametrize("horizon", HORIZONS)
@pytest.mark.parametrize("refit_freq", REFIT_FREQS)
def test_test_blocks_partition_oos_range(panel_months, horizon, refit_freq):
    # Adjacency alone would allow the blocks to cover a sub-range of the OOS period; the
    # two endpoint assertions are what make this a partition of the whole usable range.
    folds = generate_folds(panel_months, OOS_START_TEST, horizon, refit_freq)

    assert all(
        cur.test_months[0] == prev.test_months[-1] + 1
        for prev, cur in zip(folds, folds[1:])
    )
    assert folds[0].test_months[0] == OOS_START_TEST
    assert folds[-1].test_months[-1] == panel_months[-1] - horizon


@pytest.mark.parametrize("horizon", HORIZONS)
def test_every_test_label_completes_within_panel(panel_months, horizon):
    # The second constraint, independent of the embargo: a test signal whose label runs past
    # the panel end cannot be scored, so the last h signal months are unusable.
    folds = generate_folds(panel_months, OOS_START_TEST, horizon, 12)
    unusable = [m for m in panel_months if m > panel_months[-1] - horizon]

    assert all(max(f.test_months) + horizon <= panel_months[-1] for f in folds)
    assert len(unusable) == horizon


@pytest.mark.parametrize("horizon", HORIZONS)
def test_fold_count_is_ceil_of_oos_months_over_refit_freq(panel_months, horizon):
    last_test = panel_months[-1] - horizon
    n_oos = len([m for m in panel_months if OOS_START_TEST <= m <= last_test])

    folds = generate_folds(panel_months, OOS_START_TEST, horizon, 12)

    assert len(folds) == -(-n_oos // 12)


def test_final_block_may_be_short(panel_months):
    # 94 usable OOS months at h=1 is not a multiple of 12. Dropping the remainder would
    # discard ten months of out-of-sample evidence and break the partition above.
    folds = generate_folds(panel_months, OOS_START_TEST, 1, 12)

    assert len(folds[-1].test_months) == 10
    assert all(len(f.test_months) == 12 for f in folds[:-1])


# --- degenerate case --------------------------------------------------------------------


def test_h1_rf1_reduces_to_textbook_expanding_window():
    # Built by hand rather than compared against naive_folds: the two share `_generate`, so
    # at h=1 that comparison would be a tautology. This is what shows the embargo is a
    # generalisation of the standard splitter and not a replacement for it.
    months = list(pd.period_range("2000-01", "2002-12", freq="M"))
    oos_start = pd.Period("2001-01", freq="M")

    first = months.index(oos_start)
    last = months.index(months[-1] - 1)
    expected = [
        Fold(fit_month=months[i - 1], train_months=months[:i], test_months=[months[i]])
        for i in range(first, last + 1)
    ]

    assert generate_folds(months, oos_start, 1, 1) == expected


# --- preconditions ----------------------------------------------------------------------


def test_rejects_month_list_with_gaps(panel_months):
    # Contiguity is a precondition, not a relaxed invariant: on a gapped calendar the last
    # admissible training month can precede T-h, silently widening the embargo.
    gapped = panel_months[:100] + panel_months[101:]

    with pytest.raises(ValueError, match="contiguous"):
        generate_folds(gapped, OOS_START_TEST, 12, 12)


def test_rejects_unsorted_month_list(panel_months):
    swapped = panel_months[:50] + [panel_months[51], panel_months[50]] + panel_months[52:]

    with pytest.raises(ValueError, match="ascending"):
        generate_folds(swapped, OOS_START_TEST, 12, 12)


def test_rejects_empty_month_list():
    with pytest.raises(ValueError, match="empty"):
        generate_folds([], OOS_START_TEST, 1, 1)


def test_rejects_oos_start_leaving_no_usable_range(panel_months):
    with pytest.raises(ValueError, match="last test month"):
        generate_folds(panel_months, panel_months[-1], 12, 12)


def test_rejects_oos_start_leaving_no_training_history(panel_months):
    with pytest.raises(ValueError, match="No training months"):
        generate_folds(panel_months, panel_months[0], 12, 12)


def test_rejects_oos_start_outside_the_calendar(panel_months):
    with pytest.raises(ValueError, match="not in the provided months"):
        generate_folds(panel_months, pd.Period("1999-01", freq="M"), 1, 1)


@pytest.mark.parametrize("horizon,refit_freq", [(0, 1), (-1, 1), (1, 0), (1, -1)])
def test_rejects_non_positive_horizon_or_refit_freq(toy_months, horizon, refit_freq):
    with pytest.raises(ValueError, match="positive integers"):
        generate_folds(toy_months, pd.Period("2001-01", freq="M"), horizon, refit_freq)


def test_config_oos_start_is_unset():
    # Guards the decision itself: oos_start is a required argument precisely so that no
    # module inherits an undecided split by accident. When the group settles a value this
    # test is the one that should be deliberately changed.
    assert OOS_START is None


# --- the only function that touches the panel -------------------------------------------


@pytest.fixture
def toy_panel() -> pd.DataFrame:
    """Four bond-months with month-end timestamps, as the real panel stores `date`."""

    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-06-30"]),
            "retx": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_rows_for_months_selects_by_month_not_by_day(toy_panel):
    mask = rows_for_months(toy_panel, [pd.Period("2020-01", freq="M"), pd.Period("2020-02", freq="M")])

    assert mask.tolist() == [True, True, False, False]


def test_rows_for_months_accepts_string_dates():
    # test_decompose.py's fixtures store `date` as "YYYY-MM" strings; both dtypes have to
    # work or the mask silently selects nothing.
    df = pd.DataFrame({"date": ["2020-01", "2020-02", "2020-03"], "retx": [1.0, 2.0, 3.0]})

    assert rows_for_months(df, [pd.Period("2020-03", freq="M")]).tolist() == [False, False, True]


def test_rows_for_months_rejects_an_empty_mask(toy_panel):
    # An all-False mask is almost always a Period/Timestamp dtype mismatch, which would
    # otherwise pass silently and fit a fold on zero rows.
    with pytest.raises(ValueError, match="no rows matched"):
        rows_for_months(toy_panel, [pd.Period("1999-01", freq="M")])


def test_rows_for_months_rejects_a_missing_date_column(toy_panel):
    with pytest.raises(KeyError):
        rows_for_months(toy_panel.rename(columns={"date": "signal_date"}), [pd.Period("2020-01", freq="M")])


def test_rows_for_months_selects_a_real_fold(toy_panel):
    months = list(pd.period_range("2020-01", "2021-12", freq="M"))
    df = pd.DataFrame({"date": pd.PeriodIndex(months, freq="M").to_timestamp(how="end")})
    fold = generate_folds(months, pd.Period("2021-01", freq="M"), 1, 12)[0]

    assert rows_for_months(df, fold.train_months).sum() == len(fold.train_months)
    assert rows_for_months(df, fold.test_months).sum() == len(fold.test_months)
