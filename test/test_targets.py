"""
Unit tests for src/targets.py.

`retx` in these fixtures means `retx_realized_return`-shaped: the value at row `t` is
already the return over `[t, t+1]`, not the return earned during `t`.
This is what makes forward_return's h=1 output look unshifted.
It is the row's own value, not a shift ahead.

Bond A -- six months present, carrying a -103% month:

    month     2020-01  2020-02  2020-03  2020-04  2020-05  2020-06
    retx         0.10    -0.05     0.20    -1.03     0.50     0.00
    gross        1.10     0.95     1.20    -0.03     1.50     1.00

    h=1, signal 2020-01  ->  1.10 - 1                    =  0.10   (own row, no shift)
    h=1, signal 2020-04  ->  -0.03 - 1                   = -1.03
    h=3, signal 2020-01  ->  1.10 * 0.95 * 1.20  - 1     =  0.254
    h=3, signal 2020-02  ->  0.95 * 1.20 * -0.03 - 1     = -1.0342
    h=3, signal 2020-03  ->  1.20 * -0.03 * 1.50 - 1     = -1.054
    h=3, signal 2020-04  ->  -0.03 * 1.50 * 1.00 - 1     = -1.045
    h=3, signals 2020-05, 2020-06                        =  NaN

The h=3 value at 2020-02 is finite despite the -103% month sitting inside its window --
a log1p-based compounding would turn the whole bond's h=3 column to NaN instead. Only
two months are unlabelled at the tail (05, 06), not three: the window is [t, t+1, t+2],
so the last admissible signal only needs to reach two months ahead.

Bond B -- present at 2020-01, 02, 04, 05; no row at 2020-03 (a genuine gap, not a
missing value); retx = 0.10 throughout:

    h=1, signal 2020-01  ->  0.10    own row
    h=1, signal 2020-02  ->  0.10    NOT NaN -- h=1 does not look ahead at all
    h=1, signal 2020-04  ->  0.10
    h=3, signal 2020-01  ->  NaN     window [01,02,03] contains the gap
    h=3, signal 2020-02  ->  NaN     window [02,03,04] contains the gap
    h=3, signal 2020-04  ->  NaN     window [04,05,06] runs off bond B's own calendar

Read the h=1 result at 2020-02 twice: under the old (wrong) window this was NaN, since
h=1 needed the next month's value. Under the corrected window it is the row's own
value and does not depend on 2020-03 -- a gap in the *following* month only matters
starting at h>=2.

`cusip` is "A"*6 / "B"*6 rather than a literal string, purely so both fixtures read as
distinct, cusip-shaped identifiers without hardcoding one.
"""

import numpy as np
import pandas as pd
import pytest

from src.targets import _gross_grid, forward_return, check_h1, label_coverage

RET_COL = "retx_realized_return"

@pytest.fixture
def bond_A() -> pd.DataFrame:
    """Bond A has all six months present, carrying the pathological return."""
    cusip = "A" * 6
    months = ["2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06"]
    retx_realized_return = [0.10, -0.05, 0.20, -1.03, 0.50, 0.00]

    return pd.DataFrame({"date": months, "cusip": cusip, "retx_realized_return": retx_realized_return})

@pytest.fixture
def bond_B() -> pd.DataFrame:
    """Bond B is present at 2020-01, 2020-02, 2020-04, 2020-05, 2020-06, missing 2020-03, retx = 0.10 throughout."""
    cusip = "B" * 6
    months = ["2020-01", "2020-02", "2020-04", "2020-05", "2020-06"]
    retx_realized_return = [0.10, 0.10, 0.10, 0.10, 0.10]

    return pd.DataFrame({"date": months, "cusip": cusip, "retx_realized_return": retx_realized_return})

@pytest.fixture
def both_bonds(bond_A, bond_B) -> pd.DataFrame:
    """Bond A and Bond B together unsorted for the row order test."""
    return pd.concat([bond_A, bond_B], ignore_index=True)

def _at(fwd: pd.Series, df: pd.DataFrame, cusip: str, date: str) -> float:
    mask = (df["cusip"] == cusip) & (df["date"] == date)
    return fwd[mask].iloc[0]
 
 
# ---------------------------------------------------------------------------
# h=1: own row, no shift
# ---------------------------------------------------------------------------
 
 
def test_h1_reproduces_the_input_row_verbatim(bond_A):
    """h=1 is the row's own value, unshifted -- not the next row's."""
    fwd = forward_return(bond_A, RET_COL, 1)
    assert _at(fwd, bond_A, "A" * 6, "2020-01") == pytest.approx(0.10)
    assert _at(fwd, bond_A, "A" * 6, "2020-04") == pytest.approx(-1.03)
 
 
def test_h1_is_unaffected_by_a_gap_in_the_next_month(bond_B):
    """
    The behavior change from the old window: a gap in t+1 cannot poison h=1, because
    h=1 no longer looks past the row itself. This pins that independence so nobody
    "fixes" h=1 back into looking ahead -- reintroducing the old bug this test was
    written to catch.
    """
    fwd = forward_return(bond_B, RET_COL, 1)
    assert _at(fwd, bond_B, "B" * 6, "2020-02") == pytest.approx(0.10)
    assert not pd.isna(_at(fwd, bond_B, "B" * 6, "2020-02"))
 
 
# ---------------------------------------------------------------------------
# h=3: compounding
# ---------------------------------------------------------------------------
 
 
def test_h3_compounds_three_consecutive_rows(bond_A):
    """The four hand-computed h=3 values from the module docstring."""
    fwd = forward_return(bond_A, RET_COL, 3)
    cusip = "A" * 6
    # pytest.approx throughout -- 1.10*0.95*1.20 is not exact in binary
    assert _at(fwd, bond_A, cusip, "2020-01") == pytest.approx(1.10 * 0.95 * 1.20 - 1)
    assert _at(fwd, bond_A, cusip, "2020-02") == pytest.approx(0.95 * 1.20 * -0.03 - 1)
    assert _at(fwd, bond_A, cusip, "2020-03") == pytest.approx(1.20 * -0.03 * 1.50 - 1)
    assert _at(fwd, bond_A, cusip, "2020-04") == pytest.approx(-0.03 * 1.50 * 1.00 - 1)

 
 
def test_negative_gross_return_survives(bond_A):
    """
    Every window touching the -103% month is finite. Asserted over the whole bond,
    not just the row containing it -- a log1p implementation would fail this by
    turning 2020-01's and 2020-02's h=3 labels to NaN too, not just 2020-04's.
    """
    fwd = forward_return(bond_A, RET_COL, 3)
    cusip = "A" * 6
    for date in ["2020-01", "2020-02", "2020-03", "2020-04"]:
        assert not pd.isna(_at(fwd, bond_A, cusip, date)), f"expected finite h=3 label at {date}"

 
 
def test_gap_at_h3_is_not_compounded_across(bond_B):
    """A gap anywhere inside the window poisons the whole product, not just the
    window that starts on the missing month."""
    fwd = forward_return(bond_B, RET_COL, 3)
    cusip = "B" * 6
    assert pd.isna(_at(fwd, bond_B, cusip, "2020-01"))
    assert pd.isna(_at(fwd, bond_B, cusip, "2020-02"))
 
 
def test_tail_loses_h_minus_1_months_not_h(bond_A):
    """Direct pin of the corrected tail arithmetic: h-1 months lost, not h."""
    fwd = forward_return(bond_A, RET_COL, 3)
    cusip = "A" * 6
    assert not pd.isna(_at(fwd, bond_A, cusip, "2020-04"))
    assert pd.isna(_at(fwd, bond_A, cusip, "2020-05"))
    assert pd.isna(_at(fwd, bond_A, cusip, "2020-06"))
 
 
def test_row_order_does_not_matter(both_bonds):
    """The compounding runs along the calendar, not the frame's row order."""
    shuffled = both_bonds.sample(frac=1, random_state=0).reset_index(drop=True)
    fwd_original = forward_return(both_bonds, RET_COL, 3)
    fwd_shuffled = forward_return(shuffled, RET_COL, 3)

    key_original = (both_bonds[["cusip", "date"]].assign(label=fwd_original.values).sort_values(["cusip", "date"]).reset_index(drop=True))
    key_shuffled = (shuffled[["cusip", "date"]].assign(label=fwd_shuffled.values).sort_values(["cusip", "date"]).reset_index(drop=True))

    pd.testing.assert_frame_equal(key_original, key_shuffled)
 
 
def test_compounds_within_bond_only(bond_A, both_bonds):
    """Bond A's labels do not change when bond B's rows are also present."""
    fwd_alone = forward_return(bond_A, RET_COL, 3)
    fwd_together = forward_return(both_bonds, RET_COL, 3)

    alone = (bond_A[["cusip", "date"]].assign(label=fwd_alone.values).sort_values(["cusip", "date"]).reset_index(drop=True))
    together_A = (both_bonds[["cusip", "date"]].assign(label=fwd_together.values)
                  .loc[lambda d: d["cusip"] == "A" * 6]
                  .sort_values(["cusip", "date"]).reset_index(drop=True))

    pd.testing.assert_frame_equal(alone, together_A)
 
 
def test_grid_round_trip_preserves_rows(bond_A):
    """_gross_grid loses or duplicates nothing on a clean frame."""
    grid = _gross_grid(bond_A, RET_COL)
    assert grid.notna().sum().sum() == len(bond_A) == 6
 
 
def test_duplicate_bond_month_raises(bond_A):
    """A repeated (cusip, date) pair is ambiguous for the pivot and must raise."""
    df = pd.concat([bond_A, bond_A.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        _gross_grid(df, RET_COL)
 
 
def test_missing_column_raises_keyerror(bond_A):
    """Each of the three required columns is checked and named individually."""
    with pytest.raises(KeyError):
        _gross_grid(bond_A.drop(columns=[RET_COL]), RET_COL)
    with pytest.raises(KeyError):
        _gross_grid(bond_A.drop(columns=["cusip"]), RET_COL)
    with pytest.raises(KeyError):
        _gross_grid(bond_A.drop(columns=["date"]), RET_COL)
 
 
def test_non_positive_horizon_raises(bond_A):
    """h=0 would be an empty product, not a meaningful label."""
    with pytest.raises(ValueError):
        forward_return(bond_A, RET_COL, 0)
 
 
def test_output_is_named_by_ret_col_and_horizon(bond_A):
    """
    Guards the naming-collision bug: two different targets at the same horizon must
    not produce identically-named Series, or assigning both onto one design frame
    silently overwrites one with the other. Keeping this test stops a future "fix"
    (e.g. simplifying the name to a bare "fwd1") from reintroducing that collision.
    """
    df = bond_A.assign(retxrf=bond_A[RET_COL])
 
    fwd_retx = forward_return(df, RET_COL, 1)
    fwd_retxrf = forward_return(df, "retxrf", 1)
 
    assert fwd_retx.name == f"{RET_COL}_fwd1"
    assert fwd_retxrf.name == "retxrf_fwd1"
    assert fwd_retx.name != fwd_retxrf.name
 
 
# ---------------------------------------------------------------------------
# check_h1
# ---------------------------------------------------------------------------
 
 
def test_check_h1_is_exact_and_has_no_one_sided_rows(bond_A):
    """At h=1 both sides come from the same column, so agreement is exact and the
    one-sided counts are exactly 0 -- not merely close to it."""
    result = check_h1(bond_A, RET_COL)
    assert result["max_abs_diff"] == 0.0
    assert result["n_ours_only"] == 0
    assert result["n_theirs_only"] == 0
 
 
# ---------------------------------------------------------------------------
# label_coverage
# ---------------------------------------------------------------------------
 
 
def test_label_coverage_counts_bond_months(bond_A):
    """One row per signal month; tail months show reduced coverage at h=3."""
    fwd = forward_return(bond_A, RET_COL, 3)
    cov = label_coverage(fwd, bond_A)

    assert len(cov) == 6
    assert cov["n_rows"].dtype == np.int64
    assert cov["n_labelled"].dtype == np.int64
    assert cov["share_labelled"].dtype == np.float64

    by_month = cov.set_index("month")["share_labelled"]
    assert by_month[pd.Period("2020-05", freq="M")] == 0.0
    assert by_month[pd.Period("2020-06", freq="M")] == 0.0
    for date in ["2020-01", "2020-02", "2020-03", "2020-04"]:
        assert by_month[pd.Period(date, freq="M")] == 1.0