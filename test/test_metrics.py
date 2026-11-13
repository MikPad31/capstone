"""
Unit tests for src/metrics.py.

Quantiles are asserted at 25/75 rather than the 1/99 defaults, so every bound is a value
that can be read off the fixture. Pandas interpolates linearly between order statistics:
for n values the q-th quantile sits at position q*(n-1), so on a five-row group q25 lands
exactly on the second-smallest and q75 exactly on the second-largest.

    - `panel` holds two months of five bonds. Month m1 is [1, 2, 3, 4, 100] -> bounds 2 and
      4; month m2 is [10, 20, 30, 40, 50] -> bounds 20 and 40. The months are interleaved
      and out of order, which is what makes the index-preservation test meaningful.
    - `two_models` holds one month and two models whose prediction scales differ. Keyed by
      month alone the ten values pool to bounds 3.25 and 37.5 (positions 2.25 and 6.75 of
      the sorted pool); keyed by month and model each five-row group keeps its own bounds.
      The gap between those two results is the reason `key_cols` is a parameter.

The oos_r2 figures are equally small: r = [1, 2, 3] against p = [1, 2, 4] gives SSE 1, and
a benchmark of 2 gives SST 2, so R^2 is 0.5.
"""

import numpy as np
import pandas as pd
import pytest

from src.metrics import oos_r2, winsorize_cross_section


@pytest.fixture
def panel() -> pd.DataFrame:
    """Ten bond-months over two dates, deliberately interleaved rather than grouped."""

    return pd.DataFrame(
        {
            "signal_date": ["m1", "m2"] * 5,
            "v": [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 100.0, 50.0],
        }
    )


@pytest.fixture
def two_models() -> pd.DataFrame:
    """One month, two models, an order of magnitude apart in scale."""

    return pd.DataFrame(
        {
            "signal_date": ["m1"] * 10,
            "model_key": ["A"] * 5 + ["B"] * 5,
            "p": [1.0, 2.0, 3.0, 4.0, 100.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


def test_winsorize_clips_to_hand_computed_bounds(panel):
    out = winsorize_cross_section(panel, "v", ["signal_date"], lower=0.25, upper=0.75)

    assert out[panel["signal_date"] == "m1"].tolist() == [2.0, 2.0, 3.0, 4.0, 4.0]
    assert out[panel["signal_date"] == "m2"].tolist() == [20.0, 20.0, 30.0, 40.0, 40.0]


def test_winsorize_preserves_row_order(panel):
    # Grouping the frame and concatenating the results returns rows in group order, so the
    # months come back separated and the caller's positional alignment is silently wrong.
    out = winsorize_cross_section(panel, "v", ["signal_date"], lower=0.25, upper=0.75)

    assert out.index.tolist() == panel.index.tolist()


def test_winsorize_returns_a_series_for_a_single_group():
    # groupby(...).apply infers its return shape from the results
    # and returns hands a transposed DataFrame when there is only one group
    one_month = pd.DataFrame({"signal_date": ["m1"] * 5, "v": [1.0, 2.0, 3.0, 4.0, 100.0]})
    out = winsorize_cross_section(one_month, "v", ["signal_date"], lower=0.25, upper=0.75)

    assert isinstance(out, pd.Series)
    assert out.tolist() == [2.0, 2.0, 3.0, 4.0, 4.0]


def test_winsorize_is_idempotent(panel):
    # Clipping to bounds drawn from an already-clipped population must find the same bounds.
    first_pass = winsorize_cross_section(panel, "v", ["signal_date"], lower=0.25, upper=0.75)
    second_pass = winsorize_cross_section(
        panel.assign(v=first_pass), "v", ["signal_date"], lower=0.25, upper=0.75
    )

    assert second_pass.tolist() == first_pass.tolist()


def test_winsorize_leaves_single_row_groups_alone():
    lonely = pd.DataFrame({"signal_date": ["m1", "m2"], "v": [1.0, 500.0]})
    out = winsorize_cross_section(lonely, "v", ["signal_date"], lower=0.25, upper=0.75)

    assert out.tolist() == [1.0, 500.0]


def test_winsorize_does_not_mutate_its_input(panel):
    before = panel.copy(deep=True)
    winsorize_cross_section(panel, "v", ["signal_date"], lower=0.25, upper=0.75)

    pd.testing.assert_frame_equal(panel, before)


def test_winsorize_returns_float64_from_a_float32_column(panel):
    # The variance-share regeneration compares against a figure computed in float64;
    # a narrower input dtype would move the fourth digit and look like a convention mismatch.
    out = winsorize_cross_section(
        panel.astype({"v": "float32"}), "v", ["signal_date"], lower=0.25, upper=0.75
    )

    assert out.dtype == "float64"


def test_winsorize_keys_are_not_interchangeable(two_models):
    # Each model has its own prediction scale and must be clipped against its own bounds.
    # Test whether key_cols is actually used by comparing the result of a pooled clip against a per-model clip.
    pooled = winsorize_cross_section(two_models, "p", ["signal_date"], lower=0.25, upper=0.75)
    per_model = winsorize_cross_section(
        two_models, "p", ["signal_date", "model_key"], lower=0.25, upper=0.75
    )

    assert pooled.tolist() == [3.25, 3.25, 3.25, 4.0, 37.5, 10.0, 20.0, 30.0, 37.5, 37.5]
    assert per_model.tolist() == [2.0, 2.0, 3.0, 4.0, 4.0, 20.0, 20.0, 30.0, 40.0, 40.0]


def test_winsorize_rejects_missing_values(panel):
    holed = panel.assign(v=panel["v"].mask(panel.index == 0))

    with pytest.raises(ValueError, match="missing"):
        winsorize_cross_section(holed, "v", ["signal_date"])


def test_winsorize_rejects_inverted_bounds(panel):
    with pytest.raises(ValueError, match="below"):
        winsorize_cross_section(panel, "v", ["signal_date"], lower=0.9, upper=0.1)


def test_oos_r2_matches_hand_computation():
    r = pd.Series([1.0, 2.0, 3.0])
    p = pd.Series([1.0, 2.0, 4.0])

    assert oos_r2(r, p, 2.0) == pytest.approx(0.5)


def test_oos_r2_scores_a_perfect_prediction_as_one():
    r = pd.Series([0.01, -0.02, 0.03])

    assert oos_r2(r, r.copy(), 0.0) == pytest.approx(1.0)


def test_oos_r2_scores_the_benchmark_itself_as_zero():
    # Predicting the benchmark everywhere makes SSE and SST the same sum by construction.
    r = pd.Series([1.0, 2.0, 3.0])

    assert oos_r2(r, pd.Series([2.0, 2.0, 2.0]), 2.0) == pytest.approx(0.0)


def test_oos_r2_returns_a_python_float():
    r = pd.Series([1.0, 2.0, 3.0])

    assert type(oos_r2(r, r.copy(), 0.0)) is float


def test_oos_r2_matches_the_correlation_identity():
    # Under demeaned inputs, zero-benchmark R^2 = 2ck - k^2,
    # c = corr(r,p), k = std(p)/std(r).
    # Computed separately from SSE formula to verify correctness of arithmetic implementation.

    rng = np.random.default_rng(0)
    r = pd.Series(rng.standard_normal(2000) * 0.05)
    p = pd.Series(rng.standard_normal(2000) * 0.01)
    r, p = r - r.mean(), p - p.mean()

    c = float(np.corrcoef(r, p)[0, 1])
    k = float(p.std(ddof=0) / r.std(ddof=0))

    assert oos_r2(r, p, 0.0) == pytest.approx(2 * c * k - k**2)


def test_oos_r2_rejects_misaligned_inputs():
    # Pandas aligns on index, so a difference between disjoint series is entirely NaN which
    # sums to 0.0 and reports a perfect score for two series with completely misaligned rows.
    r = pd.Series([0.01, 0.02, 0.03], index=[0, 1, 2])
    p = pd.Series([9.9, -9.9, 9.9], index=[5, 6, 7])

    with pytest.raises(ValueError, match="aligned"):
        oos_r2(r, p, 0.0)


def test_oos_r2_rejects_length_mismatch():
    r = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="rows"):
        oos_r2(r, pd.Series([1.0, 2.0]), 0.0)


@pytest.mark.parametrize("to_array", [False, True], ids=["series", "numpy"])
def test_oos_r2_rejects_missing_values(to_array):
    # Series.sum() skips NaN while numpy's does not, so without a guard the same data
    # scores differently depending on which type the caller happened to pass.
    r = pd.Series([1.0, np.nan, 3.0])
    p = pd.Series([1.0, 2.0, 3.0])

    if to_array:
        r, p = r.to_numpy(), p.to_numpy()

    with pytest.raises(ValueError, match="missing"):
        oos_r2(r, p, 0.0)


def test_oos_r2_rejects_a_zero_denominator():
    # Every realized return equal to the benchmark. 
    # Returning -inf here would read as a formatting artefact in a results table rather than as a broken cell.
    flat = pd.Series([0.5, 0.5, 0.5])

    with pytest.raises(ValueError, match="benchmark"):
        oos_r2(flat, pd.Series([0.4, 0.4, 0.4]), 0.5)
