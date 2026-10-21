"""
Tests for src/config.py: the pinned OOS start and the robustness sub-periods are usable on the
real panel calendar. Months only, no data.
"""

import pandas as pd
import pytest

from src.config import HORIZONS, OOS_START, PANEL_END, PANEL_START, SUBPERIODS
from src.splits import generate_folds


def test_oos_start_is_pinned():
    # Every saved prediction was fitted with this start. Changing it means refitting the
    # whole grid, so this test should only ever change together with a full rerun.
    assert OOS_START == "2015-01"


@pytest.mark.parametrize("horizon", HORIZONS)
@pytest.mark.parametrize("refit_freq", [1, 12])
def test_oos_start_gives_valid_folds_at_every_horizon(horizon, refit_freq):
    months = list(pd.period_range(PANEL_START, PANEL_END, freq="M"))
    folds = generate_folds(months, pd.Period(OOS_START, freq="M"), horizon, refit_freq)

    assert folds[0].test_months[0] == pd.Period(OOS_START, freq="M")
    assert len(folds[0].train_months) >= 10 * 12


def test_subperiods_tile_the_oos_range():
    firsts = [pd.Period(first, freq="M") for _, first, _ in SUBPERIODS]
    lasts = [pd.Period(last, freq="M") for _, _, last in SUBPERIODS]

    assert firsts[0] == pd.Period(OOS_START, freq="M")
    assert lasts[-1] == pd.Period(PANEL_END, freq="M")
    assert all(f <= l for f, l in zip(firsts, lasts))
    # each block starts the month after the previous one ends: no gap, no overlap
    assert all(nxt == prev + 1 for prev, nxt in zip(lasts[:-1], firsts[1:]))


def test_subperiod_names_are_unique():
    names = [name for name, _, _ in SUBPERIODS]
    assert len(names) == len(set(names))
