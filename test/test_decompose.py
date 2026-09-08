"""
Unit tests for src/decompose.py.

Every figure asserted here is computed by hand on a six-row frame:

    - firm 1 holds bonds 1/3/6/10 across two entities, so its firm-month mean is 5 and the
      residuals are -4/-2/+1/+5, giving SS_total 46 = SS_between 36 + SS_within 10;
    - firm 2 holds two bonds under one entity, and is the case that must be excluded before
      any between-entity share is computed.

The between/within split is only meaningful over firm-months holding at least two entities:
a firm-month with one entity contributes to SS_total but nothing to SS_between, so leaving
it in measures how common multi-entity firms are rather than how much of the residual is
entity comovement. `multi_entity` below applies that restriction; one test deliberately
skips it to pin down what the dilution costs.
"""

import pandas as pd
import pytest

from src.decompose import demean, issuer_month_stats, variance_decomp


@pytest.fixture
def toy() -> pd.DataFrame:
    """Six bond-months over one date. `gvkey` is float, as it is in the panel."""

    return pd.DataFrame(
        {
            "date": ["2020-01"] * 6,
            "cusip": ["b1", "b2", "b3", "b4", "b5", "b6"],
            "gvkey": [1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
            "issuer_cusip": ["E1", "E1", "E2", "E2", "E3", "E3"],
            "retx": [1.0, 3.0, 6.0, 10.0, 2.0, 4.0],
        }
    )


def multi_entity(df: pd.DataFrame) -> pd.DataFrame:
    """Restricts `df` to firm-months holding >=2 bonds and >=2 entities."""

    stats = issuer_month_stats(df, "gvkey", entity="issuer_cusip")
    keep = stats.loc[(stats["n_bonds"] >= 2) & (stats["n_entities"] >= 2), "gvkey"]
    return df[df["gvkey"].astype("int64").isin(keep)]


def nested(df: pd.DataFrame) -> dict[str, float]:
    """
    Runs the two-level composition: restrict to multi-entity firm-months, demean within
    firm-month, then split the residual between the entities nested inside each firm.
    """

    sub = multi_entity(df)
    return variance_decomp(
        sub.assign(r=demean(sub, "retx", ["date", "gvkey"])), "r", ["date", "issuer_cusip"]
    )


def test_issuer_month_stats_counts_bonds_and_entities(toy):
    stats = issuer_month_stats(toy, "gvkey", entity="issuer_cusip")
    assert stats["n_bonds"].tolist() == [4, 2]
    assert stats["n_entities"].tolist() == [2, 1]


def test_issuer_month_stats_casts_float_key(toy):
    stats = issuer_month_stats(toy, "gvkey", entity="issuer_cusip")
    assert pd.api.types.is_integer_dtype(stats["gvkey"])


def test_issuer_month_stats_omits_entities_when_not_requested(toy):
    assert "n_entities" not in issuer_month_stats(toy, "issuer_cusip").columns


def test_demean_residual_has_zero_mean(toy):
    resid = demean(multi_entity(toy), "retx", ["date", "gvkey"])
    assert resid.tolist() == [-4.0, -2.0, 1.0, 5.0]
    assert resid.mean() == pytest.approx(0.0, abs=1e-12)


def test_demean_rejects_missing_values(toy):
    holed = toy.assign(retx=toy["retx"].mask(toy["cusip"] == "b1"))
    with pytest.raises(ValueError, match="missing"):
        demean(holed, "retx", ["date", "gvkey"])


def test_variance_decomp_matches_hand_computation(toy):
    d = nested(toy)

    assert d["ss_total"] == pytest.approx(46.0)
    assert d["ss_between"] == pytest.approx(36.0)
    assert d["ss_within"] == pytest.approx(10.0)
    assert d["between_share"] == pytest.approx(36.0 / 46.0)
    assert d["x_bar"] == pytest.approx(0.0, abs=1e-12)
    assert d["n_obs"] == 4
    assert d["n_groups"] == 2


def test_anova_identity_holds(toy):
    d = nested(toy)
    assert d["ss_between"] + d["ss_within"] == pytest.approx(d["ss_total"])


def test_identity_holds_when_grand_mean_is_nonzero(toy):
    # Decomposing raw retx rather than a residual: the grand mean is 5, not 0, so a dropped
    # centring term would show up here and nowhere else -- every other test starts from a
    # demeaned column, where that term is 0 and the mistake is invisible.
    d = variance_decomp(multi_entity(toy), "retx", ["date", "issuer_cusip"])

    assert d["x_bar"] == pytest.approx(5.0)
    assert d["ss_between"] + d["ss_within"] == pytest.approx(d["ss_total"])


def test_single_entity_firm_months_dilute_the_share(toy):
    # Firm 2 adds 2 to SS_total and 0 to SS_between, pulling the share from 0.783 to 0.75
    # purely by being single-entity -- which is why the restriction comes first.
    d = variance_decomp(toy.assign(r=demean(toy, "retx", ["date", "gvkey"])), "r",
                        ["date", "issuer_cusip"])

    assert d["ss_total"] == pytest.approx(48.0)
    assert d["between_share"] == pytest.approx(0.75)


def test_missing_group_key_is_not_silently_dropped(toy):
    # A NaN gvkey must fail loudly, not vanish: groupby drops null keys, so a future
    # vintage with missings would shrink SS_total with nothing in the output saying so.
    holed = toy.assign(gvkey=toy["gvkey"].mask(toy["cusip"] == "b1"))
    with pytest.raises((ValueError, TypeError, pd.errors.IntCastingNaNError)):
        demean(holed, "retx", ["date", "gvkey"])
