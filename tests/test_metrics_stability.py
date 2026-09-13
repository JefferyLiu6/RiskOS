"""PSI and CSI against hand-computed values, plus the empty-bin behaviour."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from riskos.metrics.stability import (
    DEFAULT_EPSILON,
    classify,
    csi,
    csi_for_feature,
    psi,
    psi_from_proportions,
)


def test_psi_two_bins_hand_computed() -> None:
    # PSI = (0.6-0.5)ln(0.6/0.5) + (0.4-0.5)ln(0.4/0.5)
    #     = 0.1(0.1823215568) - 0.1(-0.2231435513) = 0.04054651081
    expected = 0.1 * math.log(1.2) - 0.1 * math.log(0.8)

    result = psi_from_proportions([0.5, 0.5], [0.6, 0.4])

    assert result.value == pytest.approx(expected)
    assert result.value == pytest.approx(0.04054651081, abs=1e-9)
    assert result.band == "stable"


def test_identical_distributions_give_exactly_zero() -> None:
    result = psi_from_proportions([0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25])

    assert result.value == pytest.approx(0.0, abs=1e-15)
    assert result.empty_bins == 0


def test_psi_is_symmetric_in_its_arguments() -> None:
    # (a-e)ln(a/e) is unchanged by swapping a and e — a property worth pinning,
    # since it is why PSI is a distance rather than a divergence.
    forward = psi_from_proportions([0.5, 0.5], [0.6, 0.4]).value
    reverse = psi_from_proportions([0.6, 0.4], [0.5, 0.5]).value

    assert forward == pytest.approx(reverse)


def test_band_boundaries_follow_the_stated_convention() -> None:
    assert classify(0.099) == "stable"
    assert classify(0.10) == "moderate"
    assert classify(0.25) == "moderate"
    assert classify(0.251) == "significant"


def test_empty_bin_is_floored_flagged_and_logged() -> None:
    result = psi_from_proportions([0.5, 0.5], [1.0, 0.0])

    assert result.empty_bins == 1
    assert result.detail["floored"].to_list() == [False, True]
    # The floored bin contributes (eps - 0.5) * ln(eps / 0.5), a large positive
    # number driven by the epsilon rather than by data — hence the warning.
    assert result.value > 1.0
    assert result.detail["contribution"][1] == pytest.approx(
        (DEFAULT_EPSILON - 0.5) * math.log(DEFAULT_EPSILON / 0.5)
    )


def test_proportions_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="not 1"):
        psi_from_proportions([0.5, 0.4], [0.5, 0.5])


def test_bin_count_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="bin count mismatch"):
        psi_from_proportions([0.5, 0.5], [0.4, 0.3, 0.3])


def test_psi_of_a_sample_against_itself_is_zero() -> None:
    rng = np.random.default_rng(20250901)
    scores = rng.uniform(size=5_000)

    result = psi(scores, scores, n_bins=10)

    assert result.value == pytest.approx(0.0, abs=1e-12)
    assert result.n_bins == 10


def test_psi_detects_a_shifted_distribution() -> None:
    rng = np.random.default_rng(20250901)
    reference = rng.normal(0.0, 1.0, size=20_000)
    shifted = rng.normal(1.5, 1.0, size=20_000)

    assert psi(reference, shifted).band == "significant"


def test_bins_come_from_the_reference_not_the_window() -> None:
    # Deciles are fixed by the training reference (build plan §7.5). If the
    # window re-derived its own bins, both sides would be uniform by
    # construction and PSI would collapse to zero.
    reference = np.arange(1000, dtype=float)
    window = np.arange(500, 1500, dtype=float)

    assert psi(reference, window).value > 0.1


def test_csi_ranks_features_by_shift() -> None:
    rng = np.random.default_rng(7)
    reference = pl.DataFrame(
        {"stable_feature": rng.normal(size=5_000), "drifting_feature": rng.normal(size=5_000)}
    )
    actual = pl.DataFrame(
        {
            "stable_feature": rng.normal(size=5_000),
            "drifting_feature": rng.normal(loc=2.0, size=5_000),
        }
    )

    table = csi(reference, actual)

    assert table["feature"][0] == "drifting_feature"
    assert table["band"][0] == "significant"
    assert table.filter(pl.col("feature") == "stable_feature")["band"][0] == "stable"


def test_categorical_csi_buckets_unseen_levels() -> None:
    reference = pl.Series("channel", ["R"] * 60 + ["B"] * 40)
    actual = pl.Series("channel", ["R"] * 50 + ["B"] * 30 + ["NEW"] * 20)

    result = csi_for_feature(reference, actual)

    # Both sentinel bins are always present: __missing__ so a change in the null
    # rate registers as drift, __unseen__ for levels absent from the reference.
    assert result.detail["bin"].to_list() == ["B", "R", "__missing__", "__unseen__"]
    assert result.detail["actual"].to_list() == pytest.approx([0.3, 0.5, 0.0, 0.2])
    # __missing__ is empty on both sides here, __unseen__ only on the reference.
    assert result.empty_bins == 2
