"""Brier decomposition, Wilson intervals, and the reliability curve."""

from __future__ import annotations

import numpy as np
import pytest

from riskos.metrics.calibration import (
    brier_decomposition,
    brier_score,
    observed_vs_expected,
    reliability_curve,
    wilson_interval,
)


def test_brier_score_hand_computed() -> None:
    # ((0.1-0)^2 + (0.9-1)^2) / 2 = (0.01 + 0.01) / 2 = 0.01
    assert brier_score([0, 1], [0.1, 0.9]) == pytest.approx(0.01)


def test_brier_bounds() -> None:
    assert brier_score([0, 1], [0.0, 1.0]) == pytest.approx(0.0)  # perfect
    assert brier_score([0, 1], [1.0, 0.0]) == pytest.approx(1.0)  # maximally wrong
    assert brier_score([0, 1], [0.5, 0.5]) == pytest.approx(0.25)  # uninformative


def test_decomposition_identity_hand_computed() -> None:
    # Two observations in two bins. Bin 1: p=0.1, observed 0. Bin 2: p=0.9,
    # observed 1. Base rate 0.5.
    #   reliability = [(0.1-0)^2 + (0.9-1)^2] / 2 = 0.01
    #   resolution  = [(0-0.5)^2 + (1-0.5)^2] / 2 = 0.25
    #   uncertainty = 0.5 * 0.5                   = 0.25
    #   BS = 0.01 - 0.25 + 0.25 = 0.01
    result = brier_decomposition([0, 1], [0.1, 0.9], n_bins=2)

    assert result.reliability == pytest.approx(0.01)
    assert result.resolution == pytest.approx(0.25)
    assert result.uncertainty == pytest.approx(0.25)
    assert result.brier == pytest.approx(0.01)
    assert result.residual == pytest.approx(0.0, abs=1e-12)


def test_uncertainty_depends_only_on_the_outcome() -> None:
    # Base rate 0.25 gives uncertainty 0.25 * 0.75 = 0.1875 regardless of the
    # forecast, which is why it is not a measure of model quality.
    y = [0, 0, 0, 1]
    for prob in (0.25, 0.9):
        result = brier_decomposition(y, [prob] * 4, n_bins=2)
        assert result.uncertainty == pytest.approx(0.1875)


def test_a_constant_forecast_has_no_resolution() -> None:
    result = brier_decomposition([0, 0, 1, 1], [0.5] * 4, n_bins=4)

    assert result.resolution == pytest.approx(0.0, abs=1e-12)
    assert result.reliability == pytest.approx(0.0, abs=1e-12)


def test_a_well_calibrated_forecast_has_near_zero_reliability() -> None:
    rng = np.random.default_rng(20250901)
    p = rng.uniform(0.01, 0.5, size=200_000)
    y = rng.binomial(1, p)

    result = brier_decomposition(y, p, n_bins=10)

    assert result.reliability < 1e-4
    assert abs(result.residual) < 0.05


def test_a_biased_forecast_shows_up_as_reliability_not_resolution() -> None:
    # Doubling every prediction leaves the ranking, and therefore resolution,
    # essentially untouched while reliability deteriorates. This is the
    # "good classifier, bad risk model" failure the project is about.
    rng = np.random.default_rng(20250901)
    p = rng.uniform(0.01, 0.4, size=100_000)
    y = rng.binomial(1, p)

    honest = brier_decomposition(y, p, n_bins=10)
    biased = brier_decomposition(y, np.clip(p * 2, 0, 1), n_bins=10)

    assert biased.reliability > 50 * honest.reliability
    assert biased.resolution == pytest.approx(honest.resolution, rel=0.05)


def test_wilson_interval_hand_computed() -> None:
    # 2 successes in 4 trials, z = 1.96:
    #   centre = (0.5 + 3.8416/8) / (1 + 3.8416/4) = 0.9802 / 1.9604 = 0.5
    #   half   = (1.96/1.9604) * sqrt(0.25/4 + 3.8416/64) = 0.3499644
    low, high = wilson_interval(2, 4, z=1.96)

    assert low == pytest.approx(0.1500356, abs=1e-6)
    assert high == pytest.approx(0.8499644, abs=1e-6)


def test_wilson_interval_stays_inside_zero_one_at_the_boundary() -> None:
    # The normal approximation would give a negative lower bound here.
    low, high = wilson_interval(0, 10, z=1.96)

    assert low == pytest.approx(0.0, abs=1e-6)
    assert high == pytest.approx(0.2775, abs=1e-4)


def test_wilson_narrows_as_the_sample_grows() -> None:
    small = wilson_interval(50, 100)
    large = wilson_interval(5_000, 10_000)

    assert (large[1] - large[0]) < (small[1] - small[0])


def test_wilson_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match="outside"):
        wilson_interval(5, 4)
    with pytest.raises(ValueError, match="trials must be positive"):
        wilson_interval(0, 0)


def test_reliability_curve_shape_and_coverage() -> None:
    rng = np.random.default_rng(20250901)
    p = rng.uniform(0.01, 0.5, size=100_000)
    y = rng.binomial(1, p)

    curve = reliability_curve(y, p, n_bins=10)

    assert curve.height == 10
    assert curve["n"].sum() == 100_000
    assert (curve["mean_predicted"].diff().drop_nulls() > 0).all()  # bins ordered by risk
    # A calibrated forecast should sit inside its own interval in most bins.
    assert curve["within_interval"].sum() >= 9


def test_reliability_curve_flags_a_miscalibrated_model() -> None:
    rng = np.random.default_rng(20250901)
    p = rng.uniform(0.01, 0.4, size=50_000)
    y = rng.binomial(1, p)

    curve = reliability_curve(y, np.clip(p * 2, 0, 1), n_bins=10)

    assert curve["within_interval"].sum() == 0
    assert (curve["mean_predicted"] > curve["observed_rate"]).all()  # over-predicts


def test_observed_vs_expected_ratio_direction() -> None:
    # Predicting 5% against a realised 10% under-predicts: ratio 2.
    result = observed_vs_expected([1] * 10 + [0] * 90, [0.05] * 100)

    assert result["observed_rate"] == pytest.approx(0.10)
    assert result["expected_rate"] == pytest.approx(0.05)
    assert result["ratio"] == pytest.approx(2.0)


def test_nan_input_is_rejected_rather_than_dropped() -> None:
    with pytest.raises(ValueError, match="NaN"):
        brier_score([0, 1], [0.5, float("nan")])
