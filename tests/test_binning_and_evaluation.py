"""Phase 3: IV screening logic, config guards, and the fitted-model exhibits."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from riskos.features import binning
from riskos.panel.config import BinningConfig, feature_config

FIGURES = Path("reports/figures")
METRICS_CSV = FIGURES / "metrics_by_split.csv"

needs_model = pytest.mark.skipif(
    not METRICS_CSV.exists(), reason="scorecard not fitted; run `make train`"
)


# --------------------------------------------------------------------------
# Config guards — a relaxed control must be a documented one.
# --------------------------------------------------------------------------


def test_iv_bands_follow_the_conventional_reading() -> None:
    assert binning.classify_iv(0.01) == "unpredictive"
    assert binning.classify_iv(0.05) == "weak"
    assert binning.classify_iv(0.20) == "medium"
    assert binning.classify_iv(0.40) == "strong"
    assert binning.classify_iv(1.20) == "suspiciously_strong"


def test_a_monotone_constraint_requires_a_written_rationale() -> None:
    with pytest.raises(ValueError, match="without a rationale"):
        BinningConfig(
            max_n_bins=6,
            min_n_bins=2,
            min_bin_size=0.02,
            min_prebin_size=0.005,
            iv_floor=0.02,
            iv_leakage_ceiling=0.9,
            exclude_high_cardinality=(),
            monotonic_trend={"original_ltv": "ascending"},
            monotonic_trend_rationale={},
        )


def test_a_bin_size_override_requires_a_written_rationale() -> None:
    with pytest.raises(ValueError, match="override without a rationale"):
        BinningConfig(
            max_n_bins=6,
            min_n_bins=2,
            min_bin_size=0.02,
            min_prebin_size=0.005,
            iv_floor=0.02,
            iv_leakage_ceiling=0.9,
            exclude_high_cardinality=(),
            monotonic_trend={},
            monotonic_trend_rationale={},
            min_bin_size_overrides={"some_feature": 0.001},
        )


def test_monotone_direction_must_be_a_recognised_value() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        BinningConfig(
            max_n_bins=6,
            min_n_bins=2,
            min_bin_size=0.02,
            min_prebin_size=0.005,
            iv_floor=0.02,
            iv_leakage_ceiling=0.9,
            exclude_high_cardinality=(),
            monotonic_trend={"x": "upwards"},
            monotonic_trend_rationale={"x": "because"},
        )


def test_the_real_config_declares_the_economically_known_directions() -> None:
    trends = feature_config().binning.monotonic_trend

    # Build plan §7.6: "higher LTV must not produce lower risk".
    assert trends["original_ltv"] == "ascending"
    assert trends["original_cltv"] == "ascending"
    assert trends["original_dti"] == "ascending"
    assert trends["credit_score"] == "descending"


def test_high_iv_features_are_cleared_individually_not_by_raising_the_ceiling() -> None:
    """The ceiling is a tripwire; relaxing it wholesale would make it decoration."""
    cfg = feature_config().binning

    assert cfg.iv_leakage_ceiling == 0.9
    assert cfg.cleared_high_iv == frozenset({"credit_score", "current_loan_delinquency_status"})
    for entry in cfg.iv_cleared:
        assert entry["evidence"], f"{entry['feature']} cleared without evidence"


def test_duplicate_features_are_recorded_with_evidence_not_silently_dropped() -> None:
    cfg = feature_config().binning

    assert "current_interest_rate" in cfg.duplicate_drops
    for entry in cfg.known_duplicates:
        assert entry["keep"] and entry["evidence"]


# --------------------------------------------------------------------------
# Fitted-model exhibits.
# --------------------------------------------------------------------------


@needs_model
def test_metrics_exist_for_every_split() -> None:
    metrics = pl.read_csv(METRICS_CSV)

    assert set(metrics["split"]) >= {
        "train",
        "validation_in_time",
        "oot_stress",
        "oot_benign",
    }


@needs_model
def test_discrimination_degrades_out_of_time_but_survives() -> None:
    by_split = {r["split"]: r for r in pl.read_csv(METRICS_CSV).to_dicts()}

    assert by_split["train"]["auc"] > by_split["oot_stress"]["auc"]
    assert by_split["oot_stress"]["gini"] > 0.5, "ranking should survive the regime change"


@needs_model
def test_calibration_is_sound_in_period_and_fails_out_of_period() -> None:
    """The project's headline finding, asserted rather than asserted-in-prose."""
    by_split = {r["split"]: r for r in pl.read_csv(METRICS_CSV).to_dicts()}

    assert 0.9 < by_split["train"]["observed_over_expected"] < 1.1
    assert 0.9 < by_split["validation_in_time"]["observed_over_expected"] < 1.15
    # Under regime change the model materially under-predicts default.
    assert by_split["oot_stress"]["observed_over_expected"] > 2.0


@needs_model
def test_calibration_degrades_far_more_than_discrimination() -> None:
    """Quantifies the argument against selecting on AUC.

    Gini falls by a fraction; the observed/expected ratio more than triples.
    """
    by_split = {r["split"]: r for r in pl.read_csv(METRICS_CSV).to_dicts()}
    gini_retained = by_split["oot_stress"]["gini"] / by_split["train"]["gini"]
    calibration_error = by_split["oot_stress"]["observed_over_expected"]

    assert gini_retained > 0.75, "discrimination retains most of its power"
    assert calibration_error > 3.0, "calibration does not"


@needs_model
def test_removing_covid_forbearance_materially_improves_the_benign_split() -> None:
    """Finding F-001's sensitivity test, as an assertion."""
    by_split = {r["split"]: r for r in pl.read_csv(METRICS_CSV).to_dicts()}
    as_is = by_split["oot_benign"]
    corrected = by_split["oot_benign_ex_forbearance"]

    assert corrected["auc"] > as_is["auc"] + 0.05
    assert corrected["observed_over_expected"] < as_is["observed_over_expected"] / 2


@needs_model
def test_every_declared_monotone_constraint_actually_holds() -> None:
    """optbinning is *asked* for monotonicity; this checks it was delivered."""
    report = pl.read_csv(FIGURES / "monotonicity_check.csv")

    failed = report.filter(~pl.col("holds"))["feature"].to_list()
    assert not failed, f"declared monotone but fitted non-monotone: {failed}"


@needs_model
def test_no_surviving_coefficient_has_the_wrong_sign() -> None:
    coefficients = pl.read_csv(FIGURES / "scorecard_coefficients.csv")

    assert coefficients["sign_ok"].all()
    assert (coefficients["coefficient"] < 0).all()


@needs_model
def test_reliability_curves_and_binning_tables_are_written_for_the_report() -> None:
    # Build plan §9: the report regenerates from CSVs without refitting.
    assert (FIGURES / "reliability_by_split.png").exists()
    assert (FIGURES / "reliability_oot_stress.csv").exists()
    assert (FIGURES / "information_values.csv").exists()
    assert list((FIGURES / "binning").glob("bins_*.csv"))
