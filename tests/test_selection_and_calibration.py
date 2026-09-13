"""Phase 4: calibration behaviour, rubric arithmetic, and the selection outcome."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from riskos.metrics import auc, observed_vs_expected
from riskos.models import selection
from riskos.models.calibration import Calibrator, fit_all
from riskos.models.config import SelectionConfig, models_config

FIGURES = Path("reports/figures")
COMPARISON_CSV = FIGURES / "champion_challenger_metrics.csv"

needs_comparison = pytest.mark.skipif(
    not COMPARISON_CSV.exists(), reason="comparison not run; use `riskos challenger`"
)


@pytest.fixture
def fixture_miscalibrated() -> tuple[np.ndarray, np.ndarray]:
    """Well-ranked probabilities at systematically half the true level."""
    rng = np.random.default_rng(20250901)
    truth = rng.uniform(0.02, 0.30, size=40_000)
    y = rng.binomial(1, truth)
    return y, truth / 2.0


# --------------------------------------------------------------------------
# Calibration: monotone, so it changes level and never ranking.
# --------------------------------------------------------------------------


def test_platt_preserves_auc_exactly(
    fixture_miscalibrated: tuple[np.ndarray, np.ndarray],
) -> None:
    """Platt is STRICTLY monotone, so the ordering is untouched."""
    y, p = fixture_miscalibrated
    cal = Calibrator("platt")
    cal.fit(p, y)

    assert auc(y, cal.transform(p)) == pytest.approx(auc(y, p), abs=1e-9)


def test_isotonic_preserves_ranking_but_perturbs_auc_by_creating_ties(
    fixture_miscalibrated: tuple[np.ndarray, np.ndarray],
) -> None:
    """Isotonic is only WEAKLY monotone, and the distinction matters.

    Its flat segments map distinct raw scores onto a single calibrated value,
    creating ties where none existed. Ties count a half in AUC, so the metric
    moves slightly. What cannot happen is a REVERSAL: no pair correctly ordered
    before can be incorrectly ordered after.
    """
    y, p = fixture_miscalibrated
    cal = Calibrator("isotonic")
    cal.fit(p, y)
    calibrated = cal.transform(p)

    # No reversal: the map is non-decreasing on every pair.
    order = np.argsort(p)
    assert np.all(np.diff(calibrated[order]) >= -1e-12)
    # Ties are genuinely created, which is why AUC is not exactly preserved.
    assert len(np.unique(calibrated)) < len(np.unique(p))
    assert auc(y, calibrated) == pytest.approx(auc(y, p), abs=0.01)


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_calibration_corrects_a_level_it_was_fitted_on(
    method: str, fixture_miscalibrated: tuple[np.ndarray, np.ndarray]
) -> None:
    y, p = fixture_miscalibrated
    cal = Calibrator(method)  # type: ignore[arg-type]
    cal.fit(p, y)

    before = observed_vs_expected(y, p)["ratio"]
    after = observed_vs_expected(y, cal.transform(p))["ratio"]

    assert before == pytest.approx(2.0, rel=0.1)  # halved predictions
    assert abs(after - 1.0) < abs(before - 1.0) / 4


def test_calibration_cannot_fix_a_shift_it_never_saw() -> None:
    """The structural reason F-006 is not remediable by recalibration.

    A mapping fitted where the base rate is 1% cannot know the level will
    triple in a period it was never shown.
    """
    rng = np.random.default_rng(7)
    p_fit = rng.uniform(0.005, 0.02, size=30_000)
    y_fit = rng.binomial(1, p_fit)
    cal = Calibrator("platt")
    cal.fit(p_fit, y_fit)

    # A later regime with the same scores but a tripled realised default rate.
    p_new = rng.uniform(0.005, 0.02, size=30_000)
    y_new = rng.binomial(1, np.clip(p_new * 3.0, 0, 1))

    residual = observed_vs_expected(y_new, cal.transform(p_new))["ratio"]

    assert residual > 2.0, "recalibration cannot anticipate an unseen regime"


def test_uncalibrated_is_an_exact_pass_through() -> None:
    p = np.array([0.01, 0.5, 0.99])

    np.testing.assert_array_equal(Calibrator("uncalibrated").transform(p), p)


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown calibration method"):
        Calibrator("sigmoid")  # type: ignore[arg-type]


def test_fit_all_always_includes_the_uncalibrated_baseline() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.4, 5_000)
    y = rng.binomial(1, p)

    cals = fit_all(p, y, ("isotonic", "platt"))

    assert set(cals) == {"uncalibrated", "isotonic", "platt"}


# --------------------------------------------------------------------------
# Rubric arithmetic — hand-computable.
# --------------------------------------------------------------------------


def test_calibration_score_is_symmetric_in_over_and_under_prediction() -> None:
    """Both directions misstate the provision, so both must be penalised equally."""
    assert selection.score_calibration(2.0, 3.0) == pytest.approx(
        selection.score_calibration(0.5, 3.0)
    )


def test_calibration_score_endpoints_are_hand_computable() -> None:
    # Perfect calibration scores 1.0; the zero-point scores exactly 0.0.
    assert selection.score_calibration(1.0, 3.0) == pytest.approx(1.0)
    assert selection.score_calibration(3.0, 3.0) == pytest.approx(0.0)
    # 1 - ln(2)/ln(3) = 1 - 0.693147/1.098612 = 0.369070
    assert selection.score_calibration(2.0, 3.0) == pytest.approx(1 - math.log(2) / math.log(3))


def test_calibration_score_never_goes_negative() -> None:
    assert selection.score_calibration(10.0, 3.0) == 0.0
    assert selection.score_calibration(0.0, 3.0) == 0.0


def test_stability_score_maps_the_psi_bands() -> None:
    assert selection.score_stability(0.0, 0.25) == pytest.approx(1.0)
    assert selection.score_stability(0.25, 0.25) == pytest.approx(0.0)
    assert selection.score_stability(0.10, 0.25) == pytest.approx(0.6)


def test_weights_must_sum_to_one_and_carry_a_rationale() -> None:
    with pytest.raises(ValueError, match="sum to"):
        SelectionConfig(
            weights={"calibration": 0.5, "discrimination": 0.2},
            rationale={"calibration": "x", "discrimination": "y"},
            scoring={},
            tie_breaker="t",
        )


def test_a_weighted_dimension_without_a_rationale_is_rejected() -> None:
    with pytest.raises(ValueError, match="without a written rationale"):
        SelectionConfig(
            weights={"calibration": 1.0},
            rationale={},
            scoring={"calibration": {}},  # type: ignore[dict-item]
            tie_breaker="t",
        )


def test_the_committed_rubric_weights_calibration_above_discrimination() -> None:
    """Build plan §1: calibration and stability matter more than AUC."""
    weights = models_config().selection.weights

    assert weights["calibration"] > weights["discrimination"]
    assert weights["stability"] > weights["discrimination"]
    assert sum(weights.values()) == pytest.approx(1.0)


def test_tie_breaker_prefers_the_more_explainable_candidate() -> None:
    close = [
        selection.CandidateScore("gbm", {"explainability": 0.6}, {}, 0.500),
        selection.CandidateScore("scorecard", {"explainability": 1.0}, {}, 0.495),
    ]

    winner, reason = selection.select(close)

    assert winner.name == "scorecard"
    assert "tie-breaker" in reason


def test_a_clear_margin_does_not_invoke_the_tie_breaker() -> None:
    clear = [
        selection.CandidateScore("gbm", {"explainability": 0.6}, {}, 0.60),
        selection.CandidateScore("scorecard", {"explainability": 1.0}, {}, 0.40),
    ]

    winner, _ = selection.select(clear)

    assert winner.name == "gbm"


# --------------------------------------------------------------------------
# The realised comparison.
# --------------------------------------------------------------------------


@needs_comparison
def test_every_model_calibration_split_combination_was_evaluated() -> None:
    comparison = pl.read_csv(COMPARISON_CSV)

    assert set(comparison["model"]) == {"scorecard", "lightgbm"}
    assert set(comparison["calibration"]) == {"uncalibrated", "isotonic", "platt"}
    assert comparison.height == 2 * 3 * 4


@needs_comparison
def test_calibration_leaves_discrimination_untouched_on_real_data() -> None:
    stress = pl.read_csv(COMPARISON_CSV).filter(pl.col("split") == "oot_stress")

    for model in ("scorecard", "lightgbm"):
        ginis = stress.filter(pl.col("model") == model)["gini"].to_list()
        assert max(ginis) - min(ginis) < 0.005, "monotone maps must preserve ranking"


@needs_comparison
def test_both_candidates_materially_under_predict_under_stress() -> None:
    """Finding F-006, asserted rather than left in prose."""
    stress = pl.read_csv(COMPARISON_CSV).filter(pl.col("split") == "oot_stress")

    for row in stress.to_dicts():
        assert row["observed_over_expected"] > 2.0, (
            f"{row['model']}/{row['calibration']} unexpectedly well calibrated"
        )


@needs_comparison
def test_recalibration_does_not_rescue_the_stress_split() -> None:
    """The evidence that the Phase 5 macro overlay is necessary."""
    stress = pl.read_csv(COMPARISON_CSV).filter(pl.col("split") == "oot_stress")
    by_key = {(r["model"], r["calibration"]): r for r in stress.to_dicts()}

    for model in ("scorecard", "lightgbm"):
        raw = by_key[(model, "uncalibrated")]["observed_over_expected"]
        best = min(by_key[(model, m)]["observed_over_expected"] for m in ("isotonic", "platt"))
        assert raw - best < 0.25, "recalibration should not close a regime shift"


@needs_comparison
def test_score_psi_would_not_have_raised_an_alert_under_stress() -> None:
    """The most operationally important line in the selection memo.

    A monitoring programme watching only score distribution sees nothing while
    the model under-predicts defaults threefold.
    """
    stress = pl.read_csv(COMPARISON_CSV).filter(
        (pl.col("split") == "oot_stress") & (pl.col("calibration") == "uncalibrated")
    )

    for row in stress.to_dicts():
        assert row["score_psi"] < 0.10, "expected PSI to look reassuringly stable"
        assert row["observed_over_expected"] > 2.0, "while calibration is badly wrong"


@pytest.mark.skipif(not Path("models/selection_record.json").exists(), reason="selection not run")
def test_the_selection_record_names_a_winner_and_a_reason() -> None:
    record = json.loads(Path("models/selection_record.json").read_text(encoding="utf-8"))

    assert record["selected"] in {"scorecard", "lightgbm"}
    assert record["reason"]
    assert len(record["candidates"]) == 2
    # No monotone constraint may be silently dropped.
    assert record["constraints_ignored"] == []


@pytest.mark.skipif(
    not (FIGURES / "explanation_summary.json").exists(), reason="comparison not run"
)
def test_both_explanation_mechanisms_reconstruct_their_model_exactly() -> None:
    """Finding F-007: the rubric assumed SHAP was approximate. It is not."""
    summary = json.loads((FIGURES / "explanation_summary.json").read_text(encoding="utf-8"))

    assert summary["scorecard_reconstruction_error"] < 1e-8
    assert summary["shap_reconstruction_error"] < 1e-8
