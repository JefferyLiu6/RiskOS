"""Phase 6 — the rulebook, the observability lag, and the in-sample control.

These tests are about the monitoring machinery, not about the portfolio. The
properties that matter are the ones that would let a monitoring pack mislead a
reviewer: a metric that reads as a pass when it was never computed, an alert
dated earlier than the data that produced it, or a rule credited with a
detection it makes every quarter.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from riskos.monitor import alerts as alert_mod
from riskos.monitor import drift, performance
from riskos.monitor.config import Rule, monitoring_config

RNG = np.random.default_rng(20260912)


def _rule(**overrides: object) -> Rule:
    base = {
        "id": "T-01",
        "metric": "score_psi",
        "scope": "test",
        "comparator": "greater_than",
        "warn": 0.10,
        "breach": 0.25,
        "severity_warn": "medium",
        "severity_breach": "high",
        "owner": "model_owner",
        "action": "investigate",
        "basis": "judgement",
    }
    return Rule.model_validate(base | overrides)


# --- the rulebook itself -----------------------------------------------------


def test_greater_than_rule_bands_on_its_thresholds() -> None:
    rule = _rule()

    assert rule.evaluate(0.05)[0] == "ok"
    assert rule.evaluate(0.15)[0] == "warn"
    assert rule.evaluate(0.30)[0] == "breach"


def test_a_value_exactly_on_a_threshold_does_not_fire() -> None:
    # Strict inequality, pinned deliberately: the conventional PSI bands read
    # "below 0.10 stable", so 0.10 itself is the top of stable, not the bottom
    # of moderate.
    rule = _rule()

    assert rule.evaluate(0.10)[0] == "ok"
    assert rule.evaluate(0.25)[0] == "warn"


def test_outside_band_rule_is_two_sided() -> None:
    rule = _rule(
        metric="observed_over_expected",
        comparator="outside_band",
        warn=(0.80, 1.25),
        breach=(0.50, 2.00),
    )

    assert rule.evaluate(1.0)[0] == "ok"
    assert rule.evaluate(1.40)[0] == "warn"
    assert rule.evaluate(0.70)[0] == "warn"  # over-prediction fires too
    assert rule.evaluate(2.50)[0] == "breach"
    assert rule.evaluate(0.30)[0] == "breach"


def test_relative_decline_is_measured_against_the_reference() -> None:
    rule = _rule(
        metric="gini", comparator="relative_decline_from_reference", warn=0.10, breach=0.25
    )

    assert rule.evaluate(0.76, 0.80)[0] == "ok"  # 5% down
    assert rule.evaluate(0.68, 0.80)[0] == "warn"  # 15% down
    assert rule.evaluate(0.50, 0.80)[0] == "breach"  # 37.5% down
    # An improvement is never an alert.
    assert rule.evaluate(0.90, 0.80)[0] == "ok"


def test_a_missing_value_is_not_evaluated_rather_than_ok() -> None:
    """The distinction the whole lagging-indicator design rests on.

    A window whose outcomes have not matured has not passed the calibration
    test, it has not taken it. Reporting that as `ok` is the single easiest way
    to make a monitoring pack look healthier than the model is.
    """
    rule = _rule(
        metric="observed_over_expected",
        comparator="outside_band",
        warn=(0.80, 1.25),
        breach=(0.50, 2.00),
    )

    status, reason = rule.evaluate(None)

    assert status == "not_evaluated"
    assert status != "ok"
    assert "unavailable" in reason


def test_a_relative_rule_without_a_reference_is_not_evaluated() -> None:
    rule = _rule(
        metric="gini", comparator="relative_decline_from_reference", warn=0.10, breach=0.25
    )

    assert rule.evaluate(0.5, None)[0] == "not_evaluated"


def test_a_rule_that_breaches_before_it_warns_is_rejected() -> None:
    with pytest.raises(ValueError, match="breach"):
        _rule(warn=0.40, breach=0.25)


def test_a_warn_band_outside_the_breach_band_is_rejected() -> None:
    with pytest.raises(ValueError, match="inside"):
        _rule(comparator="outside_band", warn=(0.50, 2.00), breach=(0.80, 1.25))


def test_the_committed_rulebook_loads_and_every_rule_has_an_owner_and_an_action() -> None:
    cfg = monitoring_config()

    assert cfg.rules
    for rule in cfg.rules:
        assert rule.owner
        assert rule.action.strip()
        assert cfg.escalation.route(rule.severity_breach) != "none"


# --- the observability lag ---------------------------------------------------


def test_performance_metrics_are_stamped_with_when_they_became_computable() -> None:
    y = np.zeros(5000, dtype=np.int64)
    y[:200] = 1
    p = RNG.uniform(0.0, 0.1, 5000)

    window = performance.evaluate_window(
        "2008Q1", date(2008, 3, 31), y, p, outcome_window_months=12, min_defaults=50
    )

    assert window.outcome_matured_at == date(2009, 3, 31)
    assert window.outcome_matured_at > window.window_end


def test_a_window_with_too_few_defaults_reports_null_discrimination() -> None:
    y = np.zeros(5000, dtype=np.int64)
    y[:10] = 1
    p = RNG.uniform(0.0, 0.1, 5000)

    window = performance.evaluate_window(
        "1999Q1", date(1999, 3, 31), y, p, outcome_window_months=12, min_defaults=50
    )

    assert not window.evaluated
    assert window.auc is None
    assert window.gini is None
    # Calibration survives a thin window: it is a comparison of two means.
    assert window.observed_over_expected is not None
    assert "below the 50" in window.reason


def test_visible_at_hides_windows_whose_outcomes_have_not_matured() -> None:
    timeline = pl.DataFrame(
        {
            "window": ["2007Q4", "2008Q1", "2008Q2"],
            "outcome_matured_at": [date(2008, 12, 31), date(2009, 3, 31), date(2009, 6, 30)],
        }
    )

    visible = performance.visible_at(timeline, date(2009, 3, 31))

    assert visible["window"].to_list() == ["2007Q4", "2008Q1"]


def test_add_months_clamps_into_a_shorter_month() -> None:
    assert performance.add_months(date(2008, 1, 31), 1) == date(2008, 2, 29)
    assert performance.add_months(date(2007, 1, 31), 1) == date(2007, 2, 28)
    assert performance.add_months(date(2008, 12, 1), 12) == date(2009, 12, 1)


# --- alerts and the in-sample control ----------------------------------------


def _measurement(**overrides: object) -> alert_mod.Measurement:
    base = {
        "window": "2008Q1",
        "window_end": date(2008, 3, 31),
        "metric": "score_psi",
        "value": 0.40,
        "detectable_at": date(2008, 3, 31),
        "n_rows": 50_000,
    }
    return alert_mod.Measurement(**(base | overrides))  # type: ignore[arg-type]


def test_a_thin_window_is_suppressed_rather_than_dropped() -> None:
    cfg = monitoring_config()

    fired = alert_mod.evaluate_rules(cfg, [_measurement(n_rows=10)])
    breaching = [a for a in fired if a.status == "breach"]

    assert breaching, "the rule should still be evaluated"
    assert all(a.suppressed for a in breaching)
    assert all(not a.fired for a in breaching)
    assert all("below the" in a.suppression_reason for a in breaching)


def test_an_in_sample_breach_cannot_win_the_detection_race() -> None:
    """A rule firing on the data the model was fitted to has detected nothing.

    Without this exclusion the gap is dated from a 1999 firing, years before the
    model could have been deployed.
    """
    in_sample = _measurement(window="1999Q1", window_end=date(1999, 3, 31), in_sample=True)
    out_of_sample = _measurement(window="2008Q1", window_end=date(2008, 3, 31))

    fired = alert_mod.evaluate_rules(monitoring_config(), [in_sample, out_of_sample])
    first = alert_mod.first_breach(fired, ("score_psi",))

    assert first is not None
    assert first.window == "2008Q1"


def test_false_positive_rate_counts_only_in_sample_windows() -> None:
    cfg = monitoring_config()
    measurements = [
        _measurement(window="2000Q1", window_end=date(2000, 3, 31), value=0.40, in_sample=True),
        _measurement(window="2000Q2", window_end=date(2000, 6, 30), value=0.01, in_sample=True),
        _measurement(window="2008Q1", value=0.40),
    ]

    control = alert_mod.false_positives(cfg, alert_mod.evaluate_rules(cfg, measurements))
    row = control.filter(pl.col("metric") == "score_psi").row(0, named=True)

    assert row["in_sample_windows"] == 2
    assert row["fired"] == 1
    assert row["false_positive_rate"] == pytest.approx(0.5)


def test_the_detection_gap_carries_the_winning_rules_false_positive_rate() -> None:
    """A months-of-warning headline must not travel without its caveat.

    A rule that fires on most windows where nothing is wrong did not detect the
    event; it was already on when the event arrived, and the gap alone cannot
    tell the two apart.
    """
    cfg = monitoring_config()
    measurements = [
        _measurement(window=f"200{i}Q1", window_end=date(2000 + i, 3, 31), in_sample=True)
        for i in range(5)
    ] + [
        _measurement(window="2008Q1"),
        _measurement(
            metric="observed_over_expected",
            value=3.0,
            window_end=date(2008, 3, 31),
            detectable_at=date(2009, 3, 31),
            n_defaults=500,
        ),
    ]
    fired = alert_mod.evaluate_rules(cfg, measurements)
    control = alert_mod.false_positives(cfg, fired)

    gap = alert_mod.detection_gap(fired, ("score_psi",), ("observed_over_expected",), control)

    assert gap["months_of_warning"] == pytest.approx(12.0, abs=0.1)
    assert gap["leading_rule_false_positive_rate"] == pytest.approx(1.0)
    assert "not evidence of detection" in gap["caveat"]


def test_no_caveat_when_the_leading_rule_is_clean_in_sample() -> None:
    cfg = monitoring_config()
    measurements = [
        _measurement(
            window=f"200{i}Q1", window_end=date(2000 + i, 3, 31), value=0.01, in_sample=True
        )
        for i in range(5)
    ] + [
        _measurement(window="2008Q1"),
        _measurement(
            metric="observed_over_expected",
            value=3.0,
            window_end=date(2008, 3, 31),
            detectable_at=date(2009, 3, 31),
            n_defaults=500,
        ),
    ]
    fired = alert_mod.evaluate_rules(cfg, measurements)
    control = alert_mod.false_positives(cfg, fired)

    gap = alert_mod.detection_gap(fired, ("score_psi",), ("observed_over_expected",), control)

    assert gap["leading_rule_false_positive_rate"] == pytest.approx(0.0)
    assert gap["caveat"] == ""


def test_the_register_sorts_high_severity_first() -> None:
    cfg = monitoring_config()
    measurements = [
        _measurement(value=0.01),  # ok
        _measurement(value=0.40),  # breach, high
    ]

    frame = alert_mod.register(alert_mod.evaluate_rules(cfg, measurements))

    assert frame["severity"][0] == "high"


def test_an_empty_register_still_has_its_columns() -> None:
    frame = alert_mod.register([])

    assert frame.is_empty()
    assert "escalation" in frame.columns


# --- drift -------------------------------------------------------------------


def test_epsilon_dominated_windows_are_flagged() -> None:
    """A CSI carried by empty bins is an artefact of the floor, not drift.

    This is the mechanism behind finding F-016: a single early quarter cannot
    overlap a pooled multi-year loan-age distribution, so almost every bin is
    empty on one side and the reported CSI is a function of PSI_002.
    """
    timeline = pl.DataFrame(
        {
            "window": ["1999Q1", "2008Q1"],
            "worst_feature": ["loan_age", "credit_score"],
            "max_feature_csi": [12.4, 0.3],
            "worst_feature_floored_bins": [10, 0],
            "worst_feature_n_bins": [11, 11],
        }
    )

    flagged = drift.epsilon_dominated(timeline)

    assert flagged["window"].to_list() == ["1999Q1"]


def test_drift_timeline_reports_the_worst_feature_and_its_bin_counts() -> None:
    reference = pl.DataFrame({"x": RNG.normal(0.0, 1.0, 5000), "y": RNG.normal(0.0, 1.0, 5000)})
    shifted = pl.DataFrame({"x": RNG.normal(3.0, 1.0, 2000), "y": RNG.normal(0.0, 1.0, 2000)})
    ref_scores = RNG.uniform(0.0, 0.1, 5000)

    timeline, detail = drift.drift_timeline(
        ref_scores,
        reference,
        [("2008Q1", date(2008, 3, 31), RNG.uniform(0.0, 0.1, 2000), shifted)],
        ["x", "y"],
    )

    assert timeline["worst_feature"][0] == "x"
    assert timeline["max_feature_csi"][0] > timeline["n_features_csi_moderate"][0]
    assert set(detail["feature"].to_list()) == {"x", "y"}


def test_an_unshifted_window_reports_a_stable_band() -> None:
    reference = pl.DataFrame({"x": RNG.normal(0.0, 1.0, 20000)})
    same = pl.DataFrame({"x": RNG.normal(0.0, 1.0, 20000)})
    scores = RNG.uniform(0.0, 0.1, 20000)

    timeline, _ = drift.drift_timeline(
        scores, reference, [("w", date(2008, 3, 31), RNG.uniform(0.0, 0.1, 20000), same)], ["x"]
    )

    assert timeline["score_psi_band"][0] == "stable"
    assert timeline["max_feature_csi"][0] < 0.10
