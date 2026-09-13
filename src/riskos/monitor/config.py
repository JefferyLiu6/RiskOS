"""Typed config for Phase 6: monitoring windows, thresholds, and the rulebook.

The rulebook is data, not code. A threshold written into a module is a
threshold nobody outside the repository can review, and the point of a
monitoring control is that a second line can read what it fires on without
reading Python.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from riskos.config import CONF_DIR, Strict, load_yaml

Comparator = Literal[
    "greater_than",
    "outside_band",
    "relative_decline_from_reference",
    "relative_increase_from_reference",
]
Status = Literal["ok", "warn", "breach", "not_evaluated"]
Severity = Literal["none", "low", "medium", "high"]

# Rules whose comparator needs a development-sample value to compare against.
RELATIVE = ("relative_decline_from_reference", "relative_increase_from_reference")


class Window(Strict):
    frequency: Literal["quarterly", "monthly"]
    min_rows_for_alert: int
    min_defaults_for_performance: int


class Reference(Strict):
    split: str
    n_bins: int


class Observability(Strict):
    outcome_window_months: int
    performance_metrics_lag_months: int


class Rule(Strict):
    """One monitoring rule, with its thresholds, owner and required action."""

    id: str
    metric: str
    scope: str
    comparator: Comparator
    warn: float | tuple[float, float]
    breach: float | tuple[float, float]
    severity_warn: Severity
    severity_breach: Severity
    owner: str
    action: str
    basis: str

    @model_validator(mode="after")
    def _thresholds_match_comparator(self) -> Rule:
        banded = self.comparator == "outside_band"
        for name in ("warn", "breach"):
            value = getattr(self, name)
            if banded and not isinstance(value, tuple):
                raise ValueError(f"rule {self.id}: {name} must be a [low, high] band")
            if not banded and isinstance(value, tuple):
                raise ValueError(f"rule {self.id}: {name} must be a scalar threshold")
        if banded:
            assert isinstance(self.warn, tuple) and isinstance(self.breach, tuple)
            if not (self.breach[0] <= self.warn[0] and self.warn[1] <= self.breach[1]):
                raise ValueError(
                    f"rule {self.id}: the warn band must sit inside the breach band, "
                    "otherwise a value can breach without ever warning"
                )
        elif float(self.warn) > float(self.breach):  # type: ignore[arg-type]
            raise ValueError(
                f"rule {self.id}: warn threshold {self.warn} exceeds breach {self.breach}, "
                "so the rule would breach before it warns"
            )
        return self

    @property
    def needs_reference(self) -> bool:
        return self.comparator in RELATIVE

    def evaluate(self, value: float | None, reference: float | None = None) -> tuple[Status, str]:
        """Classify one measurement. Returns the status and a plain-English reason.

        A ``None`` value is ``not_evaluated``, never ``ok``. The distinction
        matters: a window whose outcomes have not matured yet has not passed the
        calibration test, it has not taken it, and reporting that as a pass is
        the single easiest way to make a monitoring pack look healthier than the
        model is.
        """
        if value is None:
            return "not_evaluated", "metric unavailable for this window"
        if self.needs_reference and reference in (None, 0.0):
            return "not_evaluated", "no development-sample reference to compare against"

        if self.comparator == "greater_than":
            assert not isinstance(self.warn, tuple) and not isinstance(self.breach, tuple)
            if value > self.breach:
                return "breach", f"{value:.4f} above breach threshold {self.breach}"
            if value > self.warn:
                return "warn", f"{value:.4f} above warn threshold {self.warn}"
            return "ok", f"{value:.4f} within {self.warn}"

        if self.comparator == "outside_band":
            assert isinstance(self.warn, tuple) and isinstance(self.breach, tuple)
            if not self.breach[0] <= value <= self.breach[1]:
                return "breach", f"{value:.4f} outside breach band {list(self.breach)}"
            if not self.warn[0] <= value <= self.warn[1]:
                return "warn", f"{value:.4f} outside warn band {list(self.warn)}"
            return "ok", f"{value:.4f} inside {list(self.warn)}"

        assert reference is not None
        assert not isinstance(self.warn, tuple) and not isinstance(self.breach, tuple)
        if self.comparator == "relative_decline_from_reference":
            change = (reference - value) / abs(reference)
            wording = f"{change:.1%} below the development-sample {reference:.4f}"
        else:
            change = (value - reference) / abs(reference)
            wording = f"{change:.1%} above the development-sample {reference:.4f}"
        if change > self.breach:
            return "breach", f"{wording} (breach at {self.breach:.0%})"
        if change > self.warn:
            return "warn", f"{wording} (warn at {self.warn:.0%})"
        return "ok", wording

    def severity_for(self, status: Status) -> Severity:
        return {"warn": self.severity_warn, "breach": self.severity_breach}.get(status, "none")


class Escalation(Strict):
    low: str
    medium: str
    high: str
    breach_response_days: int

    def route(self, severity: Severity) -> str:
        return {"low": self.low, "medium": self.medium, "high": self.high}.get(severity, "none")


class MonitoringConfig(Strict):
    window: Window
    reference: Reference
    observability: Observability
    rules: tuple[Rule, ...]
    escalation: Escalation

    @model_validator(mode="after")
    def _rule_ids_are_unique(self) -> MonitoringConfig:
        ids = [r.id for r in self.rules]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate monitoring rule ids: {sorted(ids)}")
        return self

    def rules_for(self, metric: str) -> list[Rule]:
        return [r for r in self.rules if r.metric == metric]


@lru_cache(maxsize=1)
def monitoring_config(conf_dir: Path = CONF_DIR) -> MonitoringConfig:
    return MonitoringConfig.model_validate(load_yaml(conf_dir / "monitoring.yaml"))
