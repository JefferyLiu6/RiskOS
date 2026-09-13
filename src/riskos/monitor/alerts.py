"""Alert records: what fired, when it became knowable, and who has to act.

An alert here is a record, not a notification. It carries the measured value,
the threshold it crossed, the rule that owns it, the escalation route, and two
dates that are deliberately kept apart:

``window_end``
    when the exposure being measured existed.
``detectable_at``
    the earliest date the alert could have been raised.

For a drift rule those dates are the same. For a performance rule they differ by
the outcome window, because the metric cannot be computed until the labels
mature. Collapsing the two would make a monitoring pack claim, in hindsight,
that it caught something twelve months before it could possibly have known.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import polars as pl

from riskos.log import get_logger
from riskos.monitor.config import MonitoringConfig, Severity, Status

log = get_logger(__name__)

SEVERITY_ORDER: dict[Severity, int] = {"none": 0, "low": 1, "medium": 2, "high": 3}

# Above this in-sample firing rate, a rule is reported as unfit to claim a
# detection rather than as an early-warning success. Registered as MON_006.
HIGH_FALSE_POSITIVE_RATE = 0.25


@dataclass(frozen=True)
class Measurement:
    """One metric on one monitoring window, with its availability dates."""

    window: str
    window_end: date
    metric: str
    value: float | None
    detectable_at: date
    n_rows: int
    n_defaults: int | None = None
    reference: float | None = None
    detail: str = ""
    # True when the window sits inside the development sample. Such a window is
    # not evidence about a deployed model; it is a control. See `false_positives`.
    in_sample: bool = False


@dataclass(frozen=True)
class Alert:
    """A rule firing on a window. Suppressed alerts are recorded, not dropped."""

    window: str
    window_end: date
    detectable_at: date
    rule_id: str
    metric: str
    value: float | None
    reference: float | None
    status: Status
    severity: Severity
    reason: str
    owner: str
    escalation: str
    action: str
    suppressed: bool
    suppression_reason: str
    n_rows: int
    in_sample: bool

    @property
    def fired(self) -> bool:
        return self.status in ("warn", "breach") and not self.suppressed


def evaluate_rules(cfg: MonitoringConfig, measurements: list[Measurement]) -> list[Alert]:
    """Apply every rule to every measurement of its metric.

    A thin window is measured and evaluated but its alert is marked suppressed
    rather than discarded. Deleting it would hide the fact that the control was
    silent on that window for a reason, and "no alert" and "alert suppressed for
    insufficient data" are different statements to a reviewer.
    """
    alerts: list[Alert] = []
    for m in measurements:
        for rule in cfg.rules_for(m.metric):
            status, reason = rule.evaluate(m.value, m.reference)
            thin = m.n_rows < cfg.window.min_rows_for_alert
            severity = rule.severity_for(status)
            alerts.append(
                Alert(
                    window=m.window,
                    window_end=m.window_end,
                    detectable_at=m.detectable_at,
                    rule_id=rule.id,
                    metric=m.metric,
                    value=m.value,
                    reference=m.reference,
                    status=status,
                    severity=severity,
                    reason=reason,
                    owner=rule.owner,
                    escalation=cfg.escalation.route(severity),
                    action=rule.action.strip(),
                    suppressed=thin and status in ("warn", "breach"),
                    suppression_reason=(
                        f"window has {m.n_rows:,} rows, below the "
                        f"{cfg.window.min_rows_for_alert:,} needed to alert"
                        if thin and status in ("warn", "breach")
                        else ""
                    ),
                    n_rows=m.n_rows,
                    in_sample=m.in_sample,
                )
            )
    fired = [a for a in alerts if a.fired]
    log.info(
        "alerts_evaluated",
        measurements=len(measurements),
        evaluated=len(alerts),
        fired=len(fired),
        breaches=sum(a.status == "breach" for a in fired),
    )
    return alerts


# Explicit for the same reason as the performance schema: `value` and
# `reference` are null on the windows whose outcomes have not matured, and those
# are the first rows of the register.
SCHEMA = pl.Schema(
    {
        "window": pl.Utf8,
        "window_end": pl.Date,
        "detectable_at": pl.Date,
        "rule_id": pl.Utf8,
        "metric": pl.Utf8,
        "value": pl.Float64,
        "reference": pl.Float64,
        "status": pl.Utf8,
        "severity": pl.Utf8,
        "reason": pl.Utf8,
        "owner": pl.Utf8,
        "escalation": pl.Utf8,
        "action": pl.Utf8,
        "suppressed": pl.Boolean,
        "suppression_reason": pl.Utf8,
        "n_rows": pl.Int64,
        "in_sample": pl.Boolean,
    }
)


def register(alerts: list[Alert]) -> pl.DataFrame:
    """The alert register, ordered as a reviewer would read it: worst first."""
    if not alerts:
        return pl.DataFrame(schema=SCHEMA)
    frame = pl.DataFrame([asdict(a) for a in alerts], schema=SCHEMA)
    return (
        frame.with_columns(
            pl.col("severity").replace_strict(SEVERITY_ORDER, return_dtype=pl.Int64).alias("_rank")
        )
        .sort(["_rank", "window_end"], descending=[True, False])
        .drop("_rank")
    )


def first_breach(alerts: list[Alert], metrics: tuple[str, ...]) -> Alert | None:
    """Earliest genuine breach on any of ``metrics``, by when it became knowable.

    Ordered on ``detectable_at`` rather than ``window_end`` because the question
    this answers is when a monitoring function would have escalated, not which
    quarter was worst in hindsight.

    In-sample windows are excluded. A rule firing on the data the model was
    fitted on is measuring the rule, not the model, and letting one win the race
    would date the detection years before the model was even deployed.
    """
    candidates = [
        a
        for a in alerts
        if a.metric in metrics and a.status == "breach" and not a.suppressed and not a.in_sample
    ]
    return min(candidates, key=lambda a: (a.detectable_at, a.window_end)) if candidates else None


def detection_gap(
    alerts: list[Alert],
    leading: tuple[str, ...],
    lagging: tuple[str, ...],
    control: pl.DataFrame | None = None,
) -> dict[str, Any]:
    """How much earlier the leading indicators breached than the lagging ones.

    This is the number that decides whether drift monitoring earns its place on
    this model. If the score distribution breaks months before the measured
    calibration does, watching inputs bought real time. If it does not, drift
    monitoring on this model is ceremony, and saying so is more useful than
    plotting it anyway.

    ``control`` is the per-rule in-sample false-positive table. When supplied,
    the winning rule's false-positive rate is reported alongside the gap and, if
    that rate is high, so is a warning. A months-of-warning figure produced by a
    rule that also fires on most windows where nothing is wrong is not early
    detection; it is a rule that is always on, being credited for being on when
    something finally happened. The two are indistinguishable from the gap alone
    and completely distinguishable with the control rate next to it.
    """
    lead, lag = first_breach(alerts, leading), first_breach(alerts, lagging)
    gap = None
    if lead and lag:
        gap = round((lag.detectable_at - lead.detectable_at).days / 30.4375, 1)

    rates: dict[str, float | None] = {}
    if control is not None and not control.is_empty():
        rates = dict(
            zip(control["rule_id"].to_list(), control["false_positive_rate"].to_list(), strict=True)
        )
    lead_fp = rates.get(lead.rule_id) if lead else None
    caveat = ""
    if lead is not None and lead_fp is not None and lead_fp > HIGH_FALSE_POSITIVE_RATE:
        caveat = (
            f"rule {lead.rule_id} also fires on {lead_fp:.0%} of development-sample "
            "windows, where the model is by construction working. Its early firing is "
            "not evidence of detection; treat the warning it bought as zero until the "
            "rule is re-specified."
        )
    return {
        "leading_metric": lead.metric if lead else None,
        "leading_rule": lead.rule_id if lead else None,
        "leading_window": lead.window if lead else None,
        "leading_detectable_at": lead.detectable_at.isoformat() if lead else None,
        "leading_rule_false_positive_rate": lead_fp,
        "lagging_metric": lag.metric if lag else None,
        "lagging_rule": lag.rule_id if lag else None,
        "lagging_window": lag.window if lag else None,
        "lagging_detectable_at": lag.detectable_at.isoformat() if lag else None,
        "months_of_warning": gap,
        "caveat": caveat,
    }


def false_positives(cfg: MonitoringConfig, alerts: list[Alert]) -> pl.DataFrame:
    """Per-rule firing rate on the development sample, which is the control.

    Every in-sample window is a window on which the model is, by construction,
    working: it is the data the coefficients were fitted to. A rule that fires
    there is producing a false positive, and a rule that fires on most of them
    is not a control at all — it is a source of alert fatigue, which is how real
    monitoring functions go blind. Measuring the rate is the only way to say
    that with a number instead of an opinion.
    """
    rows = []
    for rule in cfg.rules:
        sample = [a for a in alerts if a.rule_id == rule.id and a.in_sample]
        evaluated = [a for a in sample if a.status != "not_evaluated"]
        fired = [a for a in evaluated if a.fired]
        rows.append(
            {
                "rule_id": rule.id,
                "metric": rule.metric,
                "in_sample_windows": len(sample),
                "evaluated": len(evaluated),
                "fired": len(fired),
                "breached": sum(a.status == "breach" for a in fired),
                "false_positive_rate": (len(fired) / len(evaluated)) if evaluated else None,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "rule_id": pl.Utf8,
            "metric": pl.Utf8,
            "in_sample_windows": pl.Int64,
            "evaluated": pl.Int64,
            "fired": pl.Int64,
            "breached": pl.Int64,
            "false_positive_rate": pl.Float64,
        },
    )
