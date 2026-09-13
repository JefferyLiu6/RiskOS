"""Phase 6 orchestration: score the book forward and run the rulebook over it.

Each PD model is fitted on 1999-2006 and then pointed at every quarter through
2019 without refitting, which is what a deployed model experiences. The output
is three artefacts a monitoring pack is made of: a drift timeline, a performance
timeline stamped with when each metric became computable, and an alert register
naming the rule, the owner and the required action for every firing.

Both PD models are run, not just the one that happens to be deployed. The
scorecard and the LightGBM challenger are held to the identical rulebook on the
identical windows, so the comparison between them is a comparison of models
rather than of monitoring setups. That also closes F-018 properly: the model the
Phase 4 rubric selected is monitored rather than merely named.

The headline number is the detection gap — how many months the leading
indicators breached before the lagging ones could. That is the measurable
justification for drift monitoring, and it is reported whichever way it comes
out.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from riskos.features import binning
from riskos.log import get_logger
from riskos.models.lgbm import Challenger
from riskos.models.persistence import BundleError, load_manifest
from riskos.models.scorecard import Scorecard
from riskos.monitor import alerts as alert_mod
from riskos.monitor import drift, performance
from riskos.monitor.config import MonitoringConfig, monitoring_config

log = get_logger(__name__)

PANEL_GLOB = "data/panel/panel/*/*.parquet"
MODELS_DIR = Path("models")
ARTIFACTS = Path("reports/figures")
GOVERNANCE = Path("governance")

# Bundle directory -> how to load it. Monitoring is not the place to discover
# how a model family scores, so each is named explicitly and a bundle that is
# absent is skipped with a warning rather than failing the run.
PD_BUNDLES = ("scorecard_bundle", "challenger_bundle")

# Which rules are leading and which are lagging. Derived from the metric, not
# hardcoded per rule id, so a rule added to conf/monitoring.yaml lands on the
# correct side of the gap without touching this module.
LEADING_METRICS = ("score_psi", "max_feature_csi", "n_features_csi_significant")
LAGGING_METRICS = ("observed_over_expected", "gini", "brier_reliability")


def quarter_label(day: date) -> str:
    return f"{day.year}Q{(day.month - 1) // 3 + 1}"


def quarter_end(day: date) -> date:
    """Last calendar day of the quarter containing ``day``."""
    first_of_quarter = date(day.year, (day.month - 1) // 3 * 3 + 1, 1)
    return performance.add_months(first_of_quarter, 3) - timedelta(days=1)


def _windows(panel: pl.DataFrame) -> list[tuple[str, date, pl.DataFrame]]:
    """Split the panel into monitoring windows, oldest first."""
    labelled = panel.with_columns(
        pl.col("observation_date").dt.truncate("1q").alias("_quarter_start")
    )
    out: list[tuple[str, date, pl.DataFrame]] = []
    for (start,), frame in sorted(labelled.group_by("_quarter_start"), key=lambda kv: kv[0]):
        assert isinstance(start, date)
        out.append((quarter_label(start), quarter_end(start), frame.drop("_quarter_start")))
    return out


@dataclass(frozen=True)
class MonitoredModel:
    """A loaded model reduced to what monitoring needs: an id and a scorer."""

    model_id: str
    version: str
    family: str
    label: str
    features: list[str]
    score: Callable[[pl.DataFrame], np.ndarray]


def _load_scorecard(directory: Path) -> MonitoredModel:
    manifest = load_manifest(directory)
    card, process, _ = Scorecard.from_bundle(directory)

    def score(frame: pl.DataFrame) -> np.ndarray:
        woe, _ = binning.transform(process, frame, card.features)
        return card.predict_proba(woe)

    return MonitoredModel(
        model_id=manifest.model_id,
        version=manifest.version,
        family=manifest.model_family,
        label="scorecard",
        features=list(card.features),
        score=score,
    )


def _load_challenger(directory: Path) -> MonitoredModel:
    manifest = load_manifest(directory)
    challenger, _, _ = Challenger.from_bundle(directory)

    def score(frame: pl.DataFrame) -> np.ndarray:
        return challenger.predict_proba(challenger.prepare(frame))

    return MonitoredModel(
        model_id=manifest.model_id,
        version=manifest.version,
        family=manifest.model_family,
        label="challenger",
        features=list(challenger.features),
        score=score,
    )


LOADERS: dict[str, Callable[[Path], MonitoredModel]] = {
    "scorecard_bundle": _load_scorecard,
    "challenger_bundle": _load_challenger,
}


def load_models(models_dir: Path = MODELS_DIR) -> list[MonitoredModel]:
    """Every PD bundle that is present and loadable."""
    out: list[MonitoredModel] = []
    for name in PD_BUNDLES:
        directory = models_dir / name
        try:
            out.append(LOADERS[name](directory))
        except BundleError as exc:
            log.warning("monitored_model_unavailable", bundle=name, error=str(exc))
    if not out:
        raise BundleError(
            "no PD model bundles found; run `riskos train` and `riskos challenger` first"
        )
    return out


def _measurements(
    drift_timeline: pl.DataFrame,
    performance_timeline: pl.DataFrame,
    reference: dict[str, float],
    development_end: date,
) -> list[alert_mod.Measurement]:
    """Flatten both timelines into rule inputs, carrying availability dates."""
    out: list[alert_mod.Measurement] = []
    for row in drift_timeline.to_dicts():
        for metric in LEADING_METRICS:
            out.append(
                alert_mod.Measurement(
                    window=row["window"],
                    window_end=row["window_end"],
                    metric=metric,
                    value=float(row[metric]),
                    # No outcome needed, so the metric is knowable the moment
                    # the window is scored.
                    detectable_at=row["window_end"],
                    n_rows=int(row["n_rows"]),
                    detail=str(row["worst_feature"]),
                    in_sample=row["window_end"] <= development_end,
                )
            )
    for row in performance_timeline.to_dicts():
        for metric in LAGGING_METRICS:
            value = row[metric]
            out.append(
                alert_mod.Measurement(
                    window=row["window"],
                    window_end=row["window_end"],
                    metric=metric,
                    value=None if value is None else float(value),
                    detectable_at=row["outcome_matured_at"],
                    n_rows=int(row["n_rows"]),
                    n_defaults=int(row["n_defaults"]),
                    reference=reference.get(metric),
                    in_sample=row["window_end"] <= development_end,
                )
            )
    return out


def _reference_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Development-sample values the relative rules are measured against."""
    from riskos.metrics import brier_decomposition, gini

    return {
        "gini": gini(y, p),
        "brier_reliability": brier_decomposition(y, p).reliability,
    }


def _monitor_one(
    model: MonitoredModel,
    panel: pl.DataFrame,
    windows: list[tuple[str, date, pl.DataFrame]],
    cfg: MonitoringConfig,
    as_at: date,
) -> dict[str, Any]:
    """Drift, performance and alerting for one model, on shared windows."""
    reference_frame = panel.filter(pl.col("split") == cfg.reference.split)
    reference_scores = model.score(reference_frame)
    reference = _reference_metrics(
        reference_frame["default_12m"].to_numpy().astype(np.int64), reference_scores
    )
    log.info(
        "monitoring_reference",
        model_id=model.model_id,
        split=cfg.reference.split,
        n=reference_frame.height,
        **{k: round(v, 4) for k, v in reference.items()},
    )

    drift_inputs = []
    performance_inputs = []
    for label, window_end, frame in windows:
        scores = model.score(frame)
        drift_inputs.append((label, window_end, scores, frame))
        performance_inputs.append(
            (label, window_end, frame["default_12m"].to_numpy().astype(np.int64), scores)
        )

    timeline, csi_detail = drift.drift_timeline(
        reference_scores,
        reference_frame,
        drift_inputs,
        # Each model's CSI covers its OWN inputs. Monitoring a model against a
        # feature it does not use would report drift it cannot be affected by.
        model.features,
        n_bins=cfg.reference.n_bins,
    )
    perf = performance.performance_timeline(
        performance_inputs,
        outcome_window_months=cfg.observability.outcome_window_months,
        min_defaults=cfg.window.min_defaults_for_performance,
    )

    last_development_observation = reference_frame["observation_date"].max()
    assert isinstance(last_development_observation, date)
    development_end = quarter_end(last_development_observation)
    fired = alert_mod.evaluate_rules(cfg, _measurements(timeline, perf, reference, development_end))
    control = alert_mod.false_positives(cfg, fired)
    gap = alert_mod.detection_gap(fired, LEADING_METRICS, LAGGING_METRICS, control)

    prefix = f"monitoring_{model.label}"
    timeline.write_csv(ARTIFACTS / f"{prefix}_drift_timeline.csv")
    csi_detail.write_csv(ARTIFACTS / f"{prefix}_csi_detail.csv")
    perf.write_csv(ARTIFACTS / f"{prefix}_performance_timeline.csv")
    drift.band_summary(timeline).write_csv(ARTIFACTS / f"{prefix}_psi_bands.csv")
    drift.epsilon_dominated(timeline).write_csv(ARTIFACTS / f"{prefix}_csi_epsilon_dominated.csv")
    control.write_csv(ARTIFACTS / f"{prefix}_rule_false_positives.csv")
    performance.visible_at(perf, as_at).write_csv(ARTIFACTS / f"{prefix}_performance_visible.csv")

    breaches = [a for a in fired if a.status == "breach" and not a.suppressed]
    register = alert_mod.register(fired).with_columns(pl.lit(model.model_id).alias("model_id"))
    return {
        "model_id": model.model_id,
        "model_version": model.version,
        "model_family": model.family,
        "development_sample_ends": development_end.isoformat(),
        "windows": timeline.height,
        "first_window": timeline["window"][0] if timeline.height else None,
        "last_window": timeline["window"][-1] if timeline.height else None,
        "max_score_psi": _as_float(timeline["score_psi"].max()),
        "max_stress_score_psi": _stress_psi(timeline),
        "windows_psi_significant": int((timeline["score_psi"] > 0.25).sum())
        if timeline.height
        else 0,
        "alerts_fired": sum(a.fired for a in fired),
        "breaches": len(breaches),
        "breaches_out_of_sample": sum(not a.in_sample for a in breaches),
        "suppressed": sum(a.suppressed for a in fired),
        "rule_false_positive_rates": {
            r["rule_id"]: r["false_positive_rate"] for r in control.to_dicts()
        },
        "detection_gap": gap,
        "_register": register,
    }


def _as_float(value: object) -> float | None:
    """Polars aggregations are typed as any scalar; monitoring wants a float."""
    return None if value is None else float(value)  # type: ignore[arg-type]


def _stress_psi(timeline: pl.DataFrame) -> float | None:
    """Worst score PSI over the stress windows, the number F-015 turns on."""
    if timeline.is_empty():
        return None
    stress = timeline.filter(pl.col("window").str.slice(0, 4).cast(pl.Int64).is_between(2007, 2009))
    return _as_float(stress["score_psi"].max()) if stress.height else None


def run(cfg: MonitoringConfig | None = None, as_at: date | None = None) -> dict[str, Any]:
    """Score every quarter with every PD model and apply the committed rulebook."""
    cfg = cfg or monitoring_config()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    GOVERNANCE.mkdir(parents=True, exist_ok=True)
    as_at = as_at or date.today()

    panel = pl.read_parquet(PANEL_GLOB)
    windows = _windows(panel)
    models = load_models()

    per_model = [_monitor_one(m, panel, windows, cfg, as_at) for m in models]

    # One register across all models, so a reviewer reads a single list of what
    # fired rather than reconciling one file per model.
    pl.concat([r.pop("_register") for r in per_model]).write_csv(GOVERNANCE / "alert_register.csv")

    report: dict[str, Any] = {
        "reference_split": cfg.reference.split,
        "rules_evaluated": len(cfg.rules),
        "windows": per_model[0]["windows"],
        "models": per_model,
    }
    (ARTIFACTS / "monitoring_summary.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    for entry in per_model:
        log.info(
            "monitoring_complete",
            **{k: v for k, v in entry.items() if k not in ("detection_gap",)},
        )
    return report
