"""Phase 4 orchestration: fit the challenger, calibrate both, compare, select."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import polars as pl

from riskos.features import binning
from riskos.log import get_logger
from riskos.models import calibration, evaluate, explain, selection
from riskos.models.config import models_config
from riskos.models.lgbm import Challenger, to_frame
from riskos.models.scorecard import Scorecard

log = get_logger(__name__)

PANEL_GLOB = "data/panel/panel/*/*.parquet"
ARTIFACTS = Path("reports/figures")
MODELS = Path("models")
EXPERIMENT = "riskos-pd-12m"
TRACKING_URI = "sqlite:///mlflow.db"
LATENCY_ROWS = 50_000


def _latency_us(predict: Any, frame: Any) -> float:
    """Microseconds per row, measured on a fixed-size batch."""
    sample = frame.head(LATENCY_ROWS)
    start = time.perf_counter()
    predict(sample)
    return (time.perf_counter() - start) * 1e6 / len(sample)


def _max_oot_psi(rows: list[evaluate.SplitMetrics]) -> float:
    values = [r.score_psi for r in rows if r.split.startswith("oot") and r.score_psi is not None]
    return max(values) if values else 0.0


def _evaluate_everything(
    panel: pl.DataFrame,
    predict: Any,
    prepare: Any,
    reference: np.ndarray,
    calibrators: dict[str, calibration.Calibrator],
    model_name: str,
) -> tuple[list[dict[str, Any]], dict[str, pl.DataFrame]]:
    """Metrics for every (split x calibration method) combination.

    The PSI reference is calibrated with the SAME mapping as the window being
    measured. Comparing isotonic-calibrated scores against an uncalibrated
    reference reports the reshaping done by the calibrator itself as population
    drift — a large number that says nothing about the population.
    """
    records: list[dict[str, Any]] = []
    curves: dict[str, pl.DataFrame] = {}
    references = {m: cal.transform(reference) for m, cal in calibrators.items()}
    for split in evaluate.SPLITS:
        frame = panel.filter(pl.col("split") == split)
        x = prepare(frame)
        y = frame["default_12m"].to_numpy()
        raw_p = predict(x)
        for method, cal in calibrators.items():
            p = cal.transform(raw_p)
            metrics = evaluate.evaluate_split(
                split, y, p, None if split == "train" else references[method]
            )
            records.append({"model": model_name, "calibration": method, **asdict(metrics)})
            if method == "uncalibrated":
                curves[f"{model_name}_{split}"] = evaluate.reliability_curve_for(y, p)
    return records, curves


def run() -> pl.DataFrame:
    """Fit the challenger, calibrate both candidates, score the rubric, select."""
    cfg = models_config()
    panel = pl.read_parquet(PANEL_GLOB)
    train = panel.filter(pl.col("split") == "train")
    valid = panel.filter(pl.col("split") == "validation_in_time")

    # Same candidate pool for both families (conf/models.yaml candidate_pool).
    process, names = binning.fit(train)
    ivs = binning.information_values(process, names)
    woe_train, y_train = binning.transform(process, train, binning.selected_names(ivs))
    kept, _ = binning.prune_correlated(woe_train, ivs)

    card = Scorecard(kept)
    card_fit = card.fit(binning.transform(process, train, kept)[0], y_train)
    card_features = card.features

    challenger = Challenger(names)
    ch_fit = challenger.fit(
        to_frame(train, names),
        y_train,
        to_frame(valid, names),
        valid["default_12m"].to_numpy(),
    )

    def card_prepare(frame: pl.DataFrame) -> Any:
        return binning.transform(process, frame, card_features)[0]

    def gbm_prepare(frame: pl.DataFrame) -> Any:
        # The mapping fixed at fit time, never re-derived from the batch.
        return challenger.prepare(frame)

    y_valid = valid["default_12m"].to_numpy()
    card_cals = calibration.fit_all(
        card.predict_proba(card_prepare(valid)), y_valid, cfg.calibration.methods
    )
    gbm_cals = calibration.fit_all(
        challenger.predict_proba(gbm_prepare(valid)), y_valid, cfg.calibration.methods
    )

    card_ref = card.predict_proba(card_prepare(train))
    gbm_ref = challenger.predict_proba(gbm_prepare(train))
    card_rows, card_curves = _evaluate_everything(
        panel, card.predict_proba, card_prepare, card_ref, card_cals, "scorecard"
    )
    gbm_rows, gbm_curves = _evaluate_everything(
        panel, challenger.predict_proba, gbm_prepare, gbm_ref, gbm_cals, "lightgbm"
    )
    comparison = pl.DataFrame(card_rows + gbm_rows)

    scores = _score_rubric(comparison, card, card_prepare, challenger, gbm_prepare, panel)
    winner, reason = selection.select(scores)
    _write_artifacts(
        comparison,
        scores,
        winner,
        reason,
        card,
        challenger,
        ch_fit,
        card_prepare,
        gbm_prepare,
        panel,
        {**card_curves, **gbm_curves},
    )
    _log_mlflow(comparison, scores, winner, card_fit, ch_fit)
    return comparison


def _score_rubric(
    comparison: pl.DataFrame,
    card: Scorecard,
    card_prepare: Any,
    challenger: Challenger,
    gbm_prepare: Any,
    panel: pl.DataFrame,
) -> list[selection.CandidateScore]:
    """Apply the pre-committed rubric to the uncalibrated candidates."""
    train = panel.filter(pl.col("split") == "train")
    latencies = {
        "scorecard": _latency_us(card.predict_proba, card_prepare(train)),
        "lightgbm": _latency_us(challenger.predict_proba, gbm_prepare(train)),
    }
    scores = []
    for name, exact in (("scorecard", True), ("lightgbm", False)):
        rows = comparison.filter(
            (pl.col("model") == name) & (pl.col("calibration") == "uncalibrated")
        )
        stress = rows.filter(pl.col("split") == "oot_stress").to_dicts()[0]
        psi_values = [
            r["score_psi"]
            for r in rows.to_dicts()
            if str(r["split"]).startswith("oot") and r["score_psi"] is not None
        ]
        scores.append(
            selection.score_candidate(
                name,
                oot_stress_gini=float(stress["gini"]),
                oot_stress_observed_over_expected=float(stress["observed_over_expected"]),
                max_oot_score_psi=float(max(psi_values)) if psi_values else 0.0,
                explanation_is_exact=exact,
                microseconds_per_row=latencies[name],
            )
        )
    return scores


def _write_artifacts(
    comparison: pl.DataFrame,
    scores: list[selection.CandidateScore],
    winner: selection.CandidateScore,
    reason: str,
    card: Scorecard,
    challenger: Challenger,
    ch_fit: Any,
    card_prepare: Any,
    gbm_prepare: Any,
    panel: pl.DataFrame,
    curves: dict[str, pl.DataFrame],
) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    comparison.write_csv(ARTIFACTS / "champion_challenger_metrics.csv")
    selection.scorecard_table(scores).write_csv(ARTIFACTS / "selection_rubric.csv")
    challenger.grid_table().write_csv(ARTIFACTS / "lgbm_grid.csv")
    for name, curve in curves.items():
        curve.write_csv(ARTIFACTS / f"reliability_cc_{name}.csv")

    # Explanation comparison on ten OOT-stress borrowers (build plan §7.7).
    sample = panel.filter(pl.col("split") == "oot_stress").head(10)
    table, summary = explain.compare(card, card_prepare(sample), challenger, gbm_prepare(sample))
    table.write_csv(ARTIFACTS / "explanation_comparison.csv")
    (ARTIFACTS / "explanation_summary.json").write_text(
        json.dumps(asdict(summary), indent=2), encoding="utf-8"
    )
    # The selected model has to be loadable, not merely named. Defect F-018: the
    # rubric chose LightGBM and nothing on disk could score it, so every
    # downstream figure came from the runner-up.
    challenger.save_bundle(
        MODELS / "challenger_bundle",
        shap_background=gbm_prepare(panel.filter(pl.col("split") == "train").head(1000)),
    )
    (MODELS / "selection_record.json").write_text(
        json.dumps(
            {
                "selected": winner.name,
                "reason": reason,
                "candidates": selection.as_records(scores),
                "challenger_best_params": ch_fit.best_params,
                "constraints_ignored": ch_fit.constraints_ignored,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("wrote_selection_artifacts", path=str(ARTIFACTS), selected=winner.name)


def _log_mlflow(
    comparison: pl.DataFrame,
    scores: list[selection.CandidateScore],
    winner: selection.CandidateScore,
    card_fit: Any,
    ch_fit: Any,
) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    cfg = models_config()
    with mlflow.start_run(run_name="champion-challenger-selection"):
        mlflow.log_params(
            {
                "selected": winner.name,
                "scorecard_features": len(card_fit.features),
                "lgbm_features": len(ch_fit.features),
                **{f"lgbm_{k}": v for k, v in ch_fit.best_params.items()},
                **{f"weight_{k}": v for k, v in cfg.selection.weights.items()},
            }
        )
        for candidate in scores:
            for dim, value in candidate.dimension_scores.items():
                mlflow.log_metric(f"{candidate.name}__rubric_{dim}", value)
            mlflow.log_metric(f"{candidate.name}__rubric_total", candidate.weighted_total)
        for row in comparison.to_dicts():
            tag = f"{row['model']}__{row['calibration']}__{row['split']}"
            for metric in ("auc", "gini", "brier", "reliability", "observed_over_expected"):
                mlflow.log_metric(f"{tag}__{metric}", float(row[metric]))
        mlflow.log_artifacts(str(ARTIFACTS), artifact_path="phase4")
        log.info("logged_mlflow", experiment=EXPERIMENT, selected=winner.name)
