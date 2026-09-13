"""Phase 3 orchestration: bin, screen, fit, evaluate, and log to MLflow."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlflow
import polars as pl

from riskos.features import binning
from riskos.log import get_logger
from riskos.models import evaluate, plots
from riskos.models.scorecard import Scorecard
from riskos.panel.config import feature_config

log = get_logger(__name__)

PANEL_GLOB = "data/panel/panel/*/*.parquet"
ARTIFACTS = Path("reports/figures")
MODELS = Path("models")
EXPERIMENT = "riskos-pd-12m"
TRACKING_URI = "sqlite:///mlflow.db"


def _feature_selection(
    train: pl.DataFrame,
) -> tuple[Any, list[str], list[binning.FeatureIV], list[dict[str, object]]]:
    """Bin on training only, screen by IV, then prune collinear survivors."""
    process, names = binning.fit(train)
    ivs = binning.information_values(process, names)
    passed = binning.selected_names(ivs)
    woe, _ = binning.transform(process, train, passed)
    kept, dropped = binning.prune_correlated(woe, ivs)
    log.info("feature_selection", candidates=len(names), passed_iv=len(passed), kept=len(kept))
    return process, kept, ivs, dropped


def run() -> pl.DataFrame:
    """Fit the scorecard champion candidate and evaluate on all four splits."""
    panel = pl.read_parquet(PANEL_GLOB)
    train = panel.filter(pl.col("split") == "train")
    process, kept, ivs, pruned = _feature_selection(train)

    woe_train, y_train = binning.transform(process, train, kept)
    card = Scorecard(kept)
    fit = card.fit(woe_train, y_train)
    reference = card.predict_proba(woe_train)

    rows: list[evaluate.SplitMetrics] = []
    curves: dict[str, pl.DataFrame] = {}
    lifts: dict[str, pl.DataFrame] = {}
    for split in evaluate.SPLITS:
        frame = panel.filter(pl.col("split") == split)
        woe, y = binning.transform(process, frame, card.features)
        p = card.predict_proba(woe)
        rows.append(evaluate.evaluate_split(split, y, p, None if split == "train" else reference))
        curves[split], lifts[split] = evaluate.reliability_and_lift(y, p)
        if split == "oot_benign":  # finding F-001
            y_ex, p_ex = evaluate.excluding_forbearance(frame, y, p)
            name = "oot_benign_ex_forbearance"
            rows.append(evaluate.evaluate_split(name, y_ex, p_ex, reference))
            curves[name], lifts[name] = evaluate.reliability_and_lift(y_ex, p_ex)

    metrics = evaluate.metrics_frame(rows)
    _write_artifacts(process, card, fit, ivs, pruned, metrics, curves, lifts)
    _log_mlflow(card, fit, metrics)
    return metrics


def _write_artifacts(
    process: Any,
    card: Scorecard,
    fit: Any,
    ivs: list[binning.FeatureIV],
    pruned: list[dict[str, object]],
    metrics: pl.DataFrame,
    curves: dict[str, pl.DataFrame],
    lifts: dict[str, pl.DataFrame],
) -> None:
    """Every table as CSV, so the Quarto report never needs a refit."""
    evaluate.write_exhibits(
        [evaluate.SplitMetrics(**r) for r in metrics.to_dicts()], curves, lifts, ARTIFACTS
    )
    pl.DataFrame([asdict(f) for f in ivs]).write_csv(ARTIFACTS / "information_values.csv")
    pl.DataFrame(pruned or [{"dropped": None, "kept": None, "correlation": None}]).write_csv(
        ARTIFACTS / "collinearity_pruned.csv"
    )
    card.coefficient_table().write_csv(ARTIFACTS / "scorecard_coefficients.csv")
    binning.monotonicity_report(process).write_csv(ARTIFACTS / "monotonicity_check.csv")
    binning.save_binning_tables(process, card.features, ARTIFACTS / "binning")
    card.save(MODELS / "scorecard.json")
    # The runnable artefact, as distinct from the fit record above: the binning
    # process and the explanation baseline travel with the estimator, because a
    # scorecard cannot be reconstructed from coefficients alone.
    card.save_bundle(MODELS / "scorecard_bundle", process)
    (MODELS / "scorecard_features.json").write_text(
        json.dumps(
            {"features": card.features, "dropped_wrong_sign": fit.dropped_wrong_sign}, indent=2
        ),
        encoding="utf-8",
    )
    plots.reliability(dict(curves))
    plots.discrimination_vs_calibration(metrics)


def _log_mlflow(card: Scorecard, fit: Any, metrics: pl.DataFrame) -> None:
    """Track the run. Local SQLite backend, no server (build plan §5)."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    cfg = feature_config().binning
    with mlflow.start_run(run_name="scorecard-champion-candidate"):
        mlflow.log_params(
            {
                "model_family": "woe_scorecard",
                "n_features": len(card.features),
                "iv_floor": cfg.iv_floor,
                "iv_leakage_ceiling": cfg.iv_leakage_ceiling,
                "max_correlation": cfg.max_correlation,
                "max_n_bins": cfg.max_n_bins,
                "min_bin_size": cfg.min_bin_size,
                "pdo": card.scaling.pdo,
                "base_score": card.scaling.base_score,
                "base_odds": card.scaling.base_odds,
                "n_train": fit.n_train,
                "dropped_wrong_sign": ",".join(fit.dropped_wrong_sign) or "none",
            }
        )
        for row in metrics.to_dicts():
            split = row["split"]
            for metric in ("auc", "gini", "ks", "brier", "reliability", "observed_over_expected"):
                mlflow.log_metric(f"{split}__{metric}", float(row[metric]))
        mlflow.log_artifacts(str(ARTIFACTS), artifact_path="figures")
        log.info("logged_mlflow", experiment=EXPERIMENT)
