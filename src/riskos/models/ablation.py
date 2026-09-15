"""Delinquency ablation on the existing panel, without replacing the main models.

The baseline uses the saved fitted bundles. The ablated scorecard reuses their
training-only, univariate bins, then repeats IV/correlation screening and fitting
without delinquency. The ablated challenger repeats the configured grid without
that feature. This is a retrospective sensitivity study, not a new holdout.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from riskos.features import binning
from riskos.models import evaluate
from riskos.models.lgbm import Challenger, to_frame
from riskos.models.scorecard import Scorecard

FEATURE = "current_loan_delinquency_status"
OUTPUT = Path("reports/figures")


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run() -> pl.DataFrame:
    """Evaluate saved baselines and refitted variants on identical observations."""
    files = sorted(Path("data/panel/panel").glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError("panel absent; run `make panel` before `riskos ablate`")
    card, process, _ = Scorecard.from_bundle(Path("models/scorecard_bundle"))
    gbm, _, _ = Challenger.from_bundle(Path("models/challenger_bundle"))
    names = list(process.variable_names)
    if FEATURE not in names or FEATURE not in gbm.features:
        raise ValueError("baseline bundles must include the delinquency feature")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT / "delinquency_ablation_manifest.json"
    inputs = [
        *files,
        *sorted(Path("conf").glob("*.yaml")),
        Path("uv.lock"),
        *sorted(Path("src/riskos").rglob("*.py")),
        *sorted(Path("models/scorecard_bundle").glob("*")),
        *sorted(Path("models/challenger_bundle").glob("*")),
    ]
    manifest: dict[str, Any] = {
        "status": "started",
        "started_at": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "design": "Retrospective ablation; existing splits reused, no fresh holdout.",
        "baseline": "Reloaded scorecard and LightGBM bundles, uncalibrated.",
        "scorecard_variant": "Reuse training-only univariate bins; reselect and refit without delinquency.",
        "challenger_variant": "Repeat configured grid and validation early stopping without delinquency.",
        "excluded_feature": FEATURE,
        "input_sha256": {str(p): _hash(p) for p in inputs if p.is_file()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    columns = list(
        dict.fromkeys(
            [
                *names,
                *gbm.features,
                "loan_sequence_number",
                "observation_date",
                "split",
                "default_12m",
            ]
        )
    )
    panel = pl.read_parquet(files, columns=columns).sort(
        ["loan_sequence_number", "observation_date"]
    )
    train = panel.filter(pl.col("split") == "train")
    valid = panel.filter(pl.col("split") == "validation_in_time")
    manifest["split_counts"] = (
        panel.group_by("split")
        .agg(n=pl.len(), defaults=pl.col("default_12m").sum())
        .sort("split")
        .to_dicts()
    )
    rows: list[dict[str, Any]] = []

    def score(model: Any, family: str, variant: str) -> None:
        def predict(frame: pl.DataFrame) -> np.ndarray:
            prepared = (
                binning.transform(process, frame, model.features)[0]
                if family == "scorecard"
                else model.prepare(frame)
            )
            return model.predict_proba(prepared)  # type: ignore[no-any-return]

        reference = predict(train)
        for split in evaluate.SPLITS:
            frame = panel.filter(pl.col("split") == split)
            p = reference if split == "train" else predict(frame)
            groups = {
                "all": np.ones(frame.height, dtype=bool),
                "current": frame[FEATURE].cast(pl.String).is_in(["00", "0"]).to_numpy(),
                "30_60_dpd": frame[FEATURE]
                .cast(pl.String)
                .is_in(["01", "02", "1", "2"])
                .to_numpy(),
            }
            for group, mask in groups.items():
                y = frame["default_12m"].to_numpy()[mask]
                if len(y) == 0 or np.unique(y).size < 2:
                    continue
                metrics = evaluate.evaluate_split(split, y, p[mask])
                rows.append(
                    {"model": family, "variant": variant, "group": group, **asdict(metrics)}
                )

    score(card, "scorecard", "with_delinquency")
    score(gbm, "lightgbm", "with_delinquency")
    without = [name for name in names if name != FEATURE]
    ivs = binning.information_values(process, without)
    woe, y = binning.transform(process, train, binning.selected_names(ivs))
    kept, _ = binning.prune_correlated(woe, ivs)
    ablated_card = Scorecard(kept)
    fit = ablated_card.fit(woe[kept], y)
    manifest["scorecard_features"] = fit.features
    score(ablated_card, "scorecard", "without_delinquency")
    del woe
    ablated_gbm = Challenger([name for name in gbm.features if name != FEATURE])
    ch_fit = ablated_gbm.fit(
        to_frame(train, ablated_gbm.features),
        train["default_12m"].to_numpy(),
        to_frame(valid, ablated_gbm.features),
        valid["default_12m"].to_numpy(),
    )
    manifest["challenger_fit"] = asdict(ch_fit)
    score(ablated_gbm, "lightgbm", "without_delinquency")
    out = pl.DataFrame(rows)
    path = OUTPUT / "delinquency_ablation.csv"
    out.write_csv(path)
    manifest.update(
        status="complete", completed_at=datetime.now(UTC).isoformat(), output_sha256=_hash(path)
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out
