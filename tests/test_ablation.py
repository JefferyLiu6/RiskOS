"""Checks on the public ablation evidence, without refitting licensed data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

FIGURES = Path("reports/figures")


def test_ablation_baselines_reproduce_the_published_comparison() -> None:
    study = pl.read_csv(FIGURES / "delinquency_ablation.csv")
    baseline = study.filter((pl.col("variant") == "with_delinquency") & (pl.col("group") == "all"))
    published = pl.read_csv(FIGURES / "champion_challenger_metrics.csv").filter(
        pl.col("calibration") == "uncalibrated"
    )
    paired = baseline.join(published, on=["model", "split"], suffix="_published", validate="1:1")
    assert paired.height == 8
    for row in paired.to_dicts():
        for metric in ("n", "n_default", "auc", "observed_over_expected"):
            assert row[metric] == pytest.approx(row[f"{metric}_published"], rel=1e-9)


def test_ablation_compares_matching_cohorts_and_records_excluded_features() -> None:
    study = pl.read_csv(FIGURES / "delinquency_ablation.csv")
    assert study.height == 48  # two families, two variants, four splits, three cohorts
    for group in study.partition_by(["model", "split", "group"]):
        assert set(group["variant"]) == {"with_delinquency", "without_delinquency"}
        assert group["n"].n_unique() == 1
        assert group["n_default"].n_unique() == 1
    manifest = json.loads((FIGURES / "delinquency_ablation_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["excluded_feature"] not in manifest["scorecard_features"]
    assert manifest["excluded_feature"] not in manifest["challenger_fit"]["features"]
    assert len(manifest["challenger_fit"]["grid_results"]) == 6
    assert (
        hashlib.sha256((FIGURES / "delinquency_ablation.csv").read_bytes()).hexdigest()
        == manifest["output_sha256"]
    )
