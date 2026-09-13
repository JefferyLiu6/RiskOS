"""Load every artefact the report reads, recording what is absent."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import polars as pl

from riskos.config import CONF_DIR, PROJECT_ROOT, load_yaml
from riskos.log import get_logger

log = get_logger(__name__)

# Which pipeline command regenerates each artefact, for the missing-artefact line.
PRODUCED_BY: dict[str, str] = {
    "metrics_by_split": "riskos train",
    "information_values": "riskos train",
    "collinearity_pruned": "riskos train",
    "scorecard_coefficients": "riskos train",
    "scorecard_features": "riskos train",
    "scorecard_bundle": "riskos train",
    "champion_challenger_metrics": "riskos challenger",
    "selection_rubric": "riskos challenger",
    "selection_record": "riskos challenger",
    "lgbm_grid": "riskos challenger",
    "explanation_summary": "riskos challenger",
    "challenger_bundle": "riskos challenger",
    "reliability_cc_scorecard_oot_stress": "riskos challenger",
    "reliability_cc_lightgbm_oot_stress": "riskos challenger",
    "hazard_empirical_seasoning": "riskos hazard",
    "hazard_baseline": "riskos hazard",
    "hazard_term_structure": "riskos hazard",
    "hazard_specification_comparison": "riskos hazard",
    "hazard_fit": "riskos hazard",
    "lgd_raw_distribution": "riskos ecl",
    "lgd_tail_investigation": "riskos ecl",
    "lgd_denominator_sensitivity": "riskos ecl",
    "lgd_bounding_sensitivity": "riskos ecl",
    "lgd_segments": "riskos ecl",
    "ecl_by_stage": "riskos ecl",
    "ecl_staging_sensitivity": "riskos ecl",
    "ecl_scenarios": "riskos ecl",
    "macro_coefficients": "riskos ecl",
    "macro_scenario_shifts": "riskos ecl",
    "macro_backtest": "riskos ecl",
    "monitoring_summary": "riskos monitor",
    "inventory_reconciliation": "riskos registry",
    "alert_register": "riskos monitor",
}


@dataclass
class Sources:
    """Everything the report reads, loaded once and cached."""

    root: Path
    figures: Path
    models: Path
    governance: Path
    missing: list[str] = field(default_factory=list)
    _cache: dict[str, Any] = field(default_factory=dict)

    # -- artefacts -------------------------------------------------------------

    def csv(self, name: str) -> pl.DataFrame | None:
        return cast(
            pl.DataFrame | None, self._load(name, self.figures / f"{name}.csv", pl.read_csv)
        )

    def json(self, name: str, directory: Path | None = None) -> dict[str, Any] | None:
        path = (directory or self.figures) / f"{name}.json"
        return cast(dict[str, Any] | None, self._load(name, path, _read_json))

    def manifest(self, bundle: str) -> dict[str, Any] | None:
        return cast(
            dict[str, Any] | None,
            self._load(f"{bundle}/manifest", self.models / bundle / "manifest.json", _read_json),
        )

    def bundle_manifest(self, artefact: str | Path | None) -> dict[str, Any] | None:
        """The manifest of an inventory artefact, if that artefact is a bundle directory.

        A loose artefact such as a fitted-coefficients JSON has no manifest, and
        looking for one would record a spurious missing artefact.
        """
        if artefact is None:
            return None
        path = self.root / artefact
        if not path.is_dir() or not (path / "manifest.json").exists():
            return None
        return self.manifest(path.name)

    def _load(self, key: str, path: Path, reader: Any) -> Any:
        if key in self._cache:
            return self._cache[key]
        if not path.exists():
            if key not in self.missing:
                self.missing.append(key)
            self._cache[key] = None
            return None
        self._cache[key] = reader(path)
        return self._cache[key]

    def produced_by(self, key: str) -> str:
        return PRODUCED_BY.get(key.split("/")[0], "the pipeline")

    # -- governance and config ---------------------------------------------------

    @property
    def findings(self) -> list[dict[str, Any]]:
        return list(load_yaml(self.governance / "findings_register.yaml")["findings"])

    @property
    def inventory(self) -> dict[str, Any]:
        return load_yaml(self.governance / "model_inventory.yaml")

    @property
    def reconciliation(self) -> pl.DataFrame | None:
        return cast(
            pl.DataFrame | None,
            self._load(
                "inventory_reconciliation",
                self.governance / "inventory_reconciliation.csv",
                pl.read_csv,
            ),
        )

    @property
    def alert_register(self) -> pl.DataFrame | None:
        return cast(
            pl.DataFrame | None,
            self._load("alert_register", self.governance / "alert_register.csv", pl.read_csv),
        )

    @property
    def assumptions(self) -> list[dict[str, Any]]:
        return list(load_yaml(CONF_DIR / "assumptions.yaml")["assumptions"])

    def conf(self, name: str) -> dict[str, Any]:
        return load_yaml(CONF_DIR / f"{name}.yaml")

    def finding(self, finding_id: str) -> dict[str, Any] | None:
        return next((f for f in self.findings if f["id"] == finding_id), None)

    def assumption(self, assumption_id: str) -> dict[str, Any] | None:
        return next((a for a in self.assumptions if a["id"] == assumption_id), None)


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def load_sources(root: Path = PROJECT_ROOT) -> Sources:
    return Sources(
        root=root,
        figures=root / "reports" / "figures",
        models=root / "models",
        governance=root / "governance",
    )
