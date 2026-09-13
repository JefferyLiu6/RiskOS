"""Typed config for Phase 4: challenger hyperparameters and the selection rubric."""

from __future__ import annotations

import itertools
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import model_validator

from riskos.config import CONF_DIR, Strict, load_yaml


class CandidatePool(Strict):
    source: str
    exclude: tuple[str, ...]


class LightGBMConfig(Strict):
    fixed_params: dict[str, Any]
    grid: dict[str, list[int]]
    early_stopping_rounds: int
    monotone_constraints: dict[str, int]

    def combinations(self) -> list[dict[str, int]]:
        """Expand the fixed grid. Small and documented, never unbounded."""
        keys = sorted(self.grid)
        return [
            dict(zip(keys, values, strict=True))
            for values in itertools.product(*(self.grid[k] for k in keys))
        ]

    def constraint_vector(self, features: list[str]) -> list[int]:
        """Monotone constraints in the model's feature order.

        0 means unconstrained. LightGBM cannot constrain categorical features,
        so any categorical named here would be silently ignored — the caller
        checks for that rather than trusting it.
        """
        return [self.monotone_constraints.get(name, 0) for name in features]


class CalibrationConfig(Strict):
    methods: tuple[str, ...]
    fit_on: str


class ExplainabilityScoring(Strict):
    exact_decomposition_bonus: float
    approximate_attribution: float


class DimensionScoring(Strict):
    metric: str | None = None
    scale: tuple[float, float] | None = None
    transform: str | None = None
    zero_score_at_ratio: float | None = None
    zero_score_at_psi: float | None = None
    zero_score_at_microseconds_per_row: float | None = None
    exact_decomposition_bonus: float | None = None
    approximate_attribution: float | None = None


class SelectionConfig(Strict):
    weights: dict[str, float]
    rationale: dict[str, str]
    scoring: dict[str, DimensionScoring]
    tie_breaker: str

    @model_validator(mode="after")
    def _weights_are_coherent(self) -> SelectionConfig:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"selection weights sum to {total}, not 1")
        missing = sorted(set(self.weights) - set(self.rationale))
        if missing:
            raise ValueError(f"weighted dimension without a written rationale: {missing}")
        unscored = sorted(set(self.weights) - set(self.scoring))
        if unscored:
            raise ValueError(f"weighted dimension with no scoring rule: {unscored}")
        return self


class ModelsConfig(Strict):
    candidate_pool: CandidatePool
    lightgbm: LightGBMConfig
    calibration: CalibrationConfig
    selection: SelectionConfig


@lru_cache(maxsize=1)
def models_config(conf_dir: Path = CONF_DIR) -> ModelsConfig:
    return ModelsConfig.model_validate(load_yaml(conf_dir / "models.yaml"))
