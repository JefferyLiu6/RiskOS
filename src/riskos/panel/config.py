"""Typed config for Phase 2: panel, risk set, labels, splits, leakage control."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator

from riskos.config import CONF_DIR, Strict, load_yaml


class DefaultDefinition(Strict):
    """The locked target (build plan §7.1). Not revised after results are seen."""

    statement: str
    dpd_threshold_days: int
    dpd_status_column: str
    dpd_min_months: int
    dpd_also_default_statuses: tuple[str, ...]
    dpd_unknown_statuses: tuple[str, ...]
    credit_event_terminations: tuple[str, ...]
    not_default: tuple[str, ...]
    excluded_not_counted_as_good: tuple[str, ...]
    whole_loan_sale_note: str
    locked: bool
    locked_note: str


class PanelSpec(Strict):
    design: str
    observation_frequency: str
    observation_start: str
    observation_end: str
    horizon_months: int
    eligible_population: tuple[str, ...]
    label: str
    drop_insufficient_forward_window: bool
    partition_by: str

    @property
    def start_year(self) -> int:
        return int(self.observation_start.split("-")[0])

    @property
    def end_year(self) -> int:
        return int(self.observation_end.split("-")[0])


class Sampling(Strict):
    stratify_by: str
    target_rows_min: int
    target_rows_max: int
    rate: float | None
    seed: int

    @property
    def target(self) -> int:
        """Midpoint of the configured band."""
        return (self.target_rows_min + self.target_rows_max) // 2


class RiskSetSpec(Strict):
    frequency: str
    outcome_enum: tuple[str, ...]
    partition_by: str


class SplitWindow(Strict):
    observation_dates: tuple[str, str]
    purpose: str
    loans: str | None = None


class LoanAllocation(Strict):
    seed: int
    fractions: dict[str, float]

    @model_validator(mode="after")
    def _fractions_sum_to_one(self) -> LoanAllocation:
        total = sum(self.fractions.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split fractions sum to {total}, not 1")
        return self

    def boundaries(self) -> list[tuple[str, int, int]]:
        """Half-open [lo, hi) buckets over 0-999, in declaration order."""
        out: list[tuple[str, int, int]] = []
        cursor = 0
        for name, frac in self.fractions.items():
            width = round(frac * 1000)
            out.append((name, cursor, cursor + width))
            cursor += width
        out[-1] = (out[-1][0], out[-1][1], 1000)  # absorb rounding
        return out


class Splits(Strict):
    train: SplitWindow
    validation_in_time: SplitWindow
    oot_stress: SplitWindow
    oot_benign: SplitWindow
    no_loan_overlap_across: tuple[str, ...]
    loan_allocation: LoanAllocation
    unassigned_window_note: str

    def windows(self) -> dict[str, tuple[str, str]]:
        return {
            "train": self.train.observation_dates,
            "validation_in_time": self.validation_in_time.observation_dates,
            "oot_stress": self.oot_stress.observation_dates,
            "oot_benign": self.oot_benign.observation_dates,
        }


class PanelConfig(Strict):
    default_definition: DefaultDefinition
    panel: PanelSpec
    sampling: Sampling
    risk_set: RiskSetSpec
    splits: Splits


class BinningConfig(Strict):
    """optbinning settings and the economically-motivated monotone constraints."""

    max_n_bins: int
    min_n_bins: int
    min_bin_size: float
    min_prebin_size: float
    iv_floor: float
    iv_leakage_ceiling: float
    iv_cleared: tuple[dict[str, object], ...] = ()
    max_correlation: float = 0.95
    min_bin_size_overrides: dict[str, float] = Field(default_factory=dict)
    min_bin_size_override_rationale: dict[str, str] = Field(default_factory=dict)
    known_duplicates: tuple[dict[str, str], ...] = ()
    exclude_high_cardinality: tuple[str, ...]
    monotonic_trend: dict[str, str]
    monotonic_trend_rationale: dict[str, str]

    @property
    def duplicate_drops(self) -> tuple[str, ...]:
        return tuple(d["drop"] for d in self.known_duplicates)

    @property
    def cleared_high_iv(self) -> frozenset[str]:
        """Features that tripped the IV ceiling and were cleared on evidence."""
        return frozenset(str(entry["feature"]) for entry in self.iv_cleared)

    @model_validator(mode="after")
    def _every_constraint_is_justified(self) -> BinningConfig:
        """A monotone constraint without a written economic reason is a guess."""
        missing = sorted(set(self.monotonic_trend) - set(self.monotonic_trend_rationale))
        if missing:
            raise ValueError(f"monotonic_trend set without a rationale: {missing}")
        allowed = {"ascending", "descending", "auto"}
        bad = {k: v for k, v in self.monotonic_trend.items() if v not in allowed}
        if bad:
            raise ValueError(f"monotonic_trend values must be one of {allowed}: {bad}")
        # Same rule for bin-size overrides: relaxing a guard requires a reason.
        unjustified = sorted(
            set(self.min_bin_size_overrides) - set(self.min_bin_size_override_rationale)
        )
        if unjustified:
            raise ValueError(f"min_bin_size override without a rationale: {unjustified}")
        return self


class FeatureConfig(Strict):
    """Leakage control (build plan §7.3) and binning settings (§7.6)."""

    observation_features: dict[str, tuple[str, ...]]
    forbidden: tuple[str, ...]
    label_and_metadata: tuple[str, ...]
    excluded_for_availability: tuple[dict[str, object], ...]
    binning: BinningConfig

    @property
    def allowed(self) -> tuple[str, ...]:
        return tuple(c for group in self.observation_features.values() for c in group)

    @model_validator(mode="after")
    def _categories_are_disjoint(self) -> FeatureConfig:
        for a, b, label in (
            (self.allowed, self.forbidden, "allowed and forbidden"),
            (self.allowed, self.label_and_metadata, "allowed and label/metadata"),
            (self.forbidden, self.label_and_metadata, "forbidden and label/metadata"),
        ):
            clash = sorted(set(a) & set(b))
            if clash:
                raise ValueError(f"columns are both {label}: {clash}")
        return self


@lru_cache(maxsize=1)
def panel_config(conf_dir: Path = CONF_DIR) -> PanelConfig:
    return PanelConfig.model_validate(load_yaml(conf_dir / "panel.yaml"))


@lru_cache(maxsize=1)
def feature_config(conf_dir: Path = CONF_DIR) -> FeatureConfig:
    return FeatureConfig.model_validate(load_yaml(conf_dir / "features.yaml"))
