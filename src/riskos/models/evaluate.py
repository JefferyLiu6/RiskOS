"""Phase 3 — evaluation across all four splits (build plan §7.4, §7.5).

Every metric is reported on train, in-time validation, OOT-stress and
OOT-benign. The comparison that matters is not the level on any one split but
the *shape* across them: discrimination and calibration degrade at very
different rates under regime change, and only reporting both makes that visible.

Per finding F-001, OOT-benign is additionally reported with COVID
forbearance-driven defaults excluded, because 60% of that split's defaults are
policy artefacts rather than credit events.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

from riskos.log import get_logger
from riskos.metrics import (
    auc,
    brier_decomposition,
    gini,
    ks_statistic,
    lift_table,
    observed_vs_expected,
    psi,
    reliability_curve,
)

log = get_logger(__name__)

SPLITS = ("train", "validation_in_time", "oot_stress", "oot_benign")


@dataclass
class SplitMetrics:
    """Discrimination, calibration and stability for one split."""

    split: str
    n: int
    n_default: int
    observed_rate: float
    expected_rate: float
    observed_over_expected: float
    auc: float
    gini: float
    ks: float
    ks_decile: int
    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    score_psi: float | None
    psi_band: str | None


def evaluate_split(
    name: str,
    y: npt.NDArray[np.int64],
    p: npt.NDArray[np.float64],
    reference_scores: npt.NDArray[np.float64] | None = None,
) -> SplitMetrics:
    """All metrics for one split. PSI is against the training reference."""
    ks = ks_statistic(y, p)
    brier = brier_decomposition(y, p)
    oe = observed_vs_expected(y, p)
    shift = (
        psi(reference_scores, p, context=f"score:{name}") if reference_scores is not None else None
    )
    return SplitMetrics(
        split=name,
        n=int(y.size),
        n_default=int(y.sum()),
        observed_rate=oe["observed_rate"],
        expected_rate=oe["expected_rate"],
        observed_over_expected=oe["ratio"],
        auc=auc(y, p),
        gini=gini(y, p),
        ks=ks.statistic,
        ks_decile=ks.decile,
        brier=brier.brier,
        reliability=brier.reliability,
        resolution=brier.resolution,
        uncertainty=brier.uncertainty,
        score_psi=shift.value if shift else None,
        psi_band=shift.band if shift else None,
    )


def metrics_frame(rows: list[SplitMetrics]) -> pl.DataFrame:
    return pl.DataFrame([asdict(r) for r in rows])


def excluding_forbearance(
    frame: pl.DataFrame, y: npt.NDArray[np.int64], p: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Drop rows whose default is COVID forbearance-driven (finding F-001).

    Rows are removed rather than relabelled: a forbearance-driven 90+ DPD is not
    evidence the loan was performing, it is an unobservable outcome under a
    policy intervention. Relabelling to 0 would assert something the data cannot
    support.
    """
    keep = ~frame["default_in_forbearance"].to_numpy()
    return y[keep], p[keep]


def write_exhibits(
    rows: list[SplitMetrics],
    curves: dict[str, pl.DataFrame],
    lifts: dict[str, pl.DataFrame],
    out_dir: Path,
) -> None:
    """Persist every table as CSV so the report regenerates without a refit."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame(rows).write_csv(out_dir / "metrics_by_split.csv")
    for name, curve in curves.items():
        curve.write_csv(out_dir / f"reliability_{name}.csv")
    for name, lift in lifts.items():
        lift.write_csv(out_dir / f"lift_{name}.csv")
    log.info("wrote_evaluation_exhibits", path=str(out_dir), splits=len(rows))


def reliability_curve_for(y: npt.NDArray[np.int64], p: npt.NDArray[np.float64]) -> pl.DataFrame:
    """Reliability curve alone, for callers that do not need the lift table."""
    return reliability_curve(y, p)


def reliability_and_lift(
    y: npt.NDArray[np.int64], p: npt.NDArray[np.float64]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    return reliability_curve(y, p), lift_table(y, p)
