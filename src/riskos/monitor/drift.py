"""Drift monitoring: score PSI and per-feature CSI, window by window.

Both are computed against the development sample, with bins fixed by that
reference and never re-derived from the window under test. Re-deriving them
makes both sides uniform by construction and drives PSI toward zero exactly when
the population has moved most.

Two things are measured, and they answer different questions:

* **Score PSI** — has the model's *output* distribution moved. This is what a
  provisioning or origination process feels directly, because it is the
  distribution the cut-offs and the portfolio aggregate sit on.
* **Feature CSI** — has any single *input* moved. A stable score can hide
  offsetting input shifts, and a moved input with a stable score is a different
  problem from a moved score, so neither substitutes for the other.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import numpy.typing as npt
import polars as pl

from riskos.log import get_logger
from riskos.metrics.stability import MODERATE_SHIFT, SIGNIFICANT_SHIFT, classify, csi, psi

log = get_logger(__name__)


@dataclass(frozen=True)
class DriftWindow:
    """Drift for one monitoring window against the development sample."""

    window: str
    window_end: date
    n_rows: int
    score_psi: float
    score_psi_band: str
    score_psi_floored_bins: int
    max_feature_csi: float
    worst_feature: str
    # How many of the worst feature's bins were empty on one side and floored to
    # epsilon. A CSI carried by floored bins is an artefact of the floor, not a
    # measurement of drift, and the count is the only way to tell the two apart
    # from the timeline alone.
    worst_feature_floored_bins: int
    worst_feature_n_bins: int
    n_features_csi_moderate: int
    n_features_csi_significant: int


def _window_drift(
    label: str,
    window_end: date,
    reference_scores: npt.NDArray[np.float64],
    reference_frame: pl.DataFrame,
    scores: npt.NDArray[np.float64],
    frame: pl.DataFrame,
    features: list[str],
    n_bins: int,
) -> tuple[DriftWindow, pl.DataFrame]:
    shift = psi(reference_scores, scores, n_bins=n_bins, context=f"score:{label}")
    per_feature = csi(reference_frame, frame, features=features, n_bins=n_bins)
    worst = per_feature.row(0, named=True)
    window = DriftWindow(
        window=label,
        window_end=window_end,
        n_rows=frame.height,
        score_psi=shift.value,
        score_psi_band=shift.band,
        score_psi_floored_bins=shift.empty_bins,
        max_feature_csi=float(worst["csi"]),
        worst_feature=str(worst["feature"]),
        worst_feature_floored_bins=int(worst["empty_bins"]),
        worst_feature_n_bins=int(worst["n_bins"]),
        n_features_csi_moderate=int(
            per_feature.filter(pl.col("csi").is_between(MODERATE_SHIFT, SIGNIFICANT_SHIFT)).height
        ),
        n_features_csi_significant=int(
            per_feature.filter(pl.col("csi") > SIGNIFICANT_SHIFT).height
        ),
    )
    return window, per_feature.with_columns(
        pl.lit(label).alias("window"), pl.lit(window_end).alias("window_end")
    )


def drift_timeline(
    reference_scores: npt.NDArray[np.float64],
    reference_frame: pl.DataFrame,
    windows: list[tuple[str, date, npt.NDArray[np.float64], pl.DataFrame]],
    features: list[str],
    n_bins: int = 10,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """PSI and CSI for every window. Returns the timeline and the CSI detail.

    The detail frame is kept because a PSI headline is not actionable on its
    own: "the population moved" is where the investigation starts, and which
    feature moved is where it goes next.
    """
    rows: list[DriftWindow] = []
    details: list[pl.DataFrame] = []
    for label, window_end, scores, frame in windows:
        if frame.is_empty():
            log.warning("empty_monitoring_window", window=label)
            continue
        window, detail = _window_drift(
            label,
            window_end,
            reference_scores,
            reference_frame,
            scores,
            frame,
            features,
            n_bins,
        )
        rows.append(window)
        details.append(detail)
        log.info(
            "drift_window",
            window=label,
            n=window.n_rows,
            score_psi=round(window.score_psi, 4),
            band=window.score_psi_band,
            worst_feature=window.worst_feature,
            max_csi=round(window.max_feature_csi, 4),
            floored_bins=window.worst_feature_floored_bins,
        )
    timeline = pl.DataFrame([asdict(r) for r in rows])
    return timeline, pl.concat(details) if details else pl.DataFrame()


def epsilon_dominated(timeline: pl.DataFrame, share: float = 0.5) -> pl.DataFrame:
    """Windows whose worst CSI is carried by empty bins rather than by data.

    When more than ``share`` of a feature's bins are empty on one side, the
    reported CSI is largely a function of the epsilon floor registered as
    PSI_002. The number is still reported — suppressing it would hide a real
    population change — but it is labelled, because "this feature has moved a
    long way" and "this feature has no overlap with the reference at all" call
    for different responses.
    """
    if timeline.is_empty():
        return timeline
    return timeline.filter(
        pl.col("worst_feature_floored_bins") > share * pl.col("worst_feature_n_bins")
    ).select(
        [
            "window",
            "worst_feature",
            "max_feature_csi",
            "worst_feature_floored_bins",
            "worst_feature_n_bins",
        ]
    )


def band_summary(timeline: pl.DataFrame) -> pl.DataFrame:
    """Windows per stability band, so the timeline has a one-line reading."""
    if timeline.is_empty():
        return pl.DataFrame()
    return (
        timeline.group_by("score_psi_band")
        .agg(
            pl.len().alias("windows"),
            pl.col("window").min().alias("first_window"),
            pl.col("window").max().alias("last_window"),
        )
        .with_columns(
            pl.col("score_psi_band")
            .replace_strict({"stable": 0, "moderate": 1, "significant": 2}, return_dtype=pl.Int64)
            .alias("_rank")
        )
        .sort("_rank")
        .drop("_rank")
    )


__all__ = ["DriftWindow", "band_summary", "classify", "drift_timeline", "epsilon_dominated"]
