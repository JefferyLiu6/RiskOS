"""Calibration: Brier score with decomposition, and the reliability curve.

For a risk model these matter more than discrimination. A model can rank
borrowers perfectly and still put the wrong number on the balance sheet, and
only the calibration terms show it (build plan §1, §7.5).

The Brier score decomposes (Murphy) over K forecast bins as::

    BS = reliability - resolution + uncertainty

Reliability is the term that matters here: mean squared gap between predicted
and observed rates within bin. Lower is better. Resolution rewards separating
the population away from the base rate, and uncertainty is a property of the
outcome alone, not of the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.stats import norm

from riskos.log import get_logger
from riskos.metrics._common import FloatArray, IntArray, as_float, assign_bins, quantile_edges

log = get_logger(__name__)

# Registered in conf/assumptions.yaml as CAL_001 (interval width) and CAL_002 (bins).
DEFAULT_CONFIDENCE = 0.95
DEFAULT_N_BINS = 10

BinStrategy = Literal["quantile", "uniform"]


@dataclass(frozen=True)
class BrierDecomposition:
    """Brier score and its Murphy components.

    ``residual`` is the gap between the directly computed score and the
    decomposition, which arises because a continuous forecast is binned before
    decomposing. It is reported rather than hidden: a large residual means the
    bins are too coarse to describe the forecast.
    """

    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    residual: float
    n_bins: int


def _z(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return float(norm.ppf(0.5 + confidence / 2.0))


def _wilson(successes: FloatArray, trials: FloatArray, z: float) -> tuple[FloatArray, FloatArray]:
    """Vectorised Wilson score interval.

    Preferred over the normal approximation because default rates are small and
    bin counts in the safest deciles can be tiny, where the normal interval
    runs below zero and understates uncertainty.
    """
    p = successes / trials
    denom = 1.0 + z**2 / trials
    centre = (p + z**2 / (2.0 * trials)) / denom
    half = (z / denom) * np.sqrt(p * (1.0 - p) / trials + z**2 / (4.0 * trials**2))
    return np.clip(centre - half, 0.0, 1.0), np.clip(centre + half, 0.0, 1.0)


def wilson_interval(
    successes: int, trials: int, *, confidence: float = DEFAULT_CONFIDENCE, z: float | None = None
) -> tuple[float, float]:
    """Wilson score interval for a single observed proportion."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes {successes} outside [0, {trials}]")
    low, high = _wilson(
        np.array([successes], dtype=np.float64),
        np.array([trials], dtype=np.float64),
        z if z is not None else _z(confidence),
    )
    return float(low[0]), float(high[0])


def brier_score(y_true: npt.ArrayLike, y_prob: npt.ArrayLike) -> float:
    """Mean squared error of the predicted probabilities."""
    y = as_float(y_true)
    p = as_float(y_prob)
    if y.shape != p.shape:
        raise ValueError(f"length mismatch: {y.size} labels vs {p.size} probabilities")
    return float(np.mean((p - y) ** 2))


def _bin_edges(p: FloatArray, n_bins: int, strategy: BinStrategy) -> FloatArray:
    if strategy == "quantile":
        return quantile_edges(p, n_bins)
    return np.concatenate(([-np.inf], np.linspace(0.0, 1.0, n_bins + 1)[1:-1], [np.inf]))


def _bin_stats(
    y: FloatArray, p: FloatArray, bins: IntArray, n_bins: int
) -> tuple[IntArray, FloatArray, FloatArray]:
    """Per-bin count, mean prediction, and observed rate. Empty bins dropped."""
    counts = np.bincount(bins, minlength=n_bins)
    sum_p = np.bincount(bins, weights=p, minlength=n_bins)
    sum_y = np.bincount(bins, weights=y, minlength=n_bins)
    keep = counts > 0
    n = counts[keep].astype(np.int64)
    return n, sum_p[keep] / n, sum_y[keep] / n


def brier_decomposition(
    y_true: npt.ArrayLike,
    y_prob: npt.ArrayLike,
    *,
    n_bins: int = DEFAULT_N_BINS,
    strategy: BinStrategy = "quantile",
) -> BrierDecomposition:
    """Decompose the Brier score into reliability, resolution, and uncertainty."""
    y, p = as_float(y_true), as_float(y_prob)
    total = y.size
    edges = _bin_edges(p, n_bins, strategy)
    n, mean_p, obs = _bin_stats(y, p, assign_bins(p, edges), edges.size - 1)

    base = float(y.mean())
    reliability = float(np.sum(n * (mean_p - obs) ** 2) / total)
    resolution = float(np.sum(n * (obs - base) ** 2) / total)
    uncertainty = base * (1.0 - base)
    brier = brier_score(y, p)

    return BrierDecomposition(
        brier=brier,
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        residual=brier - (reliability - resolution + uncertainty),
        n_bins=int(n.size),
    )


def reliability_curve(
    y_true: npt.ArrayLike,
    y_prob: npt.ArrayLike,
    *,
    n_bins: int = DEFAULT_N_BINS,
    strategy: BinStrategy = "quantile",
    confidence: float = DEFAULT_CONFIDENCE,
) -> pl.DataFrame:
    """Predicted PD against observed default rate, with binomial intervals.

    The centrepiece of the calibration section (build plan §7.5). A bin whose
    interval excludes the predicted rate is miscalibrated at this confidence
    level, and ``within_interval`` flags exactly that.
    """
    y, p = as_float(y_true), as_float(y_prob)
    if y.shape != p.shape:
        raise ValueError(f"length mismatch: {y.size} labels vs {p.size} probabilities")

    edges = _bin_edges(p, n_bins, strategy)
    n, mean_p, obs = _bin_stats(y, p, assign_bins(p, edges), edges.size - 1)
    low, high = _wilson(obs * n, n.astype(np.float64), _z(confidence))

    return pl.DataFrame(
        {
            "bin": np.arange(1, n.size + 1),
            "n": n,
            "n_default": np.rint(obs * n).astype(np.int64),
            "mean_predicted": mean_p,
            "observed_rate": obs,
            "ci_low": low,
            "ci_high": high,
            "within_interval": (mean_p >= low) & (mean_p <= high),
        }
    )


def observed_vs_expected(y_true: npt.ArrayLike, y_prob: npt.ArrayLike) -> dict[str, float]:
    """Portfolio-level calibration: observed rate, expected rate, and their ratio.

    A ratio above 1 means the model under-predicts default, which understates
    ECL. Reported alongside the curve because a model can be well calibrated in
    aggregate while badly calibrated in every band.
    """
    y, p = as_float(y_true), as_float(y_prob)
    observed, expected = float(y.mean()), float(p.mean())
    return {
        "observed_rate": observed,
        "expected_rate": expected,
        "ratio": observed / expected if expected > 0 else float("nan"),
        "n": float(y.size),
    }
