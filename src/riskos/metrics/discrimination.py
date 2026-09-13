"""Discrimination: AUC, Gini, KS, and the gains/lift table.

Implemented from the definitions rather than imported (build plan §5).

Decile convention, used consistently across this package: observations are
ranked by predicted risk and split into ten equal-sized groups, with **decile 1
holding the riskiest** and decile 10 the safest. Reversing this silently is a
classic way to make a gains table look inverted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from riskos.log import get_logger
from riskos.metrics._common import FloatArray, validate_binary

log = get_logger(__name__)

N_DECILES = 10


@dataclass(frozen=True)
class KSResult:
    """KS statistic with the location at which it is attained."""

    statistic: float
    threshold: float
    decile: int
    n_positive: int
    n_negative: int


def _average_ranks(x: FloatArray) -> FloatArray:
    """1-based ranks, ties sharing their average rank."""
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ordered = x[order]
    ranks = np.empty(n, dtype=np.float64)
    start = 0
    while start < n:
        stop = start
        while stop + 1 < n and ordered[stop + 1] == ordered[start]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop + 2) / 2.0
        start = stop + 1
    return ranks


def auc(y_true: npt.ArrayLike, y_score: npt.ArrayLike) -> float:
    """Area under the ROC curve, via the Mann-Whitney rank statistic.

    Equals the probability that a randomly chosen defaulter scores above a
    randomly chosen non-defaulter, with ties counting a half.
    """
    labels, scores = validate_binary(y_true, y_score)
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    rank_sum = float(_average_ranks(scores)[labels == 1].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def gini(y_true: npt.ArrayLike, y_score: npt.ArrayLike) -> float:
    """Gini coefficient: ``2 x AUC - 1`` (build plan §7.5)."""
    return 2.0 * auc(y_true, y_score) - 1.0


def _risk_decile(index_ascending: int, n: int) -> int:
    """Decile of an observation at ``index_ascending`` in a risk-ascending sort."""
    from_riskiest = (n - index_ascending) / n
    return int(min(N_DECILES, max(1, np.ceil(from_riskiest * N_DECILES))))


def ks_statistic(y_true: npt.ArrayLike, y_score: npt.ArrayLike) -> KSResult:
    """Maximum separation between the defaulter and non-defaulter score CDFs.

    Evaluated only at tie-group boundaries: a threshold falling inside a group
    of equal scores is not attainable, and ignoring that inflates KS.
    """
    labels, scores = validate_binary(y_true, y_score)
    order = np.argsort(scores, kind="mergesort")
    ordered_labels, ordered_scores = labels[order], scores[order]

    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    cum_pos = np.cumsum(ordered_labels) / n_pos
    cum_neg = np.cumsum(1 - ordered_labels) / n_neg

    boundary = np.append(ordered_scores[:-1] != ordered_scores[1:], True)
    separation = np.where(boundary, np.abs(cum_pos - cum_neg), -np.inf)
    at = int(np.argmax(separation))

    return KSResult(
        statistic=float(separation[at]),
        threshold=float(ordered_scores[at]),
        decile=_risk_decile(at, labels.size),
        n_positive=n_pos,
        n_negative=n_neg,
    )


def lift_table(
    y_true: npt.ArrayLike, y_score: npt.ArrayLike, n_bands: int = N_DECILES
) -> pl.DataFrame:
    """Gains table by risk band, band 1 riskiest.

    Bands are equal-sized by rank, so ties may straddle a boundary; with a
    continuous PD this is immaterial, and the band counts are reported so it
    stays visible when it is not.
    """
    labels, scores = validate_binary(y_true, y_score)
    n = labels.size
    order = np.argsort(-scores, kind="mergesort")  # riskiest first
    band = np.minimum((np.arange(n) * n_bands) // n + 1, n_bands)

    frame = pl.DataFrame({"band": band, "default": labels[order]})
    overall = float(labels.mean())
    agg = (
        frame.group_by("band")
        .agg(n=pl.len(), n_default=pl.col("default").sum())
        .sort("band")
        .with_columns(default_rate=pl.col("n_default") / pl.col("n"))
        .with_columns(
            cum_n=pl.col("n").cum_sum(),
            cum_default=pl.col("n_default").cum_sum(),
            lift=pl.col("default_rate") / overall,
        )
    )
    return agg.with_columns(
        cum_capture_rate=pl.col("cum_default") / int(labels.sum()),
        cum_population_share=pl.col("cum_n") / n,
    ).with_columns(cum_lift=pl.col("cum_capture_rate") / pl.col("cum_population_share"))
