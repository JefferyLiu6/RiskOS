"""Shared validation and binning helpers for the metrics package.

Metrics are written by hand rather than imported (build plan §5). These helpers
keep the individual metric implementations short enough to read and check
against their definitions.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from riskos.log import get_logger

log = get_logger(__name__)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class DegenerateInputError(ValueError):
    """A metric is undefined on the input — e.g. only one class is present."""


def as_float(x: npt.ArrayLike) -> FloatArray:
    """Coerce to a 1-D float array, rejecting NaN."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    if np.isnan(arr).any():
        raise ValueError("input contains NaN; impute or drop upstream, never silently")
    return arr


def validate_binary(y_true: npt.ArrayLike, y_score: npt.ArrayLike) -> tuple[IntArray, FloatArray]:
    """Validate a (label, score) pair.

    Labels must be 0/1. Both classes must be present — discrimination metrics
    are undefined otherwise, and returning 0.5 would hide that.
    """
    labels = np.asarray(y_true).ravel()
    scores = as_float(y_score)
    if labels.shape != scores.shape:
        raise ValueError(f"length mismatch: {labels.shape[0]} labels vs {scores.shape[0]} scores")
    if labels.size == 0:
        raise DegenerateInputError("empty input")

    unique = np.unique(labels)
    if not np.isin(unique, (0, 1)).all():
        raise ValueError(f"labels must be 0/1, found {unique.tolist()}")
    if unique.size < 2:
        raise DegenerateInputError(
            f"only class {unique.tolist()[0]} present; discrimination is undefined"
        )
    return labels.astype(np.int64), scores


def quantile_edges(reference: npt.ArrayLike, n_bins: int) -> FloatArray:
    """Bin edges from reference-sample quantiles, open at both ends.

    Returns ``n_bins + 1`` edges with -inf/+inf outer bounds, so a scoring
    window containing values outside the training range still bins cleanly
    instead of dropping rows. Duplicate interior edges (heavy ties or a coarse
    score) collapse, which reduces the effective bin count — logged, never
    silent, because it changes the PSI scale.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")
    ref = as_float(reference)
    probs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    interior = np.unique(np.quantile(ref, probs))
    if interior.size < n_bins - 1:
        log.warning(
            "quantile_edges_collapsed",
            requested_bins=n_bins,
            effective_bins=interior.size + 1,
            reason="tied reference values produced duplicate quantiles",
        )
    return np.concatenate(([-np.inf], interior, [np.inf]))


def assign_bins(values: npt.ArrayLike, edges: FloatArray) -> IntArray:
    """Assign each value to a bin index in ``[0, len(edges) - 2]``.

    Bins are left-open, right-closed: ``(edge[i], edge[i+1]]``.
    """
    idx = np.searchsorted(edges, as_float(values), side="left") - 1
    return np.clip(idx, 0, edges.size - 2).astype(np.int64)


def bin_proportions(values: npt.ArrayLike, edges: FloatArray) -> FloatArray:
    """Share of ``values`` falling in each bin. Sums to 1."""
    arr = as_float(values)
    counts = np.bincount(assign_bins(arr, edges), minlength=edges.size - 1)
    return counts.astype(np.float64) / arr.size
