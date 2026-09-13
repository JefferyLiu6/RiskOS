"""Population and characteristic stability: PSI and CSI.

    PSI = sum over bins i of (a_i - e_i) * ln(a_i / e_i)

with ``e`` the expected (training reference) share and ``a`` the actual share.
CSI is the identical computation applied per feature rather than to the score.

Conventional reading (build plan §7.5): below 0.10 stable, 0.10-0.25 moderate,
above 0.25 significant. These cut-offs are a cited industry convention, not an
empirical result, and are registered as such in ``conf/assumptions.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from riskos.log import get_logger
from riskos.metrics._common import (
    FloatArray,
    as_float,
    assign_bins,
    bin_proportions,
    quantile_edges,
)

log = get_logger(__name__)

# Registered in conf/assumptions.yaml as PSI_001 (bands) and PSI_002 (epsilon).
MODERATE_SHIFT = 0.10
SIGNIFICANT_SHIFT = 0.25
DEFAULT_EPSILON = 1e-6
DEFAULT_N_BINS = 10

UNSEEN_LEVEL = "__unseen__"
MISSING_LEVEL = "__missing__"


@dataclass(frozen=True)
class PSIResult:
    """A PSI value with the per-bin detail needed to explain it."""

    value: float
    band: str
    n_bins: int
    empty_bins: int
    detail: pl.DataFrame

    @property
    def is_significant(self) -> bool:
        return self.value > SIGNIFICANT_SHIFT


def classify(value: float) -> str:
    """Map a PSI/CSI value onto the conventional stability bands."""
    if value < MODERATE_SHIFT:
        return "stable"
    return "moderate" if value <= SIGNIFICANT_SHIFT else "significant"


def psi_from_proportions(
    expected: npt.ArrayLike,
    actual: npt.ArrayLike,
    *,
    epsilon: float = DEFAULT_EPSILON,
    labels: list[str] | None = None,
    context: str = "",
) -> PSIResult:
    """Core PSI computation over two aligned proportion vectors.

    An empty bin makes the log term undefined, so a floor of ``epsilon`` is
    substituted. Every substitution is logged (build plan §7.5) because an
    epsilon-dominated bin can carry most of the reported PSI, and a PSI driven
    by a floor rather than by data is not evidence of drift.
    """
    exp, act = as_float(expected), as_float(actual)
    if exp.shape != act.shape:
        raise ValueError(f"bin count mismatch: expected {exp.size}, actual {act.size}")
    for name, vec in (("expected", exp), ("actual", act)):
        if not np.isclose(vec.sum(), 1.0, atol=1e-8):
            raise ValueError(f"{name} proportions sum to {vec.sum():.6f}, not 1")

    empty = int(((exp <= 0) | (act <= 0)).sum())
    if empty:
        log.warning(
            "psi_empty_bin",
            context=context or "unnamed",
            empty_bins=empty,
            total_bins=exp.size,
            epsilon=epsilon,
            note="floored to epsilon; treat the resulting PSI as a lower-confidence figure",
        )
    exp_f = np.maximum(exp, epsilon)
    act_f = np.maximum(act, epsilon)
    contribution: FloatArray = (act_f - exp_f) * np.log(act_f / exp_f)

    detail = pl.DataFrame(
        {
            "bin": labels or [str(i) for i in range(exp.size)],
            "expected": exp,
            "actual": act,
            "contribution": contribution,
            "floored": (exp <= 0) | (act <= 0),
        }
    )
    value = float(contribution.sum())
    return PSIResult(
        value=value, band=classify(value), n_bins=exp.size, empty_bins=empty, detail=detail
    )


def psi(
    reference: npt.ArrayLike,
    actual: npt.ArrayLike,
    *,
    n_bins: int = DEFAULT_N_BINS,
    epsilon: float = DEFAULT_EPSILON,
    context: str = "score",
) -> PSIResult:
    """PSI of a scoring window against the training reference.

    Bins are the reference-sample quantiles — training-set score deciles by
    default (build plan §7.5) — so they are fixed by the reference and never
    re-derived from the window being tested.
    """
    edges = quantile_edges(reference, n_bins)
    labels = [f"({edges[i]:.6g}, {edges[i + 1]:.6g}]" for i in range(edges.size - 1)]
    return psi_from_proportions(
        bin_proportions(reference, edges),
        bin_proportions(actual, edges),
        epsilon=epsilon,
        labels=labels,
        context=context,
    )


def _categorical_proportions(
    reference: pl.Series, actual: pl.Series
) -> tuple[FloatArray, FloatArray, list[str]]:
    """Share per level, plus a missing bin and an actual-only bucket.

    Missingness is a bin, not a row to discard. A feature whose null rate moves
    from 50% to 0% has drifted enormously, and dropping nulls on both sides
    reports a CSI of exactly zero - the opposite of the truth, and a direct
    contradiction of the rule that a meaningful absence is information.
    """
    levels = [str(v) for v in reference.drop_nulls().unique().sort().to_list()]
    ref_counts = reference.cast(pl.String).value_counts().to_dict(as_series=False)
    act_counts = actual.cast(pl.String).value_counts().to_dict(as_series=False)
    ref_map = dict(zip(ref_counts[reference.name], ref_counts["count"], strict=True))
    act_map = dict(zip(act_counts[actual.name], act_counts["count"], strict=True))

    unseen = sorted(lv for lv in set(act_map) - set(ref_map) if lv is not None)
    if unseen:
        log.warning("csi_unseen_levels", feature=reference.name, levels=unseen[:10])
    exp = np.array(
        [ref_map.get(lv, 0) for lv in levels] + [ref_map.get(None, 0), 0],
        dtype=np.float64,
    )
    act = np.array(
        [act_map.get(lv, 0) for lv in levels]
        + [act_map.get(None, 0), sum(act_map[lv] for lv in unseen)],
        dtype=np.float64,
    )
    return exp / exp.sum(), act / act.sum(), [*levels, MISSING_LEVEL, UNSEEN_LEVEL]


def _numeric_proportions(
    reference: pl.Series, actual: pl.Series, n_bins: int
) -> tuple[FloatArray, FloatArray, list[str]]:
    """Quantile-binned shares with an explicit missing bin.

    Bins come from the reference sample's non-null values, but the denominator
    is the FULL count on each side, so a shift in the null rate shows up as
    drift rather than being normalised away.
    """
    edges = quantile_edges(reference.drop_nulls().to_numpy(), n_bins)
    labels = [f"({edges[i]:.6g}, {edges[i + 1]:.6g}]" for i in range(edges.size - 1)]

    def shares(series: pl.Series) -> FloatArray:
        values = series.drop_nulls().to_numpy()
        counts = np.bincount(assign_bins(values, edges), minlength=edges.size - 1)
        with_missing = np.append(counts, series.null_count()).astype(np.float64)
        return with_missing / series.len()

    return shares(reference), shares(actual), [*labels, MISSING_LEVEL]


def csi_for_feature(
    reference: pl.Series,
    actual: pl.Series,
    *,
    n_bins: int = DEFAULT_N_BINS,
    epsilon: float = DEFAULT_EPSILON,
) -> PSIResult:
    """CSI for one feature, with missingness treated as its own bin."""
    if reference.dtype.is_numeric():
        exp, act, labels = _numeric_proportions(reference, actual, n_bins)
        return psi_from_proportions(
            exp, act, epsilon=epsilon, labels=labels, context=str(reference.name)
        )
    exp, act, labels = _categorical_proportions(reference, actual)
    return psi_from_proportions(
        exp, act, epsilon=epsilon, labels=labels, context=str(reference.name)
    )


def csi(
    reference: pl.DataFrame,
    actual: pl.DataFrame,
    *,
    features: list[str] | None = None,
    n_bins: int = DEFAULT_N_BINS,
) -> pl.DataFrame:
    """CSI for every feature, ranked by contribution to the total shift."""
    names = features or [c for c in reference.columns if c in actual.columns]
    rows = [
        {
            "feature": name,
            "csi": (r := csi_for_feature(reference[name], actual[name], n_bins=n_bins)).value,
            "band": r.band,
            "n_bins": r.n_bins,
            "empty_bins": r.empty_bins,
        }
        for name in names
    ]
    return pl.DataFrame(rows).sort("csi", descending=True)
