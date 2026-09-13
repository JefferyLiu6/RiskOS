"""Metrics, implemented from their definitions rather than imported.

Build plan §5 requires KS, Gini, PSI, CSI, reliability curves, and the Brier
decomposition to be written by hand: each is short, each is a domain skill, and
each is a tested module. Every metric has a unit test with an analytically
hand-computable expected value (§7.5).
"""

from riskos.metrics.calibration import (
    BrierDecomposition,
    brier_decomposition,
    brier_score,
    observed_vs_expected,
    reliability_curve,
    wilson_interval,
)
from riskos.metrics.discrimination import KSResult, auc, gini, ks_statistic, lift_table
from riskos.metrics.stability import PSIResult, classify, csi, csi_for_feature, psi

__all__ = [
    "BrierDecomposition",
    "KSResult",
    "PSIResult",
    "auc",
    "brier_decomposition",
    "brier_score",
    "classify",
    "csi",
    "csi_for_feature",
    "gini",
    "ks_statistic",
    "lift_table",
    "observed_vs_expected",
    "psi",
    "reliability_curve",
    "wilson_interval",
]
