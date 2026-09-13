"""Phase 4 — explanation mechanisms compared (build plan §7.7).

The question to answer is not "what does SHAP say" but:

    Which explanation mechanism is suitable for communicating the principal
    drivers of a credit decision to a borrower or an adjudicator, and why?

So this module puts the two mechanisms side by side on the *same* borrowers.

The structural difference matters more than any single plot:

* **Scorecard points** are an exact decomposition. Contributions plus a constant
  reproduce the score to floating-point precision, because the model *is* a sum
  of per-feature terms. Nothing is approximated and nothing depends on a
  reference population beyond the centring, which is fixed at fit time.
* **SHAP values** are an attribution, not a decomposition. They are additive
  relative to a baseline — the expected model output over a background sample —
  so a SHAP value answers "how much did this feature move the prediction away
  from the average borrower", which is a different and weaker claim.

Both are defensible; they are not interchangeable, and the difference is exactly
what a model reviewer would press on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import shap

from riskos.log import get_logger
from riskos.models.lgbm import Challenger
from riskos.models.scorecard import Scorecard

log = get_logger(__name__)


@dataclass(frozen=True)
class ExplanationComparison:
    """Agreement between the two mechanisms on the same borrowers."""

    n_borrowers: int
    top_driver_agreement: float
    top3_overlap: float
    scorecard_reconstruction_error: float
    shap_reconstruction_error: float


def shap_values(challenger: Challenger, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
    """TreeExplainer SHAP values in log-odds space, for the positive class."""
    explainer = shap.TreeExplainer(challenger.model)
    values = explainer.shap_values(frame[challenger.features])
    if isinstance(values, list):  # older API returns one array per class
        values = values[1]
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 3:  # (rows, features, classes)
        array = array[:, :, -1]
    return array


def shap_drivers(
    challenger: Challenger, frame: pd.DataFrame, top: int = 5
) -> list[list[dict[str, float]]]:
    """Per-borrower principal drivers from SHAP, ranked by absolute impact."""
    values = shap_values(challenger, frame)
    out: list[list[dict[str, float]]] = []
    for row in values:
        order = np.argsort(-np.abs(row))[:top]
        out.append([{"feature": challenger.features[i], "impact": float(row[i])} for i in order])
    return out


def _names(drivers: list[dict[str, float]]) -> list[str]:
    return [str(d["feature"]) for d in drivers]


def compare(
    card: Scorecard,
    woe: pd.DataFrame,
    challenger: Challenger,
    raw: pd.DataFrame,
    top: int = 5,
) -> tuple[pl.DataFrame, ExplanationComparison]:
    """Side-by-side explanations for the same borrowers, plus agreement metrics.

    Agreement is reported because it is informative either way. High agreement
    says the two model families found the same drivers and the choice between
    them is about mechanism rather than substance. Low agreement is a finding:
    two models with similar accuracy telling a borrower different stories about
    why they were declined.
    """
    card_drivers = card.principal_drivers(woe, top=top)
    gbm_drivers = shap_drivers(challenger, raw, top=top)

    rows = []
    top1_hits = 0
    overlaps = []
    for i, (a, b) in enumerate(zip(card_drivers, gbm_drivers, strict=True)):
        card_names, gbm_names = _names(a), _names(b)
        top1 = card_names[0] == gbm_names[0]
        top1_hits += int(top1)
        overlap = len(set(card_names[:3]) & set(gbm_names[:3])) / 3.0
        overlaps.append(overlap)
        rows.append(
            {
                "borrower": i,
                "scorecard_top_driver": card_names[0],
                "scorecard_points": round(float(a[0]["points"]), 2),
                "lgbm_top_driver": gbm_names[0],
                "lgbm_shap": round(float(b[0]["impact"]), 4),
                "same_top_driver": top1,
                "top3_overlap": round(overlap, 3),
            }
        )

    # Reconstruction error: does the mechanism reproduce the model's own output?
    points = card.points_table(woe)
    constant = card.scaling.offset - card.scaling.factor * (card._fit.intercept if card._fit else 0)
    card_error = float(np.max(np.abs(points.sum(axis=1).to_numpy() + constant - card.score(woe))))
    # SHAP's additivity check: values plus the base value should reproduce the
    # model's raw margin (log-odds). expected_value is per-class for a binary
    # classifier, so the positive class is taken explicitly.
    explainer = shap.TreeExplainer(challenger.model)
    base = float(np.atleast_1d(np.asarray(explainer.expected_value))[-1])
    shap_sum = shap_values(challenger, raw).sum(axis=1) + base
    p = np.clip(challenger.predict_proba(raw), 1e-12, 1.0 - 1e-12)
    margin = np.log(p / (1.0 - p))
    shap_error = float(np.max(np.abs(shap_sum - margin)))

    summary = ExplanationComparison(
        n_borrowers=len(rows),
        top_driver_agreement=top1_hits / len(rows) if rows else 0.0,
        top3_overlap=float(np.mean(overlaps)) if overlaps else 0.0,
        scorecard_reconstruction_error=card_error,
        shap_reconstruction_error=shap_error,
    )
    log.info(
        "explanation_comparison",
        borrowers=summary.n_borrowers,
        top_driver_agreement=round(summary.top_driver_agreement, 3),
        top3_overlap=round(summary.top3_overlap, 3),
    )
    return pl.DataFrame(rows), summary
