"""Phase 4 — champion/challenger selection on a pre-committed rubric (§7.6).

Every scoring transform is fixed in ``conf/models.yaml`` before the comparison
runs. Nothing here chooses a winner; it applies arithmetic that was settled in
advance to numbers produced afterwards.

The five dimensions and their weights come from build plan §1's stated ordering:
calibration and stability above discrimination, because a model that ranks well
and predicts the wrong level still misstates the provision.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import polars as pl

from riskos.log import get_logger
from riskos.models.config import SelectionConfig, models_config

log = get_logger(__name__)


@dataclass
class CandidateScore:
    """One candidate's rubric scores, raw inputs and weighted total."""

    name: str
    dimension_scores: dict[str, float] = field(default_factory=dict)
    raw_inputs: dict[str, float] = field(default_factory=dict)
    weighted_total: float = 0.0


def score_calibration(ratio: float, zero_at: float) -> float:
    """Symmetric in over- and under-prediction, because both misstate ECL.

    A ratio of 1.0 scores 1.0. A ratio of ``zero_at`` or its reciprocal scores
    0.0. Measured on the log scale so that predicting half the true rate and
    twice the true rate are penalised equally, which a linear distance would
    not do.
    """
    if ratio <= 0 or not math.isfinite(ratio):
        return 0.0
    return max(0.0, 1.0 - abs(math.log(ratio)) / math.log(zero_at))


def score_stability(psi: float, zero_at: float) -> float:
    """Linear from 1.0 at no shift to 0.0 at the significant-shift band."""
    return max(0.0, 1.0 - psi / zero_at)


def score_latency(microseconds_per_row: float, zero_at: float) -> float:
    return max(0.0, 1.0 - microseconds_per_row / zero_at)


def score_candidate(
    name: str,
    *,
    oot_stress_gini: float,
    oot_stress_observed_over_expected: float,
    max_oot_score_psi: float,
    explanation_is_exact: bool,
    microseconds_per_row: float,
    cfg: SelectionConfig | None = None,
) -> CandidateScore:
    """Apply the pre-committed rubric to one candidate's measured results."""
    cfg = cfg or models_config().selection
    scoring = cfg.scoring

    explain = scoring["explainability"]
    dimensions = {
        "discrimination": max(0.0, min(1.0, oot_stress_gini)),
        "calibration": score_calibration(
            oot_stress_observed_over_expected, scoring["calibration"].zero_score_at_ratio or 3.0
        ),
        "stability": score_stability(
            max_oot_score_psi, scoring["stability"].zero_score_at_psi or 0.25
        ),
        "explainability": (
            (explain.exact_decomposition_bonus or 1.0)
            if explanation_is_exact
            else (explain.approximate_attribution or 0.6)
        ),
        "latency": score_latency(
            microseconds_per_row,
            scoring["latency"].zero_score_at_microseconds_per_row or 100.0,
        ),
    }
    total = sum(cfg.weights[dim] * value for dim, value in dimensions.items())
    return CandidateScore(
        name=name,
        dimension_scores=dimensions,
        raw_inputs={
            "oot_stress_gini": oot_stress_gini,
            "oot_stress_observed_over_expected": oot_stress_observed_over_expected,
            "max_oot_score_psi": max_oot_score_psi,
            "explanation_is_exact": float(explanation_is_exact),
            "microseconds_per_row": microseconds_per_row,
        },
        weighted_total=total,
    )


def select(
    candidates: list[CandidateScore], cfg: SelectionConfig | None = None
) -> tuple[CandidateScore, str]:
    """Return the winner and a written reason, applying the tie-breaker."""
    cfg = cfg or models_config().selection
    ranked = sorted(candidates, key=lambda c: -c.weighted_total)
    winner, runner_up = ranked[0], ranked[1] if len(ranked) > 1 else None

    if runner_up and abs(winner.weighted_total - runner_up.weighted_total) < 0.02:
        more_explainable = max(ranked[:2], key=lambda c: c.dimension_scores["explainability"])
        reason = (
            f"{winner.name} and {runner_up.name} are within 0.02 "
            f"({winner.weighted_total:.4f} vs {runner_up.weighted_total:.4f}), so the "
            f"pre-committed tie-breaker applies: {cfg.tie_breaker.strip()} "
            f"Selected {more_explainable.name}."
        )
        log.info("selection_tie_break", selected=more_explainable.name)
        return more_explainable, reason

    reason = (
        f"{winner.name} scores {winner.weighted_total:.4f} against "
        f"{runner_up.weighted_total:.4f} for {runner_up.name}"
        if runner_up
        else f"{winner.name} is the only candidate"
    )
    log.info("selection", selected=winner.name, score=round(winner.weighted_total, 4))
    return winner, reason


def scorecard_table(candidates: list[CandidateScore]) -> pl.DataFrame:
    """Rubric results as a table: one row per candidate, one column per dimension."""
    cfg = models_config().selection
    rows = []
    for candidate in candidates:
        row: dict[str, object] = {"candidate": candidate.name}
        for dim, weight in cfg.weights.items():
            row[f"{dim} (w={weight})"] = round(candidate.dimension_scores[dim], 4)
        row["weighted_total"] = round(candidate.weighted_total, 4)
        rows.append(row)
    return pl.DataFrame(rows).sort("weighted_total", descending=True)


def as_records(candidates: list[CandidateScore]) -> list[dict[str, object]]:
    return [asdict(c) for c in candidates]
