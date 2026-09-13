"""Phase 5 — the ECL engine: staging, PD x LGD x EAD, scenarios (build plan §7.9).

    ECL = sum over loans of  PD x LGD x EAD x discount_factor

IFRS 9 staging decides WHICH PD applies:

===== ============================================== =================
Stage Criterion                                       Measurement
===== ============================================== =================
1     No significant increase in credit risk          12-month ECL
2     Significant increase, or the 30+ DPD backstop    lifetime ECL
3     Credit-impaired: 90+ DPD or a credit event       lifetime ECL
===== ============================================== =================

The Stage 1 to 2 boundary is a judgement call, and a commercially enormous one:
it switches a loan from a twelve-month loss estimate to a lifetime one, which
for a mortgage can multiply its provision several times. The threshold is
registered as ECL_001 and the stage-2 population is sensitivity-tested against
it rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from riskos.config import CONF_DIR, load_yaml
from riskos.log import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class StagingResult:
    counts: dict[str, int]
    shares: dict[str, float]
    threshold: float


def _register() -> dict[str, dict[str, Any]]:
    return {a["id"]: a for a in load_yaml(CONF_DIR / "assumptions.yaml")["assumptions"]}


def sicr_threshold() -> float:
    return float(_register()["ECL_001"]["value"]["lifetime_pd_ratio"])


def discount_factors(
    annual_rate: npt.NDArray[np.float64], years: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Discount at the loan's own effective interest rate (build plan §7.9)."""
    return np.asarray(1.0 / (1.0 + annual_rate / 100.0) ** years, dtype=np.float64)


def assign_stage(
    frame: pl.DataFrame,
    pd_lifetime: npt.NDArray[np.float64],
    pd_at_origination: npt.NDArray[np.float64],
    threshold: float,
) -> pl.Expr:
    """IFRS 9 stage per loan. Stage 3 first, then the backstop, then SICR.

    Order matters: a loan that is 90+ DPD is credit-impaired regardless of how
    its lifetime PD compares to origination, and the 30-DPD backstop overrides a
    SICR test that would otherwise leave it in stage 1.
    """
    dpd = frame["current_loan_delinquency_status"].cast(pl.String)
    ratio = pd_lifetime / np.maximum(pd_at_origination, 1e-9)
    expr: pl.Expr = (
        pl.when(dpd.is_in(["03", "04", "05", "06", "07", "08", "09", "RA"]))
        .then(pl.lit(3))
        .when(dpd.is_in(["01", "02"]))
        .then(pl.lit(2))
        .when(pl.Series(ratio) > threshold)
        .then(pl.lit(2))
        .otherwise(pl.lit(1))
        .alias("ifrs9_stage")
    )
    return expr


def compute(
    frame: pl.DataFrame,
    pd_12m: npt.NDArray[np.float64],
    pd_lifetime: npt.NDArray[np.float64],
    lgd: npt.NDArray[np.float64],
    stage: npt.NDArray[np.int64],
    remaining_years: npt.NDArray[np.float64],
) -> pl.DataFrame:
    """Per-loan ECL. Stage 1 uses the 12-month PD, stages 2 and 3 lifetime."""
    ead = frame["current_actual_upb"].to_numpy()
    rate = frame["current_interest_rate"].to_numpy()
    applied_pd = np.where(stage == 1, pd_12m, pd_lifetime)
    horizon = np.where(stage == 1, np.minimum(1.0, remaining_years), remaining_years)
    discount = discount_factors(rate, horizon / 2.0)  # mid-period convention
    return frame.select(["loan_sequence_number", "observation_date"]).with_columns(
        ifrs9_stage=pl.Series(stage),
        pd_applied=pl.Series(applied_pd),
        lgd=pl.Series(lgd),
        ead=pl.Series(ead),
        discount_factor=pl.Series(discount),
        ecl=pl.Series(applied_pd * lgd * ead * discount),
    )


def summarise(ecl: pl.DataFrame) -> pl.DataFrame:
    """ECL by stage, with the reconciliation columns a reviewer will check."""
    total = float(ecl["ecl"].sum() or 0.0)
    return (
        ecl.group_by("ifrs9_stage")
        .agg(
            n_loans=pl.len(),
            ead=pl.col("ead").sum(),
            mean_pd=pl.col("pd_applied").mean(),
            mean_lgd=pl.col("lgd").mean(),
            ecl=pl.col("ecl").sum(),
        )
        .with_columns(
            share_of_ecl=pl.col("ecl") / total,
            coverage_ratio=pl.col("ecl") / pl.col("ead"),
        )
        .sort("ifrs9_stage")
    )


def staging_sensitivity(
    frame: pl.DataFrame,
    pd_lifetime: npt.NDArray[np.float64],
    pd_at_origination: npt.NDArray[np.float64],
    thresholds: tuple[float, ...],
) -> pl.DataFrame:
    """Stage-2 population across candidate SICR thresholds (assumption ECL_001).

    The stage-2 share is the number a reviewer will challenge first, because it
    is the single judgement with the largest effect on the provision.
    """
    rows = []
    for threshold in thresholds:
        stage = frame.select(assign_stage(frame, pd_lifetime, pd_at_origination, threshold))[
            "ifrs9_stage"
        ].to_numpy()
        rows.append(
            {
                "sicr_threshold": threshold,
                "stage_1": int((stage == 1).sum()),
                "stage_2": int((stage == 2).sum()),
                "stage_3": int((stage == 3).sum()),
                "stage_2_share": float((stage == 2).mean()),
            }
        )
    return pl.DataFrame(rows)


def scenario_ecl(
    frame: pl.DataFrame,
    shifts: pl.DataFrame,
    pd_12m: npt.NDArray[np.float64],
    pd_lifetime: npt.NDArray[np.float64],
    base_stage: npt.NDArray[np.int64],
    lgd: npt.NDArray[np.float64],
    remaining_years: npt.NDArray[np.float64],
) -> pl.DataFrame:
    """ECL under each scenario, through the SAME mechanics as the base run.

    Every scenario is put through ``assign_stage`` and ``compute``, so staging,
    the lifetime PD for stages 2 and 3, and discounting at the loan's own rate
    all apply exactly as they do in the base figure. Computing a scenario as a
    bare shifted 12-month PD times LGD times EAD produces a number that is not
    comparable to the base ECL it is meant to stress, and silently drops the
    largest single driver of the provision: the stage mix.

    Both PDs are shifted. Shifting only the 12-month PD would leave stage 2 and
    stage 3 loans - which carry most of the coverage - entirely insensitive to
    the macroeconomic scenario, which is the opposite of the intent.

    **Staging is fixed at the base assignment and not re-derived per scenario.**
    Re-staging under each scenario compares a stressed current lifetime PD
    against an unstressed origination PD, which makes the SICR ratio true for
    essentially the entire book: the first implementation of this function put
    all 112,266 loans into stage 2 in every scenario, tripling base ECL through
    a mechanical artefact rather than through credit deterioration. Holding the
    stage fixed also matches the common practice of staging once on a
    probability-weighted basis and varying only the loss measurement. Per
    scenario re-staging is defensible, but only once the origination-vintage PD
    is itself scenario-conditioned, which requires the per-loan hazard
    projection outstanding under finding F-010.

    The band comes from the macro model's HAC confidence intervals propagated
    through the whole calculation, not applied to the total afterwards.
    """
    from riskos.ecl.macro import apply_shift

    rows = []
    for row in shifts.to_dicts():
        values: dict[str, float] = {}
        stage_mix: dict[str, int] = {}
        for label, key in (
            ("point", "logit_shift"),
            ("low", "logit_shift_low"),
            ("high", "logit_shift_high"),
        ):
            shift = float(row[key])
            pd12_s = apply_shift(pd_12m, shift)
            pdlife_s = apply_shift(pd_lifetime, shift)
            result = compute(frame, pd12_s, pdlife_s, lgd, base_stage, remaining_years)
            values[label] = float(result["ecl"].sum() or 0.0)
            if label == "point":
                stage_mix = {f"stage_{s}": int((base_stage == s).sum()) for s in (1, 2, 3)}
        rows.append(
            {
                "scenario": row["scenario"],
                "weight": row["weight"],
                "ecl": values["point"],
                "ecl_low": values["low"],
                "ecl_high": values["high"],
                "extrapolates": row["extrapolates"],
                **stage_mix,
            }
        )
    out = pl.DataFrame(rows)
    weighted = float((out["ecl"] * out["weight"]).sum())
    log.info("scenario_ecl", probability_weighted=round(weighted, 2))
    return out.with_columns(probability_weighted_ecl=pl.lit(weighted))
