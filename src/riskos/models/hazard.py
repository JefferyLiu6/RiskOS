"""Phase 5 — discrete-time hazard with competing risks (build plan §7.8).

**This replaces chaining the 12-month PD across future periods.** You do not
obtain future conditional PDs by repeating today's 12-month number, and the
empirical seasoning curve shows why: the monthly default hazard rises from
0.00018 in months 0-6 to 0.00145 in months 37-60, then falls back to 0.00100.
A reviewer asking where the year-five marginal PD came from must not
discover the answer is "year one, repeated".

Specification:

* Two **cause-specific hazards**, default and prepayment, so prepayment is a
  competing risk rather than an ignored censoring event. A borrower who
  refinances cannot subsequently default, and a model that ignores that
  overstates lifetime default.
* **Complementary log-log link**, the discrete-time analogue of proportional
  hazards. Unlike logit, its coefficients carry a proportional-hazards
  interpretation and it is the correct link when the underlying process is
  continuous but observed in monthly intervals.
* **Loan-age baseline** as age-band dummies. The seasoning curve is the point of
  the model, not a nuisance parameter.
* **Cluster-robust standard errors by calendar period.** Loan-months are not
  independent: the same borrower appears up to 325 times, and everyone
  experiences 2008 together. Naive standard errors would be far too narrow.

Survival with competing risks::

    12-month PD  = 1 - prod over m=1..12   of (1 - h_default(m))
    lifetime PD  = sum over m=1..T of  h_default(m) * S(m-1)
    S(m)         = prod over j<=m of (1 - h_default(j)) * (1 - h_prepay(j))

The lifetime form is a sum of *marginal* default probabilities, each weighted by
the probability of surviving both risks to that month — not one minus a
default-only survival, which would credit defaults to loans that had already
prepaid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import statsmodels.api as sm

from riskos.log import get_logger

log = get_logger(__name__)

AGE_BANDS = (6, 12, 24, 36, 60, 96, 144)
CAUSES = ("default", "prepaid")

# Delinquency-state bands, in months past due at the START of the month.
#
# A STATE, not a dose. The empirical monthly default hazard on the lagged state
# is 0.000014 when current, 0.001105 at 30-59 days and 0.327108 at 60-89 - a
# span of roughly 23,000x. A single linear term cannot represent that: to reach
# the delinquent level it must push the intercept far enough negative to zero
# out the 98.8% of loan-months that are current. Each state therefore carries
# its own level (finding F-012).
#
# The top band is open-ended at 2+ rather than continuing to 3, 4, ... because
# reaching 90 days past due IS the default event under the project definition:
# a loan at 3 months delinquent has already left the risk set. Bands beyond the
# observed range would be inestimable, and `estimable_columns` would drop them.
DELINQUENCY_BANDS = (0, 1)


def age_band_labels() -> list[str]:
    """Every band label, in order. Fixed regardless of what a sample contains."""
    labels = [f"age_00_{AGE_BANDS[0]:02d}"]
    labels += [f"age_{lo:02d}_{hi:02d}" for lo, hi in pairwise(AGE_BANDS)]
    return [*labels, f"age_{AGE_BANDS[-1]}_plus"]


def age_band(expr: pl.Expr) -> pl.Expr:
    """Age-band label. The baseline hazard, not a control."""
    result: Any = pl.when(expr <= AGE_BANDS[0]).then(pl.lit(f"age_00_{AGE_BANDS[0]:02d}"))
    for lo, hi in pairwise(AGE_BANDS):
        result = result.when(expr <= hi).then(pl.lit(f"age_{lo:02d}_{hi:02d}"))
    final: pl.Expr = result.otherwise(pl.lit(f"age_{AGE_BANDS[-1]}_plus"))
    return final


def delinquency_band_labels() -> list[str]:
    """Every delinquency-state label, in order. Reference level first."""
    labels = [f"dlq_{lo:02d}" for lo in DELINQUENCY_BANDS]
    return [*labels, f"dlq_{DELINQUENCY_BANDS[-1] + 1:02d}_plus"]


def delinquency_band(expr: pl.Expr) -> pl.Expr:
    """Delinquency-state label from months past due at the start of the month."""
    result: Any = pl.when(expr <= DELINQUENCY_BANDS[0]).then(
        pl.lit(f"dlq_{DELINQUENCY_BANDS[0]:02d}")
    )
    for lo in DELINQUENCY_BANDS[1:]:
        result = result.when(expr <= lo).then(pl.lit(f"dlq_{lo:02d}"))
    final: pl.Expr = result.otherwise(pl.lit(f"dlq_{DELINQUENCY_BANDS[-1] + 1:02d}_plus"))
    return final


@dataclass
class HazardFit:
    """One cause-specific hazard model."""

    cause: str
    n_rows: int
    n_events: int
    coefficients: dict[str, float]
    std_errors: dict[str, float]
    conf_int: dict[str, tuple[float, float]]
    baseline: dict[str, float] = field(default_factory=dict)
    cluster_variable: str = "calendar_period"


@dataclass(frozen=True)
class Standardisation:
    """Fitting-sample moments AND imputation medians, carried with the model.

    All three must travel together. Applying a fresh sample's own moments at
    prediction time would silently change what a coefficient multiplies, and
    applying its own medians would make a missing value mean something
    different depending on who else was scored alongside it. Scored alone, a
    row with a missing covariate has no batch median at all and imputes to NaN,
    which propagates silently into the predicted hazard.
    """

    means: dict[str, float]
    stds: dict[str, float]
    medians: dict[str, float]


def estimable_columns(design: pd.DataFrame) -> list[str]:
    """Columns with any variation. All-zero dummies make the fit singular.

    Restricting the fitting window to 1999-2006 means no loan in the sample
    exceeds about 96 months of age, so the oldest age bands are empty. They are
    not merely unhelpful, they are inestimable, and dropping them is honest
    where retaining them would fail. The consequence for lifetime PD is
    recorded as a limitation rather than papered over: the baseline beyond the
    oldest estimable band is an extrapolation.
    """
    keep = [c for c in design.columns if c == "intercept" or design[c].to_numpy().std() > 0]
    dropped = [c for c in design.columns if c not in keep]
    if dropped:
        log.warning("hazard_inestimable_bands", dropped=dropped, reason="no observations in window")
    return keep


def design_matrix(
    risk_set: pl.DataFrame,
    covariates: list[str],
    standardisation: Standardisation | None = None,
    columns: list[str] | None = None,
    delinquency_bands: bool = False,
) -> tuple[pd.DataFrame, pl.DataFrame, Standardisation]:
    """Age-band dummies plus standardised numeric covariates.

    The dummy columns are the FULL band set, not whichever bands the input
    happens to contain. A sample missing a band would otherwise produce a
    narrower matrix than the model was fitted on, and the prediction would
    either fail or — worse — silently misalign coefficients with columns.

    ``delinquency_bands`` adds the delinquency STATE as its own set of dummies,
    read from ``months_delinquent``. That column must carry the state at the
    START of the month; see ``train_hazard.load_risk_set``. Passing the state as
    a member of ``covariates`` instead makes it a linear dose and degenerates
    the fit (finding F-012).
    """
    prepared = risk_set.with_columns(age_band(pl.col("loan_age")).alias("age_band"))
    labels = age_band_labels()
    reference = labels[0]  # first band is the reference level
    observed = prepared["age_band"].to_pandas()
    dummies = pd.DataFrame(
        {band: (observed == band).astype(float).to_numpy() for band in labels[1:]}
    )

    if delinquency_bands:
        prepared = prepared.with_columns(
            delinquency_band(pl.col("months_delinquent")).alias("delinquency_band")
        )
        dlq_labels = delinquency_band_labels()
        dlq_observed = prepared["delinquency_band"].to_pandas()
        for band in dlq_labels[1:]:  # first state (current) is the reference level
            dummies[band] = (dlq_observed == band).astype(float).to_numpy()

    numeric = prepared.select(covariates).to_pandas().astype(float)
    if standardisation is None:
        # Fitting pass: capture medians BEFORE imputing, so the medians describe
        # the observed data rather than the imputed data.
        medians = {c: float(numeric[c].median()) for c in covariates}
        numeric = numeric.fillna(pd.Series(medians))
        standardisation = Standardisation(
            means={c: float(numeric[c].mean()) for c in covariates},
            stds={c: float(numeric[c].std()) or 1.0 for c in covariates},
            medians=medians,
        )
    else:
        numeric = numeric.fillna(pd.Series(standardisation.medians))
    standardised = pd.DataFrame(
        {
            c: (numeric[c] - standardisation.means[c]) / (standardisation.stds[c] or 1.0)
            for c in covariates
        }
    )

    design = pd.concat(
        [dummies.reset_index(drop=True), standardised.reset_index(drop=True)], axis=1
    )
    design.insert(0, "intercept", 1.0)
    if columns is not None:
        # Reindex to the fitted column set: absent bands become zero, and any
        # band the fit could not estimate is dropped rather than silently
        # shifting every coefficient one column to the left.
        design = design.reindex(columns=columns, fill_value=0.0)
    log.debug("hazard_design", rows=len(design), columns=design.shape[1], reference_band=reference)
    return design, prepared, standardisation


def fit_cause(
    design: pd.DataFrame,
    events: npt.NDArray[np.int64],
    clusters: npt.NDArray[Any],
    cause: str,
) -> tuple[Any, HazardFit]:
    """Fit one cause-specific discrete-time hazard by cloglog GLM.

    Standard errors are cluster-robust by calendar period. With 54 million
    correlated loan-months the naive standard errors would be meaningless, and
    an interval that is too narrow is worse than no interval at all.
    """
    model = sm.GLM(events, design, family=sm.families.Binomial(link=sm.families.links.CLogLog()))
    result = model.fit(cov_type="cluster", cov_kwds={"groups": clusters})
    conf = result.conf_int()
    fit = HazardFit(
        cause=cause,
        n_rows=len(events),
        n_events=int(events.sum()),
        coefficients={k: float(v) for k, v in result.params.items()},
        std_errors={k: float(v) for k, v in result.bse.items()},
        conf_int={k: (float(conf.loc[k, 0]), float(conf.loc[k, 1])) for k in result.params.index},
        baseline={k: float(v) for k, v in result.params.items() if k.startswith("age_")},
    )
    log.info(
        "hazard_fitted",
        cause=cause,
        rows=fit.n_rows,
        events=fit.n_events,
        converged=bool(result.converged),
    )
    return result, fit


def predict_hazard(result: Any, design: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Monthly hazard from a fitted cloglog model."""
    return np.asarray(result.predict(design), dtype=np.float64)


def survival_with_competing_risks(
    h_default: npt.NDArray[np.float64],
    h_prepay: npt.NDArray[np.float64],
    convention: str = "default_first",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Cumulative default probability and overall survival, month by month.

    Both hazards are cause-specific and estimated separately, each treating the
    other event as censoring. Within a single discrete month both could in
    principle occur, so a **convention is required** to allocate the joint
    probability ``h_default * h_prepay``. The convention is stated here rather
    than left implicit, because it is a modelling choice and not arithmetic.

    ``default_first`` (the default) resolves default before prepayment::

        P(default)  = h_d
        P(prepay)   = (1 - h_d) * h_p
        P(neither)  = (1 - h_d) * (1 - h_p)

    These sum to exactly 1, so the accounting is closed: the whole joint mass is
    attributed to default. ``symmetric`` instead splits the joint mass between
    the two causes in proportion to their hazards, which is marginally less
    conservative. Over a 300-month horizon at exaggerated hazards the two
    conventions differ by under 1% of lifetime PD; the difference is reported
    rather than assumed away.

    Returns ``(cumulative_default, survival)``. Each month's marginal default
    contribution is weighted by survival to the *previous* month, which is what
    stops the model attributing defaults to loans that already prepaid.
    """
    if convention not in ("default_first", "symmetric"):
        raise ValueError(f"unknown competing-risk convention {convention!r}")
    alive = 1.0
    cumulative = 0.0
    cum_out, surv_out = [], []
    for hd, hp in zip(h_default, h_prepay, strict=True):
        if convention == "default_first":
            p_default = hd
        else:
            joint = hd * hp
            p_default = hd - joint * hd / (hd + hp) if (hd + hp) > 0 else hd
        cumulative += alive * p_default
        alive *= (1.0 - hd) * (1.0 - hp)
        cum_out.append(cumulative)
        surv_out.append(alive)
    return np.asarray(cum_out, dtype=np.float64), np.asarray(surv_out, dtype=np.float64)


def competing_risk_accounting(
    h_default: npt.NDArray[np.float64], h_prepay: npt.NDArray[np.float64]
) -> dict[str, float]:
    """Verify the probabilities close: default + prepay + survive == 1.

    A competing-risk implementation whose outcome probabilities do not sum to
    one is silently creating or destroying loans, so this is asserted rather
    than assumed.
    """
    alive, cum_d, cum_p = 1.0, 0.0, 0.0
    for hd, hp in zip(h_default, h_prepay, strict=True):
        cum_d += alive * hd
        cum_p += alive * (1.0 - hd) * hp
        alive *= (1.0 - hd) * (1.0 - hp)
    return {
        "cumulative_default": cum_d,
        "cumulative_prepay": cum_p,
        "survival": alive,
        "total": cum_d + cum_p + alive,
    }


def term_structure(
    h_default: npt.NDArray[np.float64], h_prepay: npt.NDArray[np.float64], horizon: int = 12
) -> dict[str, float]:
    """12-month and lifetime PD from a single loan's hazard path."""
    cumulative, survival = survival_with_competing_risks(h_default, h_prepay)
    return {
        "pd_12m": float(cumulative[min(horizon, len(cumulative)) - 1]),
        "pd_lifetime": float(cumulative[-1]),
        "survival_end": float(survival[-1]),
        "months": float(len(cumulative)),
    }


def pd_12m_default_only(h_default: npt.NDArray[np.float64], horizon: int = 12) -> float:
    """12-month PD ignoring prepayment, for the competing-risk comparison.

    Reported alongside the competing-risk figure to quantify what ignoring
    prepayment costs. Over twelve months the difference is small; over a
    twenty-five year lifetime it is not, because prepayment removes most of the
    book long before maturity.
    """
    window = h_default[:horizon]
    return float(1.0 - np.prod(1.0 - window))


def coefficient_table(fits: dict[str, HazardFit]) -> pl.DataFrame:
    """Coefficients with cluster-robust intervals, both causes side by side."""
    rows = []
    for cause, fit in fits.items():
        for name, beta in fit.coefficients.items():
            low, high = fit.conf_int[name]
            rows.append(
                {
                    "cause": cause,
                    "term": name,
                    "coefficient": beta,
                    "std_error": fit.std_errors[name],
                    "ci_low": low,
                    "ci_high": high,
                    "significant": not (low <= 0.0 <= high),
                    "hazard_ratio": float(np.exp(beta)),
                }
            )
    return pl.DataFrame(rows)


def baseline_table(fits: dict[str, HazardFit]) -> pl.DataFrame:
    """The seasoning curve as fitted, in hazard-ratio terms."""
    rows = []
    for cause, fit in fits.items():
        for band, beta in sorted(fit.baseline.items()):
            rows.append(
                {
                    "cause": cause,
                    "age_band": band,
                    "log_hazard_ratio": beta,
                    "hazard_ratio": float(np.exp(beta)),
                }
            )
    return pl.DataFrame(rows)


def cloglog_inverse(linear_predictor: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """h = 1 - exp(-exp(eta)), the inverse complementary log-log link."""
    return np.asarray(
        1.0 - np.exp(-np.exp(np.clip(linear_predictor, -30.0, 10.0))), dtype=np.float64
    )


def _band_offsets(coefficients: dict[str, float]) -> dict[str, float]:
    """Age-band contribution to the linear predictor, reference band at zero."""
    labels = age_band_labels()
    return {band: float(coefficients.get(band, 0.0)) for band in labels}


def _band_for_age(ages: npt.NDArray[np.int64]) -> npt.NDArray[np.str_]:
    labels = age_band_labels()
    index = np.searchsorted(np.asarray(AGE_BANDS), ages, side="left")
    return np.asarray(labels, dtype=object)[np.minimum(index, len(labels) - 1)]


def project_paths(
    results: dict[str, Any],
    frame: pl.DataFrame,
    covariates: list[str],
    scaling: Standardisation,
    columns: list[str],
    horizon_months: int,
    start_age: npt.NDArray[np.int64] | None = None,
    term_months: npt.NDArray[np.int64] | None = None,
    convention: str = "default_first",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-loan cumulative default probability over each loan's own horizon.

    Closes finding F-010. Every loan is projected using **its own** covariates
    and **its own** age path, rather than the portfolio-average hazard applied
    uniformly, which is what made the SICR ratio identical for every loan and
    the staging sensitivity vacuous.

    The linear predictor decomposes: the covariate contribution is fixed per
    loan and the only term that varies with the projection month is the
    age-band offset. Computing the covariate part once and adding the band
    offset per month avoids rebuilding a design matrix on every step, which
    turns 300 matrix constructions into 300 vector additions.

    ``term_months`` truncates each loan at its own remaining term, so a loan
    with 60 months left stops accruing default probability at month 60 rather
    than running to the common horizon.

    Returns ``(cumulative_default, survival)``, both per loan.
    """
    design, prepared, _ = design_matrix(frame, covariates, scaling, columns)
    ages = (start_age if start_age is not None else prepared["loan_age"].to_numpy()).astype(
        np.int64
    )

    # Covariate-only linear predictor: zero out every age-band column.
    covariate_only = design.copy()
    for band in age_band_labels():
        if band in covariate_only.columns:
            covariate_only[band] = 0.0

    base: dict[str, npt.NDArray[np.float64]] = {}
    offsets: dict[str, dict[str, float]] = {}
    for cause in CAUSES:
        params = results[cause].params
        aligned = np.asarray([float(params.get(c, 0.0)) for c in design.columns], dtype=np.float64)
        base[cause] = covariate_only.to_numpy(dtype=np.float64) @ aligned
        offsets[cause] = _band_offsets({k: float(v) for k, v in params.items()})

    n = frame.height
    cumulative = np.zeros(n, dtype=np.float64)
    alive = np.ones(n, dtype=np.float64)
    horizon = (
        np.full(n, horizon_months)
        if term_months is None
        else np.minimum(term_months, horizon_months)
    )

    for step in range(horizon_months):
        active = step < horizon
        if not active.any():
            break
        bands = _band_for_age(ages + step)
        hd = cloglog_inverse(base["default"] + np.array([offsets["default"][b] for b in bands]))
        hp = cloglog_inverse(base["prepaid"] + np.array([offsets["prepaid"][b] for b in bands]))
        if convention == "symmetric":
            joint = hd * hp
            p_default = np.where(hd + hp > 0, hd - joint * hd / np.maximum(hd + hp, 1e-12), hd)
        else:
            p_default = hd
        cumulative = np.where(active, cumulative + alive * p_default, cumulative)
        alive = np.where(active, alive * (1.0 - hd) * (1.0 - hp), alive)

    log.info(
        "projected_pd_paths",
        loans=n,
        horizon_months=horizon_months,
        mean_cumulative_default=round(float(cumulative.mean()), 6),
    )
    return cumulative, alive
