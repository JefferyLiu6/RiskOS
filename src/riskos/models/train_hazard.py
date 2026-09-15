"""Phase 5 orchestration: fit both cause-specific hazards and reconcile.

The reconciliation is the acceptance criterion (build plan §7.8): the
hazard-implied 12-month PD is compared against the direct 12-month model on the
same population. Material disagreement means one of them is wrong, so it is
investigated and reported either way.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import numpy.typing as npt
import polars as pl

from riskos.config import CONF_DIR, load_yaml
from riskos.log import get_logger
from riskos.models import hazard
from riskos.panel.alignment import require_same_observations

log = get_logger(__name__)

RISK_SET_GLOB = "data/panel/risk_set/*/*.parquet"
ARTIFACTS = Path("reports/figures")
MODELS = Path("models")
# Matches the direct 12-month model's training window (build plan §7.4).
TRAIN_END_YEAR = 2006

# Origination covariates: fixed for the life of the loan.
STATIC_COVARIATES = [
    "credit_score",
    "original_ltv",
    "original_dti",
    "original_interest_rate",
    "original_upb",
    "number_of_borrowers",
]

# Time-varying covariates, added to close finding F-011. Without at least one
# covariate that updates as credit quality changes, the model's estimate of a
# loan's lifetime risk today differs from its estimate at origination only
# through seasoning, the SICR ratio sits at 1.0 for every loan, and IFRS 9
# staging collapses onto the 30-DPD backstop.
#
# `mtm_ltv` is the economically important one: mark-to-market loan-to-value,
# which moves with BOTH amortisation and house prices. Negative equity is the
# central driver of mortgage default, and it is precisely what a static
# original LTV cannot express.
#
# `months_delinquent` is the strongest single predictor in the dataset, and is
# retained with the caveat recorded in F-004: it is mechanically close to the
# target, so the model is fitted and reported both with and without it.
TIME_VARYING_COVARIATES = ["mtm_ltv", "amortisation_ratio"]

# PRIMARY specification. `months_delinquent` is deliberately NOT here, which is
# independently required by F-004 - the model must be reported without the
# variable that is mechanically closest to the target.
COVARIATES = STATIC_COVARIATES + TIME_VARYING_COVARIATES

# The DEGENERATE variant, retained only so the regression test can assert it
# stays broken. Entering the delinquency state as a linear dose collapses the
# intercept to about -27 and zeroes the hazard for current loans.
#
# F-012 recorded the cause as a range problem. It is not, or not only: before
# the state was lagged, `months_delinquent` was measured contemporaneously and
# default is DEFINED as reaching 90 days past due, so the covariate WAS the
# outcome - hazard exactly 1.0 at 3+ months delinquent, exactly 0.0 when
# current. That is perfect separation, the same contamination class as R-009.
# The lag in `load_risk_set` fixes it. What remains after the lag is the genuine
# range problem F-012 described: 0.000014 to 0.327108, roughly 23,000x, which a
# single linear term still cannot hold.
COVARIATES_WITH_DELINQUENCY = [*COVARIATES, "months_delinquent"]

# The REMEDIATED variant (F-012): state lagged, then entered as band dummies so
# each delinquency state carries its own level instead of sharing a slope. The
# covariate list is the primary one - the state enters through
# `delinquency_bands=True` on the design matrix, not as a numeric column.
COVARIATES_BANDED = COVARIATES


def _sample_size() -> int:
    register = {a["id"]: a for a in load_yaml(CONF_DIR / "assumptions.yaml")["assumptions"]}
    return int(register["HAZ_001"]["value"]["n_loans"])


def load_risk_set(
    n_loans: int,
    seed: int = 20250901,
    max_year: int = TRAIN_END_YEAR,
    lag_delinquency: bool = True,
) -> pl.DataFrame:
    """Loan-level sample of the risk set, restricted to the training window.

    Sampling is by LOAN, never by loan-month: taking a random sample of months
    would break each loan's survival path, which is the object the hazard model
    estimates. Registered as assumption HAZ_001.

    The window restriction is not cosmetic. Fitting across all periods gives a
    baseline hazard averaged over 1999-2026, which includes the crisis: the
    monthly default hazard is 0.000528 in 1999-2006 against 0.001017 across all
    periods, a factor of 1.93. A model fitted on everything and compared against
    a 1999-2006 panel therefore over-predicts by roughly that factor for reasons
    that have nothing to do with model form. Both models must see the same
    training window for the reconciliation in §7.8 to mean anything, and for the
    §7.10 two-stage macro design to hold - the loan-level model is meant to
    carry idiosyncratic risk, with the cycle supplied by the overlay.

    ``lag_delinquency`` exists only so a regression test can reconstruct the
    contaminated covariate and assert the contamination is real, on the same
    principle as the R-009 tests. Fitting on it is never correct.
    """
    status_sql = (
        "CASE WHEN r.current_loan_delinquency_status = 'RA' THEN 12 "
        "     ELSE COALESCE("
        "         TRY_CAST(r.current_loan_delinquency_status AS INTEGER), 0) "
        "END"
    )
    delinquency_sql = (
        f"COALESCE(LAG({status_sql}) OVER (PARTITION BY r.loan_sequence_number "
        "ORDER BY r.observation_date), 0)"
        if lag_delinquency
        else status_sql
    )
    con = duckdb.connect()
    try:
        con.execute(
            "CREATE TEMP VIEW hpi AS SELECT date, value "
            "FROM read_parquet('data/macro/SPCS20RSA.parquet') WHERE value IS NOT NULL"
        )
        return con.execute(f"""
            WITH sampled AS (
                SELECT DISTINCT loan_sequence_number
                FROM read_parquet('{RISK_SET_GLOB}')
                WHERE hash(loan_sequence_number || '{seed}') % 1000000
                      < {int(1_000_000 * n_loans / 947_027)}
            )
            SELECT r.*,
                   YEAR(r.observation_date) * 10 + QUARTER(r.observation_date)
                       AS calendar_period,
                   -- LAGGED by one month, deliberately. The balance in a
                   -- termination month is zero: 100.00% of prepayment months
                   -- carry a zero balance against 0.00% of alive months, so the
                   -- contemporaneous ratio reveals the outcome exactly and
                   -- separates the prepayment hazard perfectly. The covariate
                   -- driving the hazard FOR month m must be known at the start
                   -- of month m, which is the §7.3 rule applied per risk-set
                   -- month. The first month of a loan has no predecessor and
                   -- falls back to its original balance. See finding R-009.
                   COALESCE(
                       LAG(r.current_actual_upb) OVER (
                           PARTITION BY r.loan_sequence_number
                           ORDER BY r.observation_date),
                       r.original_upb
                   ) / NULLIF(r.original_upb, 0) AS amortisation_ratio,
                   -- Months delinquent, LAGGED by one month for the same
                   -- reason as the balance above, and it matters far more here.
                   -- Default is DEFINED as reaching 90 days past due, so the
                   -- contemporaneous status IS the outcome: measured on the
                   -- fitting window the monthly default hazard is exactly
                   -- 1.000000 for every row at 3+ months delinquent and exactly
                   -- 0.000000 for every current row. That is perfect
                   -- separation, and it is what drove the intercept to -27.4
                   -- and zeroed the hazard for current loans - not the range
                   -- problem originally recorded in F-012.
                   --
                   -- Lagged, the state is a genuine predictor known at the
                   -- start of the month: hazard 0.000014 current, 0.001105 at
                   -- 30-59 days, 0.327108 at 60-89. A loan's first observation
                   -- has no predecessor and is current by construction.
                   -- 'RA' (REO acquisition) is the most severe observable
                   -- state and is mapped above the numeric range, not to null.
                   {delinquency_sql} AS months_delinquent,
                   -- Mark-to-market LTV. Both HPI values are at or before the
                   -- observation month, never after it: the origination index
                   -- is historical and the current index is contemporaneous.
                   -- Where origination predates the HPI series the ratio falls
                   -- back to 1.0, leaving amortisation only (assumption HAZ_004).
                   -- Same one-month lag on the balance, for the same reason.
                   r.original_ltv
                       * (COALESCE(
                              LAG(r.current_actual_upb) OVER (
                                  PARTITION BY r.loan_sequence_number
                                  ORDER BY r.observation_date),
                              r.original_upb
                          ) / NULLIF(r.original_upb, 0))
                       * COALESCE(h_orig.value / NULLIF(h_now.value, 0), 1.0)
                       AS mtm_ltv
            FROM read_parquet('{RISK_SET_GLOB}') r
            JOIN sampled USING (loan_sequence_number)
            LEFT JOIN hpi h_now
                   ON h_now.date = DATE_TRUNC('month', r.observation_date)
            LEFT JOIN hpi h_orig
                   ON h_orig.date = DATE_TRUNC(
                          'month', r.observation_date - INTERVAL (r.loan_age) MONTH)
            WHERE YEAR(r.observation_date) <= {max_year}
            -- Deterministic order. DuckDB's parallel scan returns rows in a
            -- different order run to run, which moves the standardisation
            -- moments in their last bits and makes every downstream PD - and
            -- therefore the reported provision - irreproducible at the 1e-14
            -- level. Cheap to pin, and a provisioning figure that changes
            -- between identical runs cannot be reconciled by a reviewer.
            ORDER BY r.loan_sequence_number, r.observation_date
        """).pl()
    finally:
        con.close()


@dataclass
class Specification:
    """One fitted hazard specification, with everything needed to score it."""

    label: str
    covariates: list[str]
    delinquency_bands: bool
    results: dict[str, Any]
    fits: dict[str, hazard.HazardFit]
    design: Any
    prepared: pl.DataFrame
    scaling: hazard.Standardisation
    columns: list[str]


def fit_specification(
    risk_set: pl.DataFrame,
    covariates: list[str],
    label: str,
    delinquency_bands: bool = False,
) -> Specification:
    """Fit both cause-specific hazards for one specification."""
    design, prepared, scaling = hazard.design_matrix(
        risk_set, covariates, delinquency_bands=delinquency_bands
    )
    kept = hazard.estimable_columns(design)
    design = design[kept]
    clusters = prepared["calendar_period"].to_numpy()

    results: dict[str, Any] = {}
    fits: dict[str, hazard.HazardFit] = {}
    for cause in hazard.CAUSES:
        events = (prepared["outcome"] == cause).cast(pl.Int64).to_numpy()
        results[cause], fits[cause] = hazard.fit_cause(design, events, clusters, cause)

    return Specification(
        label=label,
        covariates=covariates,
        delinquency_bands=delinquency_bands,
        results=results,
        fits=fits,
        design=design,
        prepared=prepared,
        scaling=scaling,
        columns=kept,
    )


def run() -> dict[str, Any]:
    """Fit both hazards, derive the term structure, and reconcile.

    Two specifications are fitted. The PRIMARY excludes the delinquency state,
    which F-004 requires independently. The BANDED variant adds it as state
    dummies and is the F-012 remediation; it is reported alongside rather than
    silently promoted, because adopting it would move every downstream ECL
    figure and that is a decision for the review, not for this function.
    """
    risk_set = load_risk_set(_sample_size())
    log.info(
        "hazard_sample",
        loans=risk_set["loan_sequence_number"].n_unique(),
        loan_months=risk_set.height,
    )

    primary = fit_specification(risk_set, COVARIATES, "primary")
    reconciliation = _reconcile(primary.results, primary.design, primary.prepared)
    reconciliation.update(
        _reconcile_against_panel(
            primary.results, primary.covariates, primary.scaling, primary.columns
        )
    )
    _write(primary.fits, reconciliation, primary.results, primary.design, primary.prepared)

    banded = fit_specification(
        risk_set, COVARIATES_BANDED, "delinquency_banded", delinquency_bands=True
    )
    comparison = _compare_specifications(primary, banded)

    return {
        "fits": primary.fits,
        "reconciliation": reconciliation,
        "specification_comparison": comparison,
    }


def _compare_specifications(primary: Specification, banded: Specification) -> list[dict[str, Any]]:
    """Primary against the delinquency-banded variant, on the same population.

    The comparison the F-012 remediation has to pass: the banded specification
    must be non-degenerate. A collapsed intercept and a near-zero hazard for
    current loans is exactly the failure the linear term produced, so the
    intercept and the mean hazard are reported for both.
    """
    rows: list[dict[str, Any]] = []
    for spec in (primary, banded):
        internal = _reconcile(spec.results, spec.design, spec.prepared)
        panel = _reconcile_against_panel(
            spec.results,
            spec.covariates,
            spec.scaling,
            spec.columns,
            delinquency_bands=spec.delinquency_bands,
        )
        rows.append(
            {
                "specification": spec.label,
                "delinquency_bands": spec.delinquency_bands,
                "n_terms": len(spec.columns),
                "intercept_default": spec.fits["default"].coefficients["intercept"],
                "mean_monthly_hazard_default": internal["mean_monthly_hazard_default"],
                "realised_monthly_default_rate": internal["realised_monthly_default_rate"],
                "hazard_implied_pd_12m": panel["hazard_implied_pd_12m"],
                "panel_observed_default_rate_12m": panel["panel_observed_default_rate_12m"],
                "ratio_hazard_to_observed": panel["ratio_hazard_to_observed"],
                **{
                    f"coef_{band}": spec.fits["default"].coefficients.get(band)
                    for band in hazard.delinquency_band_labels()[1:]
                },
            }
        )
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, strict=False).write_csv(ARTIFACTS / "hazard_specification_comparison.csv")
    log.info("wrote_specification_comparison", specifications=len(rows))
    return rows


def project_12m_pd(
    results: dict[str, Any],
    prepared: pl.DataFrame,
    covariates: list[str],
    scaling: hazard.Standardisation,
    columns: list[str],
    horizon: int = 12,
    delinquency_bands: bool = False,
) -> npt.NDArray[np.float64]:
    """Competing-risk 12-month PD for each row, ageing the loan forward.

    The loan is aged month by month across the horizon so it can cross an
    age-band boundary, which is the entire reason a hazard model differs from
    repeating a single monthly rate. Building one design matrix per future month
    is the honest way to do it.
    """
    cumulative = np.zeros(prepared.height, dtype=np.float64)
    alive = np.ones(prepared.height, dtype=np.float64)
    for step in range(horizon):
        aged = prepared.with_columns((pl.col("loan_age") + step).alias("loan_age"))
        design, _, _ = hazard.design_matrix(
            aged, covariates, scaling, columns, delinquency_bands=delinquency_bands
        )
        hd = hazard.predict_hazard(results["default"], design)
        hp = hazard.predict_hazard(results["prepaid"], design)
        cumulative += alive * hd
        alive *= (1.0 - hd) * (1.0 - hp)
    return cumulative


def _reconcile_against_panel(
    results: dict[str, Any],
    covariates: list[str],
    scaling: hazard.Standardisation,
    columns: list[str],
    n_sample: int = 60_000,
    delinquency_bands: bool = False,
) -> dict[str, float]:
    """The Phase 5 acceptance criterion (build plan §7.8).

    Hazard-implied 12-month PD against the direct 12-month model, on the SAME
    population: the training-split panel rows. Both are restricted to the same
    loans, the same observation dates and the same forward window, so any
    disagreement is about model form rather than about who was measured.
    """
    con = duckdb.connect()
    try:
        panel = con.execute(f"""
            SELECT * FROM read_parquet('data/panel/panel/*/*.parquet')
            WHERE split = 'train'
            USING SAMPLE {n_sample} ROWS (reservoir, 20250901)
        """).pl()
    finally:
        con.close()

    # The panel carries origination fields only. The time-varying covariates
    # must be derived on it with the SAME definitions used at fitting, or the
    # design matrix is missing columns outright - which is what broke this
    # acceptance criterion when `mtm_ltv` and `amortisation_ratio` were added
    # to the specification and nothing re-ran it.
    panel = with_time_varying(panel)

    predicted = project_12m_pd(
        results, panel, covariates, scaling, columns, delinquency_bands=delinquency_bands
    )
    observed = float(panel["default_12m"].cast(pl.Float64).to_numpy().mean())
    out = {
        "panel_rows_compared": float(panel.height),
        "hazard_implied_pd_12m": float(predicted.mean()),
        "panel_observed_default_rate_12m": observed,
        "ratio_hazard_to_observed": float(predicted.mean()) / observed
        if observed
        else float("nan"),
    }
    log.info("hazard_panel_reconciliation", **{k: round(v, 6) for k, v in out.items()})
    return out


def _reconcile(results: dict[str, Any], design: Any, prepared: pl.DataFrame) -> dict[str, float]:
    """Internal consistency: fitted hazard against the empirical hazard.

    A weaker check than the panel reconciliation, but a necessary one: it
    confirms the GLM reproduces the average monthly rate in its own sample
    before anything is projected forward.
    """
    h_default = hazard.predict_hazard(results["default"], design)
    h_prepay = hazard.predict_hazard(results["prepaid"], design)

    # Portfolio-average monthly hazards, converted to a 12-month figure.
    mean_hd = float(np.mean(h_default))
    mean_hp = float(np.mean(h_prepay))
    implied_12m = 1.0 - (1.0 - mean_hd) ** 12
    implied_12m_competing = hazard.term_structure(
        np.full(12, mean_hd), np.full(12, mean_hp), horizon=12
    )["pd_12m"]

    realised = float((prepared["outcome"] == "default").cast(pl.Float64).to_numpy().mean())
    realised_12m = 1.0 - (1.0 - realised) ** 12

    out = {
        "mean_monthly_hazard_default": mean_hd,
        "mean_monthly_hazard_prepay": mean_hp,
        "hazard_implied_pd_12m_default_only": implied_12m,
        "hazard_implied_pd_12m_competing": implied_12m_competing,
        "realised_monthly_default_rate": realised,
        "realised_implied_pd_12m": realised_12m,
        "relative_difference": implied_12m / realised_12m - 1.0 if realised_12m else float("nan"),
    }
    log.info("hazard_reconciliation", **{k: round(v, 6) for k, v in out.items()})
    return out


def _write(
    fits: dict[str, hazard.HazardFit],
    reconciliation: dict[str, float],
    results: dict[str, Any],
    design: Any,
    prepared: pl.DataFrame,
) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    hazard.coefficient_table(fits).write_csv(ARTIFACTS / "hazard_coefficients.csv")
    hazard.baseline_table(fits).write_csv(ARTIFACTS / "hazard_baseline.csv")

    # Portfolio-average term structure over 25 years, the exhibit that shows why
    # the 12-month PD cannot simply be repeated.
    h_default = hazard.predict_hazard(results["default"], design)
    h_prepay = hazard.predict_hazard(results["prepaid"], design)
    months = 300
    path_d = np.full(months, float(np.mean(h_default)))
    path_p = np.full(months, float(np.mean(h_prepay)))
    cumulative, survival = hazard.survival_with_competing_risks(path_d, path_p)
    pl.DataFrame(
        {
            "month": np.arange(1, months + 1),
            "cumulative_default": cumulative,
            "survival": survival,
            "naive_chained": 1.0
            - (1.0 - reconciliation["hazard_implied_pd_12m_default_only"])
            ** (np.arange(1, months + 1) / 12.0),
        }
    ).write_csv(ARTIFACTS / "hazard_term_structure.csv")

    empirical = (
        prepared.group_by("age_band")
        .agg(
            n=pl.len(),
            empirical_default_hazard=(pl.col("outcome") == "default").cast(pl.Float64).mean(),
            empirical_prepay_hazard=(pl.col("outcome") == "prepaid").cast(pl.Float64).mean(),
        )
        .sort("age_band")
    )
    empirical.write_csv(ARTIFACTS / "hazard_empirical_seasoning.csv")
    (MODELS / "hazard_fit.json").write_text(
        json.dumps(
            {"fits": {k: asdict(v) for k, v in fits.items()}, "reconciliation": reconciliation},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    log.info("wrote_hazard_artifacts", path=str(ARTIFACTS))


def with_time_varying(frame: pl.DataFrame) -> pl.DataFrame:
    """Add the time-varying covariates to a panel frame.

    The same definitions as the risk-set query, so a loan scored from the panel
    sees exactly what the model was fitted on. Both HPI values are at or before
    the observation month.

    ROW ORDER IS PRESERVED, and that is load-bearing rather than cosmetic. The
    callers pair the returned frame against arrays built from the frame they
    passed in - exposure, LGD, remaining term, stage. DuckDB does not preserve
    input order across a join, so without the explicit ordering every loan is
    silently paired with another loan's covariates. See R-012.
    """
    con = duckdb.connect()
    try:
        con.execute(
            "CREATE TEMP VIEW hpi AS SELECT date, value "
            "FROM read_parquet('data/macro/SPCS20RSA.parquet') WHERE value IS NOT NULL"
        )
        # Prior-month delinquency state, taken from the risk set because a panel
        # frame is not monthly-contiguous and cannot lag over its own rows. The
        # definition must match the fitting query exactly: a covariate that means
        # "state at the start of the month" when fitted and "state now" when
        # scored is the contamination F-012 records, reintroduced at prediction
        # time. Risk-set observation dates are all first-of-month, so the join is
        # exact rather than approximate.
        con.execute(
            "CREATE TEMP VIEW dlq_prev AS "
            "SELECT loan_sequence_number, observation_date, "
            "       CASE WHEN current_loan_delinquency_status = 'RA' THEN 12 "
            "            ELSE COALESCE("
            "                TRY_CAST(current_loan_delinquency_status AS INTEGER), 0) "
            "       END AS months_delinquent "
            f"FROM read_parquet('{RISK_SET_GLOB}')"
        )
        con.register("panel_frame", frame.with_row_index("_row_order").to_arrow())
        enriched = (
            con.execute("""
            SELECT p.*,
                   p.current_actual_upb / NULLIF(p.original_upb, 0) AS amortisation_ratio,
                   COALESCE(d_prev.months_delinquent, 0) AS months_delinquent,
                   p.original_ltv
                       * (p.current_actual_upb / NULLIF(p.original_upb, 0))
                       * COALESCE(h_orig.value / NULLIF(h_now.value, 0), 1.0) AS mtm_ltv
            FROM panel_frame p
            LEFT JOIN dlq_prev d_prev
                   ON d_prev.loan_sequence_number = p.loan_sequence_number
                  AND d_prev.observation_date
                      = DATE_TRUNC('month', p.observation_date) - INTERVAL 1 MONTH
            LEFT JOIN hpi h_now
                   ON h_now.date = DATE_TRUNC('month', p.observation_date)
            LEFT JOIN hpi h_orig
                   ON h_orig.date = DATE_TRUNC(
                          'month', p.observation_date - INTERVAL (p.loan_age) MONTH)
            ORDER BY p._row_order
        """)
            .pl()
            .drop("_row_order")
        )
        require_same_observations(frame, enriched)
        return enriched
    finally:
        con.close()


def at_origination(frame: pl.DataFrame) -> pl.DataFrame:
    """Reset the time-varying covariates to their values when the loan was written.

    This is what makes the IFRS 9 SICR test meaningful. The comparator is not
    "the portfolio average at origination" but "THIS loan, as it looked on day
    one": no amortisation, no house-price movement, and current. Comparing a
    seasoned loan's lifetime PD against that is the question the standard
    actually asks.
    """
    return frame.with_columns(
        mtm_ltv=pl.col("original_ltv").cast(pl.Float64),
        amortisation_ratio=pl.lit(1.0),
        months_delinquent=pl.lit(0),
    )
