"""Phase 5 orchestration: LGD, EAD, staging, ECL, scenarios, macro backtest.

Makes `riskos ecl` reachable. Previously the modules existed but nothing wired
them together, so `make all` could not complete and the phase was not runnable
end to end regardless of what the modules could do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl

from riskos.ecl import engine, macro
from riskos.ecl import lgd as lgd_mod
from riskos.log import get_logger
from riskos.panel.alignment import require_same_observations

log = get_logger(__name__)

ARTIFACTS = Path("reports/figures")
AS_AT = "2006-12-01"  # last observation date in the training window
SICR_GRID = (1.5, 2.0, 2.5, 3.0, 4.0)


def load_portfolio(as_at: str = AS_AT) -> pl.DataFrame:
    con = duckdb.connect()
    try:
        return con.execute(f"""
            SELECT * FROM read_parquet('data/panel/panel/*/*.parquet')
            WHERE split = 'train' AND observation_date = DATE '{as_at}'
        """).pl()
    finally:
        con.close()


def attach_lgd(portfolio: pl.DataFrame) -> np.ndarray:
    """Segment LGD by LTV band and state, falling back to the portfolio mean."""
    usable, segments, _ = lgd_mod.estimate()
    joined = portfolio.with_columns(
        lgd_mod.ltv_band(pl.col("original_ltv")).alias("ltv_band")
    ).join(
        segments.select(["ltv_band", "property_state", "shrunk_lgd"]),
        on=["ltv_band", "property_state"],
        how="left",
        validate="m:1",
        maintain_order="left",
    )
    require_same_observations(portfolio, joined)
    fallback = float(usable["lgd"].to_numpy().mean())
    return joined["shrunk_lgd"].fill_null(fallback).to_numpy()


def per_loan_pds(
    portfolio: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-loan 12-month, lifetime, and origination-vintage PD (closes F-010).

    Each loan is projected through the fitted hazards using its own covariates
    and its own age path, truncated at its own remaining term. The
    origination-vintage PD projects the SAME loan from age zero over its
    original term, which is an illustrative comparator for the SICR sensitivity:
    "has this loan's lifetime risk increased significantly since ITS
    origination", not since the portfolio average.

    The previous implementation applied one portfolio-average PD to every loan,
    which made the SICR ratio a constant and the staging sensitivity vacuous.
    """
    from riskos.models import hazard
    from riskos.models.train_hazard import (
        COVARIATES,
        _sample_size,
        at_origination,
        load_risk_set,
        with_time_varying,
    )

    risk_set = load_risk_set(_sample_size())
    design, prepared, scaling = hazard.design_matrix(risk_set, COVARIATES)
    kept = hazard.estimable_columns(design)
    design = design[kept]
    clusters = prepared["calendar_period"].to_numpy()
    results = {}
    for cause in hazard.CAUSES:
        events = (prepared["outcome"] == cause).cast(pl.Int64).to_numpy()
        results[cause], _ = hazard.fit_cause(design, events, clusters, cause)

    # The panel needs the same time-varying covariates the model was fitted on.
    enriched = with_time_varying(portfolio)
    require_same_observations(portfolio, enriched)
    portfolio = enriched
    origination_view = at_origination(portfolio)

    remaining = portfolio["remaining_months_to_legal_maturity"].to_numpy().astype(np.int64)
    original_term = portfolio["original_loan_term"].to_numpy().astype(np.int64)
    ages = portfolio["loan_age"].to_numpy().astype(np.int64)
    horizon = int(max(remaining.max(), original_term.max()))

    pd_12m, _ = hazard.project_paths(
        results,
        portfolio,
        COVARIATES,
        scaling,
        kept,
        12,
        start_age=ages,
        term_months=np.minimum(remaining, 12),
    )
    pd_life, _ = hazard.project_paths(
        results,
        portfolio,
        COVARIATES,
        scaling,
        kept,
        horizon,
        start_age=ages,
        term_months=remaining,
    )
    pd_orig, _ = hazard.project_paths(
        results,
        origination_view,
        COVARIATES,
        scaling,
        kept,
        horizon,
        start_age=np.zeros_like(ages),
        term_months=original_term,
    )
    log.info(
        "per_loan_pds",
        mean_pd_12m=round(float(pd_12m.mean()), 6),
        mean_pd_lifetime=round(float(pd_life.mean()), 6),
        mean_pd_origination=round(float(pd_orig.mean()), 6),
        sicr_ratio_p50=round(float(np.median(pd_life / np.maximum(pd_orig, 1e-9))), 4),
    )
    return pd_12m, pd_life, pd_orig


def run(as_at: str = AS_AT) -> dict[str, Any]:
    """Base ECL, staging sensitivity, scenario ECL, and the macro backtest."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    portfolio = load_portfolio(as_at)
    lgd_values = attach_lgd(portfolio)

    pd_12m, pd_life, pd_orig = per_loan_pds(portfolio)
    remaining_years = portfolio["remaining_months_to_legal_maturity"].to_numpy() / 12.0
    threshold = engine.sicr_threshold()

    stage = portfolio.select(engine.assign_stage(portfolio, pd_life, pd_orig, threshold))[
        "ifrs9_stage"
    ].to_numpy()
    result = engine.compute(portfolio, pd_12m, pd_life, lgd_values, stage, remaining_years)
    summary = engine.summarise(result)
    summary.write_csv(ARTIFACTS / "ecl_by_stage.csv")
    engine.staging_sensitivity(portfolio, pd_life, pd_orig, SICR_GRID).write_csv(
        ARTIFACTS / "ecl_staging_sensitivity.csv"
    )

    series = macro.quarterly_series(Path("data/macro"), ARTIFACTS / "default_rate_by_quarter.csv")
    pre = series.filter(pl.col("quarter").dt.year() <= 2006)
    fit_pre, fit_full = macro.fit(pre, "1999-2006"), macro.fit(series, "1999-2019")
    macro.coefficient_table({"1999-2006": fit_pre, "1999-2019": fit_full}).write_csv(
        ARTIFACTS / "macro_coefficients.csv"
    )
    shifts = macro.scenario_shifts(fit_pre, pre)
    shifts.write_csv(ARTIFACTS / "macro_scenario_shifts.csv")
    macro.backtest(fit_pre, series, ("2008Q1", "2009Q4")).write_csv(
        ARTIFACTS / "macro_backtest.csv"
    )

    scenarios = engine.scenario_ecl(
        portfolio, shifts, pd_12m, pd_life, stage, lgd_values, remaining_years
    )
    scenarios.write_csv(ARTIFACTS / "ecl_scenarios.csv")

    total_ecl = float(result["ecl"].sum() or 0.0)
    total_ead = float(portfolio["current_actual_upb"].sum() or 0.0)
    report = {
        "as_at": as_at,
        "n_loans": portfolio.height,
        "total_ecl": total_ecl,
        "total_ead": total_ead,
        "coverage": total_ecl / total_ead if total_ead else float("nan"),
        "probability_weighted_ecl": float(scenarios["probability_weighted_ecl"][0]),
    }
    log.info("ecl_complete", **{k: v for k, v in report.items() if k != "as_at"})
    return report
