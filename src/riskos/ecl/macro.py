"""Phase 5 — the macroeconomic overlay (build plan §7.10).

Two-stage by design. The loan-level model carries idiosyncratic borrower and
loan risk; this stage supplies the cycle, as a shift in logit space::

    PD_scenario = sigmoid( logit(PD_base) + delta )

with ``delta`` from a portfolio-level regression of the logit of the observed
quarterly default rate on macro covariates.

**Do not invent elasticities. Fit them — and be honest about how weakly
identified the fit is.** Three constraints follow from that, all of them
consequences of having 24 quarterly observations. Not 32: the primary window is
requested as 1999Q1-2006Q4, but the house-price covariate is the year-on-year
change in a Case-Shiller index that begins in 2000, so the first defined
observation is 2001Q1 (F-019, MACRO_001).

* **At most two covariates**, unemployment and year-on-year HPI change. Three on
  24 serially-correlated points is not identified.
* **Newey-West standard errors**, four lags. Quarterly default rates are
  strongly autocorrelated; OLS standard errors would be far too narrow, and an
  interval that is too narrow is worse than no interval.
* **Confidence intervals propagated into scenario ECL as a range**, never a
  point estimate.

The material limitation, stated rather than buried: 1999-2006 contains little
default-rate variation, so the fitted sensitivity **extrapolates** into the
stress region. The severe-stress scenario at 10% unemployment lies well outside
anything the estimation sample contains.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import statsmodels.api as sm

from riskos.config import CONF_DIR, load_yaml
from riskos.log import get_logger

log = get_logger(__name__)

COVARIATES = ("unemployment", "hpi_yoy")


@dataclass
class MacroFit:
    """Fitted overlay with HAC inference."""

    window: str
    n_observations: int
    coefficients: dict[str, float]
    std_errors: dict[str, float]
    conf_int: dict[str, tuple[float, float]]
    r_squared: float
    newey_west_lags: int
    mean_logit_default_rate: float


def scenario_config() -> dict[str, object]:
    return load_yaml(CONF_DIR / "scenarios.yaml")


def logit(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    clipped = np.clip(p, 1e-9, 1.0 - 1e-9)
    return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=np.float64)


def sigmoid(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float64)


def quarterly_series(macro_dir: Path, panel_csv: Path) -> pl.DataFrame:
    """Join the observed quarterly default rate to macro covariates.

    HPI year-on-year change is computed from the national index, differenced at
    four quarters. Unemployment is the quarterly mean of the monthly series.
    """
    rates = pl.read_csv(panel_csv).select(
        pl.col("observation_date").str.to_date().alias("date"),
        pl.col("default_rate"),
        pl.col("n"),
    )
    unrate = (
        pl.read_parquet(macro_dir / "UNRATE.parquet")
        .with_columns(pl.col("date").dt.truncate("1q").alias("quarter"))
        .group_by("quarter")
        .agg(unemployment=pl.col("value").mean())
    )
    hpi = (
        pl.read_parquet(macro_dir / "SPCS20RSA.parquet")
        .with_columns(pl.col("date").dt.truncate("1q").alias("quarter"))
        .group_by("quarter")
        .agg(hpi=pl.col("value").mean())
        .sort("quarter")
        .with_columns(hpi_yoy=100.0 * (pl.col("hpi") / pl.col("hpi").shift(4) - 1.0))
    )
    return (
        rates.with_columns(pl.col("date").dt.truncate("1q").alias("quarter"))
        .join(unrate, on="quarter")
        .join(hpi, on="quarter")
        .drop_nulls(["unemployment", "hpi_yoy", "default_rate"])
        .filter(pl.col("default_rate") > 0)
        .sort("quarter")
    )


def fit(series: pl.DataFrame, window: str, lags: int = 4, alpha: float = 0.05) -> MacroFit:
    """OLS of logit(default rate) on macro covariates, with HAC inference."""
    y = logit(series["default_rate"].to_numpy())
    x = sm.add_constant(series.select(COVARIATES).to_pandas().astype(float))
    result = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    conf = result.conf_int(alpha=alpha)
    out = MacroFit(
        window=window,
        n_observations=int(series.height),
        coefficients={k: float(v) for k, v in result.params.items()},
        std_errors={k: float(v) for k, v in result.bse.items()},
        conf_int={k: (float(conf.loc[k, 0]), float(conf.loc[k, 1])) for k in result.params.index},
        r_squared=float(result.rsquared),
        newey_west_lags=lags,
        mean_logit_default_rate=float(np.mean(y)),
    )
    log.info(
        "macro_fitted",
        window=window,
        n=out.n_observations,
        r_squared=round(out.r_squared, 4),
        unemployment=round(out.coefficients.get("unemployment", 0.0), 4),
        hpi_yoy=round(out.coefficients.get("hpi_yoy", 0.0), 4),
    )
    return out


def delta(
    fit_result: MacroFit,
    unemployment: float,
    hpi_yoy: float,
    sample_means: tuple[float, float],
    bound: str = "point",
) -> float:
    """Logit shift for a scenario, relative to the estimation-sample mean.

    Expressed as a deviation from the sample mean so a scenario at average
    conditions produces no shift and the base PD is not double-counted. The
    means are passed in rather than inferred: reading them from the coefficient
    dictionary would silently yield zero if the key were absent, which is a
    quiet way to disable the entire overlay.
    """
    which = {"point": 0, "low": 1, "high": 2}[bound]
    mean_u, mean_h = sample_means

    def coefficient(name: str) -> float:
        if which == 0:
            return fit_result.coefficients[name]
        lo, hi = fit_result.conf_int[name]
        return lo if which == 1 else hi

    return coefficient("unemployment") * (unemployment - mean_u) + coefficient("hpi_yoy") * (
        hpi_yoy - mean_h
    )


def scenario_shifts(fit_result: MacroFit, series: pl.DataFrame) -> pl.DataFrame:
    """Logit shift per scenario, with a band from the HAC intervals."""
    mean_u = float(series["unemployment"].to_numpy().mean())
    mean_h = float(series["hpi_yoy"].to_numpy().mean())
    rows = []
    scenarios: list[dict[str, Any]] = scenario_config()["scenarios"]  # type: ignore[assignment]
    for scenario in scenarios:
        du = scenario["unemployment"] - mean_u
        dh = scenario["hpi_yoy"] - mean_h
        point = (
            fit_result.coefficients["unemployment"] * du + fit_result.coefficients["hpi_yoy"] * dh
        )
        # The band takes each coefficient to the endpoint that moves the shift
        # in the same direction, which is the conservative reading.
        lo_u, hi_u = fit_result.conf_int["unemployment"]
        lo_h, hi_h = fit_result.conf_int["hpi_yoy"]
        low = min(lo_u * du, hi_u * du) + min(lo_h * dh, hi_h * dh)
        high = max(lo_u * du, hi_u * du) + max(lo_h * dh, hi_h * dh)
        rows.append(
            {
                "scenario": scenario["name"],
                "weight": scenario["weight"],
                "unemployment": scenario["unemployment"],
                "hpi_yoy": scenario["hpi_yoy"],
                "delta_unemployment": du,
                "delta_hpi_yoy": dh,
                "logit_shift": point,
                "logit_shift_low": low,
                "logit_shift_high": high,
                "pd_multiplier": float(np.exp(point)),
                "extrapolates": bool(
                    scenario["unemployment"] > series["unemployment"].max()
                    or scenario["hpi_yoy"] < series["hpi_yoy"].min()
                ),
            }
        )
    return pl.DataFrame(rows)


def apply_shift(pd_base: npt.NDArray[np.float64], logit_shift: float) -> npt.NDArray[np.float64]:
    """PD_scenario = sigmoid(logit(PD_base) + delta)."""
    return sigmoid(logit(pd_base) + logit_shift)


def backtest(fit_result: MacroFit, series: pl.DataFrame, window: tuple[str, str]) -> pl.DataFrame:
    """Feed realised crisis macro into the pre-2007 model, compare to realised.

    Reported whether or not it flatters the model. A stress model nobody
    backtested is exactly what E-23 exists to prevent.
    """
    start_year, end_year = int(window[0][:4]), int(window[1][:4])
    actual = series.filter(pl.col("quarter").dt.year().is_between(start_year, end_year))
    mean_u = float(series["unemployment"].to_numpy().mean())
    mean_h = float(series["hpi_yoy"].to_numpy().mean())
    base_logit = fit_result.mean_logit_default_rate

    rows = []
    for row in actual.to_dicts():
        shift = fit_result.coefficients["unemployment"] * (row["unemployment"] - mean_u) + (
            fit_result.coefficients["hpi_yoy"] * (row["hpi_yoy"] - mean_h)
        )
        predicted = float(sigmoid(np.array([base_logit + shift]))[0])
        rows.append(
            {
                "quarter": row["quarter"],
                "unemployment": row["unemployment"],
                "hpi_yoy": row["hpi_yoy"],
                "predicted_default_rate": predicted,
                "observed_default_rate": row["default_rate"],
                "ratio_observed_to_predicted": row["default_rate"] / predicted,
            }
        )
    return pl.DataFrame(rows)


def coefficient_table(fits: dict[str, MacroFit]) -> pl.DataFrame:
    """Coefficients with HAC intervals for every estimation window."""
    rows = []
    for label, fitted in fits.items():
        for name, beta in fitted.coefficients.items():
            low, high = fitted.conf_int[name]
            rows.append(
                {
                    "window": label,
                    "n_observations": fitted.n_observations,
                    "term": name,
                    "coefficient": beta,
                    "hac_std_error": fitted.std_errors[name],
                    "ci_low": low,
                    "ci_high": high,
                    "significant_at_95": not (low <= 0.0 <= high),
                    "r_squared": fitted.r_squared,
                }
            )
    return pl.DataFrame(rows)
