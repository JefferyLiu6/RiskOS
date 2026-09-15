"""Public, synthetic integration checks: no licensed panel or fitted bundle required.

Only the training-data loader and LGD estimator are supplied synthetic inputs.
The hazard fits, SQL enrichment, projection, staging, discounting, aggregation,
scenario calculation, and report rendering are production implementations.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from riskos.ecl import engine
from riskos.ecl import run as ecl_run
from riskos.models import train_hazard
from riskos.report import load_sources
from riskos.report.sections import executive_summary


@pytest.fixture
def synthetic_portfolio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> pl.DataFrame:
    monkeypatch.chdir(tmp_path)
    macro_dir = tmp_path / "data/macro"
    macro_dir.mkdir(parents=True)
    dates = pl.date_range(date(2000, 1, 1), date(2007, 1, 1), interval="1mo", eager=True)
    pl.DataFrame({"date": dates, "value": 100 + np.arange(len(dates)) / 4}).write_parquet(
        macro_dir / "SPCS20RSA.parquet"
    )
    frame = pl.DataFrame(
        {
            "loan_sequence_number": ["SYN_C", "SYN_A", "SYN_B"],
            "observation_date": [date(2006, 12, 1)] * 3,
            "loan_age": [3, 7, 11],
            "original_loan_term": [24] * 3,
            "remaining_months_to_legal_maturity": [21, 17, 13],
            "current_actual_upb": [100.0, 200.0, 300.0],
            "original_upb": [120.0, 250.0, 400.0],
            "original_ltv": [50.0, 70.0, 95.0],
            "original_dti": [25.0, 35.0, 45.0],
            "credit_score": [740.0, 680.0, 620.0],
            "number_of_borrowers": [1.0, 2.0, 1.0],
            "original_interest_rate": [3.0, 5.0, 7.0],
            "current_interest_rate": [3.0, 5.0, 7.0],
            "current_loan_delinquency_status": ["00", "01", "03"],
            "property_state": ["TX", "NY", "CA"],
        }
    )
    previous = (
        frame.select("loan_sequence_number", "current_loan_delinquency_status")
        .with_columns(observation_date=pl.lit(date(2006, 11, 1)))
        .reverse()
    )
    path = tmp_path / "prior_states.parquet"
    previous.write_parquet(path)
    monkeypatch.setattr(train_hazard, "RISK_SET_GLOB", str(path))
    rng = np.random.default_rng(412)
    n = 800
    training = pl.DataFrame(
        {
            "loan_age": rng.integers(1, 90, n),
            "credit_score": rng.uniform(550, 800, n),
            "original_ltv": rng.uniform(40, 110, n),
            "original_dti": rng.uniform(10, 60, n),
            "original_interest_rate": rng.uniform(2, 9, n),
            "original_upb": rng.uniform(80, 500, n),
            "number_of_borrowers": rng.integers(1, 3, n).astype(float),
            "mtm_ltv": rng.uniform(30, 120, n),
            "amortisation_ratio": rng.uniform(0.4, 1, n),
            "calendar_period": rng.integers(0, 32, n),
            "outcome": rng.choice(["default", "prepaid", "alive"], n, p=[0.05, 0.12, 0.83]),
        }
    )
    monkeypatch.setattr(train_hazard, "load_risk_set", lambda *_: training)
    segments = (
        frame.select(
            ecl_run.lgd_mod.ltv_band(pl.col("original_ltv")).alias("ltv_band"),
            "property_state",
        )
        .with_columns(shrunk_lgd=pl.Series([0.1, 0.4, 0.7]))
        .reverse()
    )
    usable = pl.DataFrame({"lgd": [0.1, 0.4, 0.7]})
    monkeypatch.setattr(ecl_run.lgd_mod, "estimate", lambda: (usable, segments, {}))
    return frame


def _losses(frame: pl.DataFrame) -> pl.DataFrame:
    pd12, life, orig = ecl_run.per_loan_pds(frame)
    stage = frame.select(engine.assign_stage(frame, life, orig, 1000))["ifrs9_stage"].to_numpy()
    return engine.compute(
        frame,
        pd12,
        life,
        ecl_run.attach_lgd(frame),
        stage,
        frame["remaining_months_to_legal_maturity"].to_numpy() / 12,
    )


def test_each_loan_keeps_its_pd_and_loss_when_reordered_or_scored_alone(
    synthetic_portfolio: pl.DataFrame,
) -> None:
    expected = _losses(synthetic_portfolio)
    assert expected["pd_applied"].n_unique() == 3, "a constant portfolio PD is not per-loan scoring"
    assert expected["ifrs9_stage"].to_list() == [1, 2, 3]
    np.testing.assert_allclose(expected["lgd"], [0.1, 0.4, 0.7])
    for subset in (synthetic_portfolio.reverse(), synthetic_portfolio[[1]]):
        actual = _losses(subset)
        for row in actual.to_dicts():
            original = expected.filter(
                pl.col("loan_sequence_number") == row["loan_sequence_number"]
            ).row(0, named=True)
            for field in ("pd_applied", "lgd", "discount_factor", "ecl"):
                assert row[field] == pytest.approx(original[field], rel=1e-10)


def test_projections_reject_a_reordered_enrichment_before_losses_are_computed(
    synthetic_portfolio: pl.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = train_hazard.with_time_varying
    monkeypatch.setattr(train_hazard, "with_time_varying", lambda frame: original(frame).reverse())
    with pytest.raises(ValueError, match="alignment changed"):
        ecl_run.per_loan_pds(synthetic_portfolio)


def test_duplicate_join_matches_fail_instead_of_duplicating_exposure(
    synthetic_portfolio: pl.DataFrame,
) -> None:
    path = Path(train_hazard.RISK_SET_GLOB)
    previous = pl.read_parquet(path)
    pl.concat([previous, previous.head(1)]).write_parquet(path)
    with pytest.raises(ValueError, match="alignment changed"):
        train_hazard.with_time_varying(synthetic_portfolio)


def test_neutral_scenario_matches_base_and_report_uses_the_computed_total(
    synthetic_portfolio: pl.DataFrame,
    tmp_path: Path,
) -> None:
    frame = synthetic_portfolio
    pd12, life, orig = ecl_run.per_loan_pds(frame)
    stage = frame.select(engine.assign_stage(frame, life, orig, 1000))["ifrs9_stage"].to_numpy()
    lgd = ecl_run.attach_lgd(frame)
    years = frame["remaining_months_to_legal_maturity"].to_numpy() / 12
    base = engine.compute(frame, pd12, life, lgd, stage, years)
    # Independent arithmetic: no production compute call constructs this oracle.
    applied = np.array([pd12[0], life[1], life[2]])
    horizons = np.array([min(1, years[0]), years[1], years[2]])
    expected = (
        applied
        * np.array([0.1, 0.4, 0.7])
        * np.array([100, 200, 300])
        / (1 + np.array([3, 5, 7]) / 100) ** (horizons / 2)
    )
    np.testing.assert_allclose(base["ecl"], expected)
    shifts = pl.DataFrame(
        {
            "scenario": ["neutral"],
            "weight": [1.0],
            "logit_shift": [0.0],
            "logit_shift_low": [0.0],
            "logit_shift_high": [0.0],
            "extrapolates": [False],
        }
    )
    scenario = engine.scenario_ecl(frame, shifts, pd12, life, stage, lgd, years)
    assert scenario["ecl"][0] == pytest.approx(expected.sum())
    assert [scenario[f"stage_{i}"][0] for i in (1, 2, 3)] == [1, 1, 1]
    sources = load_sources()
    sources.figures = tmp_path / "figures"
    sources.figures.mkdir()
    engine.summarise(base).write_csv(sources.figures / "ecl_by_stage.csv")
    scenario.write_csv(sources.figures / "ecl_scenarios.csv")
    report = executive_summary(sources)
    assert f"Portfolio ECL as at the last training date is ${expected.sum():,.0f}" in report


def test_ecl_cli_runs_the_synthetic_pipeline_and_writes_reconciled_outputs(
    synthetic_portfolio: pl.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typer.testing import CliRunner

    from riskos.cli import app

    monkeypatch.setattr(ecl_run, "load_portfolio", lambda _: synthetic_portfolio)
    t = np.arange(40)
    unemployment = 6 + np.sin(t / 3)
    hpi_yoy = 3 + np.cos(t / 4)
    series = pl.DataFrame(
        {
            "quarter": [date(2000 + i // 4, 1 + 3 * (i % 4), 1) for i in t],
            "unemployment": unemployment,
            "hpi_yoy": hpi_yoy,
            "default_rate": 1
            / (1 + np.exp(-(-5 + 0.12 * unemployment - 0.04 * hpi_yoy + 0.01 * np.sin(t)))),
        }
    )
    monkeypatch.setattr(ecl_run.macro, "quarterly_series", lambda *_: series)
    result = CliRunner().invoke(app, ["ecl"])
    assert result.exit_code == 0, result.exception
    stage = pl.read_csv("reports/figures/ecl_by_stage.csv")
    assert stage["n_loans"].sum() == synthetic_portfolio.height
    assert stage["ifrs9_stage"].to_list() == [1, 2, 3]
    assert stage["ead"].sum() == synthetic_portfolio["current_actual_upb"].sum()
    scenarios = pl.read_csv("reports/figures/ecl_scenarios.csv")
    expected_weighted = float((scenarios["weight"] * scenarios["ecl"]).sum())
    np.testing.assert_allclose(scenarios["probability_weighted_ecl"], expected_weighted)
    assert "probability-weighted" in result.stdout
