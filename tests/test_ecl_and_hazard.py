"""Phase 5: ECL arithmetic, staging order, competing risks, and macro mechanics.

Written after a code review found that the findings register claimed the ECL
engine was unit-tested when nothing imported it. These are the tests that claim
should have referred to.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from riskos.ecl import engine, macro
from riskos.models import hazard


@pytest.fixture
def fixture_book() -> pl.DataFrame:
    """Four loans spanning each staging route. Not real loan data."""
    return pl.DataFrame(
        {
            "loan_sequence_number": ["A", "B", "C", "D"],
            "observation_date": ["2006-12-01"] * 4,
            "current_loan_delinquency_status": ["00", "01", "03", "00"],
            "current_actual_upb": [100_000.0, 200_000.0, 150_000.0, 50_000.0],
            "current_interest_rate": [6.0, 6.0, 6.0, 6.0],
        }
    )


# --------------------------------------------------------------------------
# Staging — order of precedence is the part that must not drift.
# --------------------------------------------------------------------------


def test_stage_3_wins_over_everything(fixture_book: pl.DataFrame) -> None:
    """90+ DPD is credit-impaired regardless of the SICR ratio."""
    flat = np.array([0.01, 0.01, 0.01, 0.01])
    stage = fixture_book.select(engine.assign_stage(fixture_book, flat, flat, 2.0))[
        "ifrs9_stage"
    ].to_numpy()

    assert stage[2] == 3  # status "03"


def test_the_30_dpd_backstop_overrides_a_passing_sicr_test(
    fixture_book: pl.DataFrame,
) -> None:
    """A delinquent loan cannot sit in stage 1 however good its ratio."""
    flat = np.array([0.01, 0.01, 0.01, 0.01])
    stage = fixture_book.select(engine.assign_stage(fixture_book, flat, flat, 2.0))[
        "ifrs9_stage"
    ].to_numpy()

    assert stage[1] == 2  # status "01", ratio 1.0, still stage 2
    assert stage[0] == 1  # current, ratio 1.0, stage 1


def test_sicr_ratio_moves_a_current_loan_to_stage_2(fixture_book: pl.DataFrame) -> None:
    lifetime = np.array([0.05, 0.01, 0.01, 0.01])
    origination = np.array([0.01, 0.01, 0.01, 0.01])

    stage = fixture_book.select(engine.assign_stage(fixture_book, lifetime, origination, 2.0))[
        "ifrs9_stage"
    ].to_numpy()

    assert stage[0] == 2  # ratio 5.0 > 2.0
    assert stage[3] == 1  # ratio 1.0


# --------------------------------------------------------------------------
# ECL arithmetic.
# --------------------------------------------------------------------------


def test_ecl_is_pd_times_lgd_times_ead_times_discount(fixture_book: pl.DataFrame) -> None:
    """The core identity, checked against a hand-computed value."""
    pd_12m = np.array([0.02, 0.02, 0.02, 0.02])
    lgd = np.array([0.5, 0.5, 0.5, 0.5])
    stage = np.array([1, 1, 1, 1])
    years = np.array([1.0, 1.0, 1.0, 1.0])

    result = engine.compute(fixture_book, pd_12m, pd_12m, lgd, stage, years)

    # Loan A: 0.02 * 0.5 * 100,000 = 1,000 before discounting at 6% for half a
    # year: 1000 / 1.06**0.5 = 971.29
    expected = 0.02 * 0.5 * 100_000 / (1.06**0.5)
    assert float(result["ecl"][0]) == pytest.approx(expected, rel=1e-9)


def test_stage_1_uses_the_12_month_pd_and_stages_2_3_use_lifetime(
    fixture_book: pl.DataFrame,
) -> None:
    pd_12m = np.full(4, 0.01)
    pd_life = np.full(4, 0.09)
    stage = np.array([1, 2, 3, 1])

    result = engine.compute(fixture_book, pd_12m, pd_life, np.full(4, 0.5), stage, np.full(4, 5.0))

    assert result["pd_applied"].to_list() == [0.01, 0.09, 0.09, 0.01]


def test_discounting_reduces_ecl_and_is_stronger_over_a_longer_horizon() -> None:
    rate = np.array([6.0, 6.0])
    near = engine.discount_factors(rate, np.array([0.5, 0.5]))
    far = engine.discount_factors(rate, np.array([10.0, 10.0]))

    assert (near < 1.0).all()
    assert (far < near).all()


def test_summarise_reconciles_to_the_total(fixture_book: pl.DataFrame) -> None:
    """ECL must reconcile across stages — a Phase 5 acceptance criterion."""
    result = engine.compute(
        fixture_book,
        np.full(4, 0.02),
        np.full(4, 0.05),
        np.full(4, 0.4),
        np.array([1, 2, 3, 1]),
        np.full(4, 3.0),
    )

    summary = engine.summarise(result)

    assert float(summary["ecl"].sum()) == pytest.approx(float(result["ecl"].sum()))
    assert float(summary["share_of_ecl"].sum()) == pytest.approx(1.0)
    assert int(summary["n_loans"].sum()) == fixture_book.height


# --------------------------------------------------------------------------
# Competing risks — the accounting must close.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("hd", "hp"), [(0.001, 0.02), (0.01, 0.05), (0.05, 0.05)])
def test_competing_risk_probabilities_sum_to_one(hd: float, hp: float) -> None:
    """Default + prepay + survive == 1, exactly.

    An implementation whose outcome probabilities do not close is silently
    creating or destroying loans.
    """
    accounting = hazard.competing_risk_accounting(np.full(200, hd), np.full(200, hp))

    assert accounting["total"] == pytest.approx(1.0, abs=1e-12)


def test_prepayment_reduces_cumulative_default() -> None:
    """The whole point of treating prepayment as a competing risk."""
    hd = np.full(120, 0.002)
    with_prepay, _ = hazard.survival_with_competing_risks(hd, np.full(120, 0.02))
    without, _ = hazard.survival_with_competing_risks(hd, np.zeros(120))

    assert with_prepay[-1] < without[-1]
    assert without[-1] == pytest.approx(1.0 - (1.0 - 0.002) ** 120, rel=1e-9)


def test_the_two_conventions_differ_only_slightly_and_are_both_closed() -> None:
    """The convention is a stated modelling choice, not an accident."""
    hd, hp = np.full(200, 0.01), np.full(200, 0.05)
    first, _ = hazard.survival_with_competing_risks(hd, hp, convention="default_first")
    symmetric, _ = hazard.survival_with_competing_risks(hd, hp, convention="symmetric")

    assert first[-1] > symmetric[-1]  # default_first attributes the joint mass to default
    assert abs(first[-1] / symmetric[-1] - 1.0) < 0.02


def test_unknown_convention_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown competing-risk convention"):
        hazard.survival_with_competing_risks(np.full(3, 0.01), np.full(3, 0.01), convention="x")


def test_survival_is_monotone_non_increasing() -> None:
    _, survival = hazard.survival_with_competing_risks(np.full(60, 0.003), np.full(60, 0.015))

    assert (np.diff(survival) <= 0).all()


# --------------------------------------------------------------------------
# Hazard design matrix — the batch-independence property (review finding).
# --------------------------------------------------------------------------


def _frame(ages: list[int], scores: list[int | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {"loan_age": ages, "credit_score": scores},
        schema={"loan_age": pl.Int64, "credit_score": pl.Int64},
    )


def test_imputation_uses_the_fitting_median_not_the_batch_median() -> None:
    """A missing value must mean the same thing however it is scored.

    Before the fix, a row with a missing covariate imputed to NaN when scored
    alone — silently poisoning the predicted hazard — and to the batch median
    otherwise.
    """
    fit_frame = _frame([24, 24, 24], [None, 700, 800])
    _, _, scaling = hazard.design_matrix(fit_frame, ["credit_score"])

    alone, _, _ = hazard.design_matrix(_frame([24], [None]), ["credit_score"], scaling)
    batched, _, _ = hazard.design_matrix(fit_frame, ["credit_score"], scaling)

    assert np.isfinite(alone["credit_score"].iloc[0])
    assert alone["credit_score"].iloc[0] == pytest.approx(batched["credit_score"].iloc[0])


def test_design_columns_are_stable_when_a_band_is_absent() -> None:
    """A sample missing an age band must not produce a narrower matrix."""
    wide, _, scaling = hazard.design_matrix(_frame([3, 30, 200], [700, 700, 700]), ["credit_score"])
    narrow, _, _ = hazard.design_matrix(
        _frame([30], [700]), ["credit_score"], scaling, list(wide.columns)
    )

    assert list(narrow.columns) == list(wide.columns)


def _dlq_frame(ages: list[int], delinquent: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {"loan_age": ages, "credit_score": [700] * len(ages), "months_delinquent": delinquent},
        schema={"loan_age": pl.Int64, "credit_score": pl.Int64, "months_delinquent": pl.Int64},
    )


# --------------------------------------------------------------------------
# Delinquency as a STATE (finding F-012). A single linear term cannot span a
# hazard range of roughly 19,000x, so each state carries its own level.
# --------------------------------------------------------------------------


def test_delinquency_bands_partition_the_state_space() -> None:
    """Every state maps to exactly one band, and 90+ DPD collapses into the top.

    The top band is open-ended because reaching 90 days past due IS the default
    event: a loan at 3+ months delinquent has already left the risk set, so
    finer bands above it would be inestimable.
    """
    banded = pl.DataFrame({"months_delinquent": [0, 1, 2, 3, 12]}).with_columns(
        hazard.delinquency_band(pl.col("months_delinquent")).alias("band")
    )

    assert banded["band"].to_list() == [
        "dlq_00",
        "dlq_01",
        "dlq_02_plus",
        "dlq_02_plus",
        "dlq_02_plus",
    ]
    assert hazard.delinquency_band_labels() == ["dlq_00", "dlq_01", "dlq_02_plus"]


def test_the_current_state_is_the_reference_level_and_gets_no_column() -> None:
    """Including a dummy for every state would make the design singular."""
    design, _, _ = hazard.design_matrix(
        _dlq_frame([24, 24, 24], [0, 1, 2]), ["credit_score"], delinquency_bands=True
    )

    assert "dlq_00" not in design.columns
    assert {"dlq_01", "dlq_02_plus"} <= set(design.columns)


def test_delinquency_dummies_are_mutually_exclusive() -> None:
    """A loan is in exactly one state, so the dummies never both fire."""
    design, _, _ = hazard.design_matrix(
        _dlq_frame([24] * 4, [0, 1, 2, 9]), ["credit_score"], delinquency_bands=True
    )
    fired = design[["dlq_01", "dlq_02_plus"]].sum(axis=1).to_numpy()

    assert set(np.unique(fired)) <= {0.0, 1.0}
    assert fired.tolist() == [0.0, 1.0, 1.0, 1.0]


def test_the_state_is_absent_from_the_design_unless_it_is_asked_for() -> None:
    """The primary specification must not pick up delinquency by accident."""
    design, _, _ = hazard.design_matrix(_dlq_frame([24], [1]), ["credit_score"])

    assert not [c for c in design.columns if c.startswith("dlq_")]


def test_design_columns_are_stable_when_a_delinquency_state_is_absent() -> None:
    """A batch of only-current loans must not produce a narrower matrix.

    Same failure mode as the age-band case: a narrower matrix at prediction time
    silently misaligns coefficients with columns.
    """
    wide, _, scaling = hazard.design_matrix(
        _dlq_frame([3, 30, 200], [0, 1, 2]), ["credit_score"], delinquency_bands=True
    )
    narrow, _, _ = hazard.design_matrix(
        _dlq_frame([30], [0]),
        ["credit_score"],
        scaling,
        list(wide.columns),
        delinquency_bands=True,
    )

    assert list(narrow.columns) == list(wide.columns)
    assert narrow[["dlq_01", "dlq_02_plus"]].to_numpy().sum() == 0.0


# --------------------------------------------------------------------------
# Macro overlay.
# --------------------------------------------------------------------------


def test_logit_and_sigmoid_round_trip() -> None:
    p = np.array([0.001, 0.05, 0.5, 0.9])

    np.testing.assert_allclose(macro.sigmoid(macro.logit(p)), p, rtol=1e-9)


def test_a_zero_shift_leaves_pd_unchanged() -> None:
    p = np.array([0.005, 0.02, 0.10])

    np.testing.assert_allclose(macro.apply_shift(p, 0.0), p, rtol=1e-9)


def test_a_positive_shift_raises_pd_and_stays_in_the_unit_interval() -> None:
    p = np.array([0.001, 0.05, 0.5, 0.95])

    shifted = macro.apply_shift(p, 1.5)

    assert (shifted > p).all()
    assert (shifted < 1.0).all()


def test_scenario_weights_sum_to_one() -> None:
    scenarios = macro.scenario_config()["scenarios"]

    assert sum(s["weight"] for s in scenarios) == pytest.approx(1.0)  # type: ignore[union-attr]


# --------------------------------------------------------------------------
# Per-loan projection (F-010) — the path that makes ECL loan-sensitive.
# --------------------------------------------------------------------------


class _StubResult:
    """Minimal stand-in for a fitted statsmodels GLM result."""

    def __init__(self, params: dict[str, float]) -> None:
        self.params = params


def _stub_hazards(default_level: float, band_bump: float = 0.0) -> dict[str, _StubResult]:
    """Two causes with known coefficients, so the projection is checkable."""
    import math

    intercept = math.log(-math.log(1.0 - default_level))  # cloglog inverse of the level
    default_params = {"intercept": intercept, "credit_score": 0.0}
    prepay_params = {"intercept": -30.0, "credit_score": 0.0}  # effectively no prepayment
    for band in hazard.age_band_labels()[1:]:
        default_params[band] = band_bump
        prepay_params[band] = 0.0
    return {"default": _StubResult(default_params), "prepaid": _StubResult(prepay_params)}


def _projection_frame(n: int, age: int = 24) -> pl.DataFrame:
    return pl.DataFrame(
        {"loan_age": [age] * n, "credit_score": [700] * n},
        schema={"loan_age": pl.Int64, "credit_score": pl.Int64},
    )


def test_cloglog_inverse_matches_its_definition() -> None:
    eta = np.array([-5.0, -2.0, 0.0])

    np.testing.assert_allclose(hazard.cloglog_inverse(eta), 1.0 - np.exp(-np.exp(eta)), rtol=1e-12)


def test_projection_reproduces_a_constant_hazard_analytically() -> None:
    """With a flat baseline and no prepayment, cumulative default is closed-form."""
    frame = _projection_frame(3)
    _, _, scaling = hazard.design_matrix(frame, ["credit_score"])
    design, _, _ = hazard.design_matrix(frame, ["credit_score"], scaling)

    cumulative, _ = hazard.project_paths(
        _stub_hazards(0.01), frame, ["credit_score"], scaling, list(design.columns), 12
    )

    assert cumulative[0] == pytest.approx(1.0 - (1.0 - 0.01) ** 12, rel=1e-6)


def test_each_loan_is_truncated_at_its_own_remaining_term() -> None:
    """A loan with 6 months left must stop accruing at month 6."""
    frame = _projection_frame(2)
    _, _, scaling = hazard.design_matrix(frame, ["credit_score"])
    design, _, _ = hazard.design_matrix(frame, ["credit_score"], scaling)

    cumulative, _ = hazard.project_paths(
        _stub_hazards(0.01),
        frame,
        ["credit_score"],
        scaling,
        list(design.columns),
        12,
        term_months=np.array([6, 12], dtype=np.int64),
    )

    assert cumulative[0] == pytest.approx(1.0 - (1.0 - 0.01) ** 6, rel=1e-6)
    assert cumulative[1] == pytest.approx(1.0 - (1.0 - 0.01) ** 12, rel=1e-6)
    assert cumulative[0] < cumulative[1]


def test_the_start_age_changes_the_path_through_the_baseline() -> None:
    """The origination-vintage PD projects the same loan from age zero."""
    frame = _projection_frame(2, age=100)
    _, _, scaling = hazard.design_matrix(frame, ["credit_score"])
    design, _, _ = hazard.design_matrix(frame, ["credit_score"], scaling)
    stub = _stub_hazards(0.001, band_bump=1.0)  # later bands are riskier

    now, _ = hazard.project_paths(
        stub,
        frame,
        ["credit_score"],
        scaling,
        list(design.columns),
        36,
        start_age=np.array([100, 100], dtype=np.int64),
    )
    at_origination, _ = hazard.project_paths(
        stub,
        frame,
        ["credit_score"],
        scaling,
        list(design.columns),
        36,
        start_age=np.zeros(2, dtype=np.int64),
    )

    # From age zero the loan spends its first months in the low-risk reference
    # band, so its projected default is lower than from age 100.
    assert at_origination[0] < now[0]


def test_projection_is_deterministic_and_batch_independent() -> None:
    frame = _projection_frame(5)
    _, _, scaling = hazard.design_matrix(frame, ["credit_score"])
    design, _, _ = hazard.design_matrix(frame, ["credit_score"], scaling)
    stub = _stub_hazards(0.005)

    full, _ = hazard.project_paths(stub, frame, ["credit_score"], scaling, list(design.columns), 24)
    solo, _ = hazard.project_paths(
        stub, frame.head(1), ["credit_score"], scaling, list(design.columns), 24
    )

    assert solo[0] == pytest.approx(full[0])


def test_survival_returned_by_the_projection_is_a_probability() -> None:
    frame = _projection_frame(4)
    _, _, scaling = hazard.design_matrix(frame, ["credit_score"])
    design, _, _ = hazard.design_matrix(frame, ["credit_score"], scaling)

    cumulative, survival = hazard.project_paths(
        _stub_hazards(0.02), frame, ["credit_score"], scaling, list(design.columns), 60
    )

    assert ((survival >= 0.0) & (survival <= 1.0)).all()
    assert ((cumulative >= 0.0) & (cumulative <= 1.0)).all()


# --------------------------------------------------------------------------
# Macro overlay mechanics on synthetic series.
# --------------------------------------------------------------------------


@pytest.fixture
def fixture_macro_series() -> pl.DataFrame:
    """A series where the default rate responds to both covariates by design.

    The two covariates must not be collinear or the design is rank-deficient
    and no coefficient is identified — which is exactly the failure mode the
    §7.10 two-covariate limit exists to avoid, and worth reproducing correctly
    in a fixture rather than by accident.

    Residuals are AR(1) so the Newey-West correction has genuine
    autocorrelation to correct for.
    """
    import datetime as dt

    rng = np.random.default_rng(20250901)
    n = 60
    unemployment = 6.0 + 2.0 * np.sin(np.arange(n) / 5.0) + rng.normal(0, 0.3, n)
    hpi = 2.0 + 8.0 * np.cos(np.arange(n) / 3.0) + rng.normal(0, 0.5, n)

    innovations = rng.normal(0, 0.05, n)
    residual = np.zeros(n)
    for i in range(1, n):
        residual[i] = 0.7 * residual[i - 1] + innovations[i]

    logit = -5.0 + 0.12 * unemployment - 0.04 * hpi + residual
    return pl.DataFrame(
        {
            "quarter": [dt.date(1999 + i // 4, 1 + 3 * (i % 4), 1) for i in range(n)],
            "unemployment": unemployment,
            "hpi_yoy": hpi,
            "default_rate": 1.0 / (1.0 + np.exp(-logit)),
        }
    )


def test_macro_fit_recovers_known_coefficients(fixture_macro_series: pl.DataFrame) -> None:
    fitted = macro.fit(fixture_macro_series, "synthetic")

    assert fitted.coefficients["unemployment"] == pytest.approx(0.12, abs=0.05)
    assert fitted.coefficients["hpi_yoy"] == pytest.approx(-0.04, abs=0.02)
    assert fitted.n_observations == 60


def test_hac_intervals_bracket_the_point_estimates(
    fixture_macro_series: pl.DataFrame,
) -> None:
    fitted = macro.fit(fixture_macro_series, "synthetic")

    for term, beta in fitted.coefficients.items():
        low, high = fitted.conf_int[term]
        assert low <= beta <= high


def test_hac_widens_intervals_when_residuals_are_autocorrelated(
    fixture_macro_series: pl.DataFrame,
) -> None:
    """The reason §7.10 mandates Newey-West.

    The fixture has AR(1) residuals. Ignoring that autocorrelation produces
    intervals that are too narrow, and an interval that is too narrow is worse
    than no interval at all.
    """
    naive = macro.fit(fixture_macro_series, "s", lags=0)
    corrected = macro.fit(fixture_macro_series, "s", lags=4)

    naive_width = naive.conf_int["unemployment"][1] - naive.conf_int["unemployment"][0]
    corrected_width = corrected.conf_int["unemployment"][1] - corrected.conf_int["unemployment"][0]
    assert corrected_width > naive_width
    assert corrected.newey_west_lags == 4


def test_scenario_shifts_flag_extrapolation_beyond_the_sample(
    fixture_macro_series: pl.DataFrame,
) -> None:
    """The severe scenario lies outside anything the sample contains."""
    shifts = macro.scenario_shifts(macro.fit(fixture_macro_series, "s"), fixture_macro_series)

    severe = shifts.filter(pl.col("scenario") == "severe_stress").to_dicts()[0]
    assert severe["extrapolates"] is True
    assert severe["logit_shift_low"] <= severe["logit_shift"] <= severe["logit_shift_high"]


def test_a_worse_scenario_produces_a_larger_upward_shift(
    fixture_macro_series: pl.DataFrame,
) -> None:
    shifts = macro.scenario_shifts(macro.fit(fixture_macro_series, "s"), fixture_macro_series)
    by_name = {r["scenario"]: r["logit_shift"] for r in shifts.to_dicts()}

    assert by_name["severe_stress"] > by_name["mild_recession"] > by_name["base"]


def test_backtest_reports_one_row_per_quarter_in_the_window(
    fixture_macro_series: pl.DataFrame,
) -> None:
    fitted = macro.fit(fixture_macro_series, "s")

    result = macro.backtest(fitted, fixture_macro_series, ("2000Q1", "2001Q4"))

    assert result.height == 8
    assert (result["predicted_default_rate"] > 0).all()
    assert "ratio_observed_to_predicted" in result.columns


def test_coefficient_table_marks_significance_from_the_hac_interval(
    fixture_macro_series: pl.DataFrame,
) -> None:
    table = macro.coefficient_table({"synthetic": macro.fit(fixture_macro_series, "s")})

    for row in table.to_dicts():
        spans_zero = row["ci_low"] <= 0.0 <= row["ci_high"]
        assert row["significant_at_95"] == (not spans_zero)
