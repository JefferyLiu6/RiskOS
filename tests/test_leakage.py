"""Leakage control (build plan §7.3) — the single most important correctness test.

THE RULE: at observation date t, a feature may use only information available at
or before t.

Every forbidden column here is measured at or after the credit event. If one
reached a training matrix, the model would learn "this loan has a disposition
expense, therefore it defaulted" — producing spectacular AUC and a worthless
risk model. These assertions run against the built artifacts, not against
intentions.

Tests that need the built panel are marked `needs_data` and skip cleanly when it
is absent, so the suite still runs on a fresh clone.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pytest

from riskos.config import data_config
from riskos.panel.config import feature_config, panel_config

PANEL_GLOB = "data/panel/panel/*/*.parquet"
RISK_SET_GLOB = "data/panel/risk_set/*/*.parquet"


def _columns(glob: str) -> set[str]:
    con = duckdb.connect()
    try:
        rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{glob}')").fetchall()
    finally:
        con.close()
    return {r[0] for r in rows}


def _built(glob: str) -> bool:
    return any(Path().glob(glob))


needs_panel = pytest.mark.skipif(not _built(PANEL_GLOB), reason="panel not built; run `make panel`")
needs_risk_set = pytest.mark.skipif(
    not _built(RISK_SET_GLOB), reason="risk set not built; run `make panel`"
)


# --------------------------------------------------------------------------
# Config-level: the allow-list must be coherent before anything is built.
# --------------------------------------------------------------------------


def test_allowed_forbidden_and_label_sets_are_disjoint() -> None:
    fcfg = feature_config()

    assert not set(fcfg.allowed) & set(fcfg.forbidden)
    assert not set(fcfg.allowed) & set(fcfg.label_and_metadata)
    assert not set(fcfg.forbidden) & set(fcfg.label_and_metadata)


def test_every_loss_and_recovery_field_is_forbidden() -> None:
    # These are the §7.3 named fields. Spelled out rather than derived, so that
    # deleting one from the config fails here instead of passing quietly.
    forbidden = set(feature_config().forbidden)

    for field in (
        "zero_balance_code",
        "zero_balance_effective_date",
        "zero_balance_removal_upb",
        "actual_loss",
        "net_sales_proceeds",
        "mi_recoveries",
        "non_mi_recoveries",
        "total_expenses",
        "delinquent_accrued_interest",
        "ddlpi",
        "estimated_ltv",
    ):
        assert field in forbidden, f"{field} must be forbidden (build plan §7.3)"


def test_termination_type_is_forbidden_because_it_derives_from_the_zero_balance_code() -> None:
    # termination_type is our own derived column, not a raw field, which makes
    # it easy to forget. It encodes exactly how the loan defaulted.
    assert "termination_type" in feature_config().forbidden


# --------------------------------------------------------------------------
# Artifact-level: assert against what was actually written to disk.
# --------------------------------------------------------------------------


@needs_panel
def test_no_forbidden_column_reaches_the_12_month_panel() -> None:
    leaked = sorted(_columns(PANEL_GLOB) & set(feature_config().forbidden))

    assert not leaked, f"target leakage in the 12-month panel: {leaked}"


@needs_risk_set
def test_no_forbidden_column_reaches_the_hazard_risk_set() -> None:
    # §7.3 applies per risk-set month: covariates at month m may use only
    # information through m. The same forbidden set applies.
    leaked = sorted(_columns(RISK_SET_GLOB) & set(feature_config().forbidden))

    assert not leaked, f"target leakage in the hazard risk set: {leaked}"


@needs_panel
def test_panel_contains_nothing_outside_the_allow_list() -> None:
    fcfg = feature_config()
    permitted = set(fcfg.allowed) | set(fcfg.label_and_metadata)

    unexpected = sorted(_columns(PANEL_GLOB) - permitted)

    assert not unexpected, f"columns present but on no list: {unexpected}"


@needs_panel
def test_every_allowed_feature_is_actually_present() -> None:
    # The mirror of the above: an allow-list entry that never materialises means
    # the panel is quietly missing a feature the config promises.
    missing = sorted(set(feature_config().allowed) - _columns(PANEL_GLOB))

    assert not missing, f"allow-listed but absent from the panel: {missing}"


@needs_risk_set
def test_risk_set_carries_its_outcome_and_no_label_from_the_12_month_panel() -> None:
    columns = _columns(RISK_SET_GLOB)

    assert "outcome" in columns
    assert "default_12m" not in columns  # the 12-month label is not a hazard covariate


# --------------------------------------------------------------------------
# Semantic leakage: the label must not be reconstructible from a feature.
# --------------------------------------------------------------------------


@needs_panel
def test_delinquency_status_at_t_never_already_satisfies_the_default_definition() -> None:
    """Eligibility requires the loan is not *already* 90+ DPD at t.

    If it were, the label would be trivially implied by a feature and every
    metric would be meaningless. This is the check that would catch an
    off-by-one in the observation window.
    """
    dd = panel_config().default_definition
    con = duckdb.connect()
    try:
        already = con.execute(f"""
            SELECT COUNT(*) FROM read_parquet('{PANEL_GLOB}')
            WHERE COALESCE(TRY_CAST({dd.dpd_status_column} AS INTEGER), -1) >= {dd.dpd_min_months}
               OR {dd.dpd_status_column} IN ('RA')
        """).fetchone()
    finally:
        con.close()

    assert already is not None and already[0] == 0, (
        f"{already[0] if already else '?'} panel rows are already in default at the "
        "observation date; the label leaks into current_loan_delinquency_status"
    )


@needs_panel
def test_the_interim_tables_still_contain_the_forbidden_fields() -> None:
    """Guard against the wrong kind of fix.

    The forbidden fields must survive in `data/interim/` — Phase 5 needs
    actual_loss and the recovery components to estimate LGD. Leakage control is
    about what enters the *training matrix*, not about deleting data. If someone
    "fixes" a leakage failure by dropping columns at ingest, this fails.
    """
    interim = data_config().paths.interim / "performance"
    if not any(interim.glob("*/*.parquet")):
        pytest.skip("interim performance tables not built; run `make ingest`")

    columns = _columns((interim / "*" / "*.parquet").as_posix())

    for field in ("actual_loss", "net_sales_proceeds", "mi_recoveries", "zero_balance_code"):
        assert field in columns, f"{field} must remain in interim for Phase 5 LGD estimation"


# --------------------------------------------------------------------------
# Terminal-state contamination: a row's own state revealing its own outcome.
# --------------------------------------------------------------------------


@needs_risk_set
def test_the_raw_balance_reveals_the_termination_and_must_not_be_used_unlagged() -> None:
    """Documents the contamination that defect R-009 exposed.

    A loan's balance in the month it terminates is zero. This is not a
    forbidden column — `current_actual_upb` is legitimately allow-listed,
    because at an observation date for a LIVE loan it is exactly the exposure
    the model should see. The defect is narrower and subtler: within the risk
    set, the terminal month's state encodes the terminal event, so any covariate
    derived from the contemporaneous balance separates the outcome perfectly.

    Column-membership leakage tests cannot catch this. The assertion here is
    that the contamination is real, so the lag in the hazard query is protecting
    against something rather than being decorative.
    """
    con = duckdb.connect()
    try:
        prepaid_zero, alive_zero, alive_n = con.execute(f"""
            SELECT
                AVG(CASE WHEN outcome = 'prepaid'
                         THEN (current_actual_upb = 0)::INT END),
                AVG(CASE WHEN outcome = 'alive'
                         THEN (current_actual_upb = 0)::INT END),
                SUM((outcome = 'alive')::INT)
            FROM read_parquet('{RISK_SET_GLOB}')
            WHERE YEAR(observation_date) <= 2006
        """).fetchone()  # type: ignore[misc]
    finally:
        con.close()

    assert prepaid_zero > 0.99, "expected a zero balance in essentially every prepayment month"
    assert alive_zero < 0.001, "expected a non-zero balance in essentially every alive month"
    assert alive_n > 1_000_000


@needs_risk_set
def test_hazard_covariates_are_lagged_so_they_do_not_separate_the_outcome() -> None:
    """The fix for R-009, asserted on the covariates the model actually sees.

    A perfectly separating covariate is not merely bad practice: it drove the
    prepayment GLM to coefficients of order 1e14 and a non-converged fit.
    """
    from riskos.models.train_hazard import load_risk_set

    risk_set = load_risk_set(4_000)
    if risk_set.height == 0:  # pragma: no cover - depends on sampling
        pytest.skip("empty hazard sample")

    prepaid = risk_set.filter(pl.col("outcome") == "prepaid")["amortisation_ratio"]
    alive = risk_set.filter(pl.col("outcome") == "alive")["amortisation_ratio"]
    if prepaid.len() == 0:  # pragma: no cover
        pytest.skip("no prepayment events in this sample")

    # After lagging, a prepaying loan's balance is its balance the month BEFORE
    # it prepaid, which is an ordinary positive number.
    assert float(prepaid.median() or 0.0) > 0.1
    assert abs(float(prepaid.median() or 0.0) - float(alive.median() or 0.0)) < 0.5


@needs_risk_set
def test_the_contemporaneous_delinquency_state_is_the_default_event_itself() -> None:
    """Documents the contamination behind the F-012 re-diagnosis.

    Default is DEFINED as reaching 90 days past due. So within the risk set the
    contemporaneous delinquency status does not merely predict the outcome, it
    IS the outcome: every row at 3+ months delinquent defaults that month and no
    current row ever does. Fitting on it is not a model, it is an identity.

    This is the same class of defect as R-009 and is invisible to the
    column-membership tests, because `current_loan_delinquency_status` is
    legitimately allow-listed — at an observation date for a live loan it is
    exactly what a PD model should see. The panel is guarded by
    `test_delinquency_status_at_t_never_already_satisfies_the_default_definition`;
    the risk set cannot be, because those rows are the events.

    Asserted so the lag in the hazard query cannot be quietly reverted.
    """
    from riskos.models.train_hazard import load_risk_set

    raw = load_risk_set(6_000, lag_delinquency=False)
    if raw.height == 0:  # pragma: no cover - depends on sampling
        pytest.skip("empty hazard sample")

    defaulted = raw.filter(pl.col("months_delinquent") >= 3)
    current = raw.filter(pl.col("months_delinquent") == 0)
    if defaulted.height == 0:  # pragma: no cover - depends on sampling
        pytest.skip("no rows at 90+ DPD in this sample")

    hazard_when_90_plus = float((defaulted["outcome"] == "default").cast(pl.Float64).mean() or 0.0)
    hazard_when_current = float((current["outcome"] == "default").cast(pl.Float64).mean() or 0.0)

    assert hazard_when_90_plus == 1.0, (
        "expected the contemporaneous 90+ DPD state to imply default exactly; "
        f"got {hazard_when_90_plus}"
    )
    assert hazard_when_current == 0.0, (
        f"expected a current loan never to default in the same month; got {hazard_when_current}"
    )


@needs_risk_set
def test_the_lagged_delinquency_state_predicts_without_separating() -> None:
    """The fix for F-012, asserted on the covariate the model actually sees.

    Lagged, the state is known at the START of the month and is a genuine
    predictor: the hazard rises monotonically across the states without any of
    them being degenerate. A band at exactly 0.0 or exactly 1.0 would mean the
    separation is back.
    """
    from riskos.models import hazard as hz
    from riskos.models.train_hazard import load_risk_set

    risk_set = load_risk_set(6_000)
    if risk_set.height == 0:  # pragma: no cover - depends on sampling
        pytest.skip("empty hazard sample")

    by_band = (
        risk_set.with_columns(hz.delinquency_band(pl.col("months_delinquent")).alias("band"))
        .group_by("band")
        .agg(
            n=pl.len(),
            hazard=(pl.col("outcome") == "default").cast(pl.Float64).mean(),
        )
        .sort("band")
    )
    observed = {r["band"]: r for r in by_band.to_dicts()}
    if len(observed) < 3:  # pragma: no cover - depends on sampling
        pytest.skip("not every delinquency state present in this sample")

    hazards = [observed[b]["hazard"] for b in hz.delinquency_band_labels()]
    for band, h in zip(hz.delinquency_band_labels(), hazards, strict=True):
        assert 0.0 < h < 1.0, f"{band} hazard {h} is degenerate; the lag has been lost"
    assert hazards == sorted(hazards), f"expected hazard to rise with delinquency, got {hazards}"


@needs_risk_set
def test_both_cause_specific_hazards_converge() -> None:
    """Non-convergence was the symptom that led to R-009; it must stay fixed."""
    import warnings

    from riskos.models import hazard
    from riskos.models.train_hazard import COVARIATES, load_risk_set

    risk_set = load_risk_set(6_000)
    design, prepared, _ = hazard.design_matrix(risk_set, COVARIATES)
    design = design[hazard.estimable_columns(design)]
    clusters = prepared["calendar_period"].to_numpy()

    for cause in hazard.CAUSES:
        events = (prepared["outcome"] == cause).cast(pl.Int64).to_numpy()
        if events.sum() < 20:  # pragma: no cover - small-sample guard
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result, _ = hazard.fit_cause(design, events, clusters, cause)
        assert result.converged, f"{cause} hazard did not converge"
        assert float(np.abs(result.params).max()) < 100.0, f"{cause} coefficients diverged"
