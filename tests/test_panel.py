"""Phase 2 acceptance: splits, labels, the crisis spike, and the exclusions record."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import duckdb
import polars as pl
import pytest

from riskos.panel.config import LoanAllocation, feature_config, panel_config

PANEL_GLOB = "data/panel/panel/*/*.parquet"
RISK_SET_GLOB = "data/panel/risk_set/*/*.parquet"
EXHIBIT_CSV = Path("reports/figures/default_rate_by_quarter.csv")

needs_panel = pytest.mark.skipif(
    not any(Path().glob(PANEL_GLOB)), reason="panel not built; run `make panel`"
)


def _q(sql: str) -> list[tuple[object, ...]]:
    con = duckdb.connect()
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


# --------------------------------------------------------------------------
# Config logic — no data required.
# --------------------------------------------------------------------------


def test_loan_allocation_buckets_tile_the_whole_range_without_overlap() -> None:
    alloc = LoanAllocation(
        seed=1, fractions={"train": 0.6, "validation_in_time": 0.15, "a": 0.125, "b": 0.125}
    )

    bounds = alloc.boundaries()

    assert bounds[0][1] == 0
    assert bounds[-1][2] == 1000
    for (_, _, prev_hi), (_, lo, _) in pairwise(bounds):
        assert prev_hi == lo, "buckets must be contiguous and non-overlapping"


def test_split_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to"):
        LoanAllocation(seed=1, fractions={"train": 0.6, "test": 0.3})


def test_default_definition_is_locked() -> None:
    dd = panel_config().default_definition

    assert dd.locked is True
    assert dd.dpd_min_months == 3
    assert "whole_loan_sale" in dd.credit_event_terminations
    assert dd.excluded_not_counted_as_good == ("defect_prior_to_credit_event",)


def test_risk_set_separates_other_exit_from_prepayment() -> None:
    # Folding code 16 into `prepaid` would misattribute it to the prepayment
    # hazard in the Phase 5 competing-risks model.
    outcomes = panel_config().risk_set.outcome_enum

    assert set(outcomes) == {"alive", "default", "prepaid", "other_exit"}


# --------------------------------------------------------------------------
# Acceptance criteria, asserted against the built panel.
# --------------------------------------------------------------------------


@needs_panel
def test_no_loan_appears_in_more_than_one_split() -> None:
    overlap = _q(f"""
        SELECT COUNT(*) FROM (
            SELECT loan_sequence_number FROM read_parquet('{PANEL_GLOB}')
            GROUP BY 1 HAVING COUNT(DISTINCT split) > 1)
    """)

    assert overlap[0][0] == 0, "splits must be loan-disjoint (Phase 2 acceptance criterion)"


@needs_panel
def test_every_observation_date_falls_inside_its_split_window() -> None:
    windows = panel_config().splits.windows()
    clauses = " OR ".join(
        f"(split = '{name}' AND YEAR(observation_date) "
        f"NOT BETWEEN {lo.split('-')[0]} AND {hi.split('-')[0]})"
        for name, (lo, hi) in windows.items()
    )

    stray = _q(f"SELECT COUNT(*) FROM read_parquet('{PANEL_GLOB}') WHERE {clauses}")

    assert stray[0][0] == 0


@needs_panel
def test_the_unassigned_window_is_absent_from_the_panel() -> None:
    rows = _q(f"""
        SELECT COUNT(*) FROM read_parquet('{PANEL_GLOB}')
        WHERE YEAR(observation_date) BETWEEN 2010 AND 2014
    """)

    assert rows[0][0] == 0, "2010-2014 belongs to no split and must not be written"


@needs_panel
def test_panel_size_is_inside_the_configured_band() -> None:
    sampling = panel_config().sampling
    n = _q(f"SELECT COUNT(*) FROM read_parquet('{PANEL_GLOB}')")[0][0]

    assert sampling.target_rows_min <= n <= sampling.target_rows_max  # type: ignore[operator]


@needs_panel
def test_labels_are_binary_and_the_portfolio_rate_is_plausible() -> None:
    rows = _q(f"""
        SELECT MIN(default_12m), MAX(default_12m), AVG(default_12m::DOUBLE)
        FROM read_parquet('{PANEL_GLOB}')
    """)
    lo, hi, rate = rows[0]

    assert (lo, hi) == (0, 1)
    # A prime conforming mortgage book: sub-1% through the cycle, never 0.
    assert 0.001 < rate < 0.05  # type: ignore[operator]


@needs_panel
def test_the_stress_split_is_materially_riskier_than_training() -> None:
    """The whole point of the out-of-time design.

    If OOT-stress did not carry a far higher default rate than train, the split
    would not be testing regime change and every stability finding downstream
    would be vacuous.
    """
    rates = dict(
        _q(f"""
            SELECT split, AVG(default_12m::DOUBLE)
            FROM read_parquet('{PANEL_GLOB}') GROUP BY 1
        """)  # type: ignore[arg-type]
    )

    assert rates["oot_stress"] > 2.5 * rates["train"]


@needs_panel
def test_train_and_in_time_validation_agree_because_they_share_a_period() -> None:
    # Same window, disjoint loans. A large gap would mean the hash allocation is
    # not random with respect to risk.
    rates = dict(
        _q(f"""
            SELECT split, AVG(default_12m::DOUBLE)
            FROM read_parquet('{PANEL_GLOB}') GROUP BY 1
        """)  # type: ignore[arg-type]
    )

    assert abs(rates["train"] - rates["validation_in_time"]) < 0.002


@needs_panel
def test_risk_set_outcomes_are_confined_to_the_declared_enum() -> None:
    declared = set(panel_config().risk_set.outcome_enum)
    seen = {r[0] for r in _q(f"SELECT DISTINCT outcome FROM read_parquet('{RISK_SET_GLOB}')")}

    assert seen <= declared, f"undeclared risk-set outcomes: {sorted(seen - declared)}"


@needs_panel
def test_no_loan_month_appears_twice_in_the_risk_set() -> None:
    dupes = _q(f"""
        SELECT COUNT(*) FROM (
            SELECT loan_sequence_number, observation_date
            FROM read_parquet('{RISK_SET_GLOB}')
            GROUP BY 1, 2 HAVING COUNT(*) > 1)
    """)

    assert dupes[0][0] == 0


# --------------------------------------------------------------------------
# The exhibit that gates the phase.
# --------------------------------------------------------------------------


@pytest.mark.skipif(not EXHIBIT_CSV.exists(), reason="exhibit not built; run `make panel`")
def test_default_rate_exhibit_shows_the_2007_2009_spike() -> None:
    """Phase 2's stated acceptance criterion.

    Nothing downstream is meaningful unless the label reproduces the crisis.
    """
    df = pl.read_csv(EXHIBIT_CSV)
    pre = df.filter(pl.col("year").is_between(2004, 2006))["default_rate"].mean()
    crisis = df.filter(pl.col("year").is_between(2008, 2009))["default_rate"].max()

    assert pre is not None and crisis is not None
    assert crisis > 4 * pre, f"crisis peak {crisis:.3%} vs pre-crisis mean {pre:.3%}"
    assert crisis > 0.02


@pytest.mark.skipif(not EXHIBIT_CSV.exists(), reason="exhibit not built; run `make panel`")
def test_exhibit_png_is_written_alongside_its_data() -> None:
    # Build plan §9: every plot ships as PNG *and* CSV so the report can be
    # regenerated without rerunning models.
    assert EXHIBIT_CSV.with_suffix(".png").exists()


@pytest.mark.skipif(
    not Path("data/panel/panel_manifest.json").exists(), reason="manifest not built"
)
def test_manifest_records_every_exclusion_with_a_reason() -> None:
    manifest = json.loads(Path("data/panel/panel_manifest.json").read_text(encoding="utf-8"))

    assert manifest["exclusions"], "dropped rows must be counted with reasons"
    for exclusion in manifest["exclusions"]:
        assert exclusion["reason"]
        assert exclusion["loans"] is not None or exclusion["rows"] is not None


@needs_panel
def test_forbearance_diagnostic_is_present_and_concentrated_in_the_benign_split() -> None:
    """The COVID contamination must be visible in the data, not just the docs."""
    assert "default_in_forbearance" in set(feature_config().label_and_metadata)

    rows = dict(
        _q(f"""
            SELECT split,
                   SUM(CASE WHEN default_in_forbearance THEN 1 ELSE 0 END)::DOUBLE
                   / NULLIF(SUM(default_12m), 0)
            FROM read_parquet('{PANEL_GLOB}') GROUP BY 1
        """)  # type: ignore[arg-type]
    )

    assert rows["oot_benign"] > 0.4, "expected substantial COVID forbearance in 2019 windows"


@needs_panel
def test_no_pre_2014_split_can_carry_a_forbearance_flag() -> None:
    """The structural impossibility that should have exposed defect R-001.

    borrower_assistance_status_code is only populated from January 2014, so a
    2005 observation cannot be forbearance-flagged. The original derivation
    reported 103 such rows in train and 163 in oot_stress, because DuckDB's
    arg_min skipped null values and reached forward to a later month's code.
    Exact zero is the only admissible answer.
    """
    counts = dict(
        _q(f"""
            SELECT split, SUM(CASE WHEN default_in_forbearance THEN 1 ELSE 0 END)
            FROM read_parquet('{PANEL_GLOB}')
            WHERE split IN ('train', 'validation_in_time', 'oot_stress')
            GROUP BY 1
        """)  # type: ignore[arg-type]
    )

    assert set(counts.values()) == {0}, f"look-ahead in the forbearance flag: {counts}"
