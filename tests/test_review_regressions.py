"""Regression tests for defects found in code review.

Each test fails against the original implementation and passes against the fix.
Grouped here rather than scattered so the review trail stays legible.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import polars as pl
import pytest

import riskos.log as log_module
from riskos.metrics.stability import MISSING_LEVEL, csi_for_feature
from riskos.models.scorecard import Scorecard

# --------------------------------------------------------------------------
# R1 (high) — arg_min skipped null values and reached forward in time.
# --------------------------------------------------------------------------


def test_assistance_code_comes_from_the_first_default_month_even_when_null() -> None:
    """DuckDB's arg_min ignores rows whose VALUE is null.

    Passing the bare column skipped a first default month with no assistance
    code and returned a LATER month's code instead — a look-ahead that falsely
    flagged 1,600 loans as COVID forbearance. Wrapping the value in a struct
    makes the ranked value non-null, so the earliest default row always wins.
    """
    rows = "(DATE '2020-01-01', NULL, true), (DATE '2020-05-01', 'F', true)"
    bare, wrapped = duckdb.sql(f"""
        SELECT arg_min(b, p) FILTER (WHERE d),
               (arg_min({{'code': b}}, p) FILTER (WHERE d))['code']
        FROM (VALUES {rows}) t(p, b, d)
    """).fetchone()  # type: ignore[misc]

    assert bare == "F", "documents the original defect"
    assert wrapped is None, "the first default month had no assistance code"


def test_a_genuine_forbearance_at_first_default_is_still_detected() -> None:
    rows = "(DATE '2020-01-01', 'F', true), (DATE '2020-05-01', NULL, true)"
    (code,) = duckdb.sql(f"""
        SELECT (arg_min({{'code': b}}, p) FILTER (WHERE d))['code']
        FROM (VALUES {rows}) t(p, b, d)
    """).fetchone()  # type: ignore[misc]

    assert code == "F"


# --------------------------------------------------------------------------
# R2 (medium) — CSI was blind to a change in missingness.
# --------------------------------------------------------------------------


def test_numeric_csi_detects_a_change_in_the_null_rate() -> None:
    """Dropping nulls on both sides reported exactly 0.0 for a 50%->0% shift."""
    reference = pl.Series("x", [1.0, 2.0, None, None] * 250)
    actual = pl.Series("x", [1.0, 2.0, 1.0, 2.0] * 250)

    result = csi_for_feature(reference, actual)

    assert result.value > 0.25, "a halving of the null rate is significant drift"
    assert MISSING_LEVEL in result.detail["bin"].to_list()


def test_categorical_csi_detects_a_change_in_the_null_rate() -> None:
    reference = pl.Series("c", ["A", "B", None, None] * 250)
    actual = pl.Series("c", ["A", "B", "A", "B"] * 250)

    result = csi_for_feature(reference, actual)

    assert result.value > 0.25
    assert MISSING_LEVEL in result.detail["bin"].to_list()


@pytest.mark.parametrize(
    "series",
    [
        pl.Series("x", [1.0, 2.0, None, None] * 250),
        pl.Series("c", ["A", "B", None, None] * 250),
    ],
    ids=["numeric", "categorical"],
)
def test_a_feature_against_itself_is_still_exactly_zero(series: pl.Series) -> None:
    """Adding a missing bin must not manufacture drift where there is none."""
    assert csi_for_feature(series, series).value == pytest.approx(0.0, abs=1e-12)


def test_missing_share_is_measured_against_the_full_population() -> None:
    reference = pl.Series("x", [1.0, 2.0, None, None] * 250)

    exp = csi_for_feature(reference, reference).detail

    missing = exp.filter(pl.col("bin") == MISSING_LEVEL)["expected"][0]
    assert missing == pytest.approx(0.5), "denominator must include nulls"


# --------------------------------------------------------------------------
# R3 (medium) — explanations depended on the batch being scored.
# --------------------------------------------------------------------------


@pytest.fixture
def fitted_card() -> tuple[Scorecard, pd.DataFrame]:
    rng = np.random.default_rng(20250901)
    n = 3_000
    y = rng.binomial(1, 0.1, n)
    woe = pd.DataFrame(
        {
            "a": np.where(y == 1, rng.normal(-1.0, 0.5, n), rng.normal(0.6, 0.5, n)),
            "b": np.where(y == 1, rng.normal(-0.3, 0.5, n), rng.normal(0.2, 0.5, n)),
        }
    )
    card = Scorecard(["a", "b"])
    card.fit(woe, y)
    return card, woe


def test_a_borrower_scored_alone_gets_a_real_explanation(
    fitted_card: tuple[Scorecard, pd.DataFrame],
) -> None:
    """Centring on the batch made a single-row explanation identically zero.

    This is the Phase 6 API path: /score receives one loan.
    """
    card, woe = fitted_card

    drivers = card.principal_drivers(woe.head(1))[0]

    assert any(abs(d["points"]) > 1e-6 for d in drivers)


def test_the_same_borrower_gets_the_same_explanation_in_any_batch(
    fitted_card: tuple[Scorecard, pd.DataFrame],
) -> None:
    card, woe = fitted_card

    alone = card.principal_drivers(woe.head(1))[0]
    in_batch = card.principal_drivers(woe.head(500))[0]
    in_other_batch = card.principal_drivers(woe.iloc[list(range(0, 2000, 7))])[0]

    for a, b, c in zip(alone, in_batch, in_other_batch, strict=True):
        assert a["feature"] == b["feature"] == c["feature"]
        assert a["points"] == pytest.approx(b["points"])
        assert a["points"] == pytest.approx(c["points"])


# --------------------------------------------------------------------------
# R4 (low) — CLI logging flags were swallowed by the idempotency guard.
# --------------------------------------------------------------------------


def test_configure_reapplies_when_the_settings_actually_change() -> None:
    log_module.configure(level="INFO", json_output=False)
    log_module.configure(level="INFO", json_output=True)

    assert log_module._CONFIGURED == ("INFO", True)


def test_get_logger_does_not_revert_an_explicit_configuration() -> None:
    """The actual defect: any later import silently reset the CLI's choice."""
    log_module.configure(level="DEBUG", json_output=True)

    log_module.get_logger("some.module.imported.later")

    assert log_module._CONFIGURED == ("DEBUG", True)


def test_get_logger_still_applies_defaults_when_nothing_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(log_module, "_CONFIGURED", None)

    log_module.get_logger("first.use")

    assert log_module._CONFIGURED == ("INFO", False)
