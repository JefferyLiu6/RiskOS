"""Performance monitoring: discrimination and calibration as outcomes mature.

Every row here carries ``outcome_matured_at`` — the window's end plus the
outcome window — because a performance metric does not exist before that date.
Reporting a 2008Q1 AUC without recording that it could not be computed until
2009Q1 turns a monitoring timeline into a hindsight chart, and hindsight charts
make every control look prescient.

Windows with too few defaults report null rather than a number. AUC on a dozen
defaults is not a weak estimate of discrimination, it is not an estimate of
discrimination, and a null that a reviewer must account for is safer than a
figure they might quote.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import numpy.typing as npt
import polars as pl

from riskos.log import get_logger
from riskos.metrics import auc, brier_decomposition, gini, ks_statistic, observed_vs_expected

log = get_logger(__name__)


def add_months(day: date, months: int) -> date:
    """Shift a date by whole months, clamping the day to the target month."""
    total = day.month - 1 + months
    year, month = day.year + total // 12, total % 12 + 1
    last = [
        31,
        29 if year % 4 == 0 and (year % 100 or year % 400 == 0) else 28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ][month - 1]
    return date(year, month, min(day.day, last))


@dataclass(frozen=True)
class PerformanceWindow:
    """Discrimination and calibration for one window, with its maturity date."""

    window: str
    window_end: date
    outcome_matured_at: date
    n_rows: int
    n_defaults: int
    observed_rate: float
    expected_rate: float
    observed_over_expected: float | None
    auc: float | None
    gini: float | None
    ks: float | None
    brier: float | None
    brier_reliability: float | None
    evaluated: bool
    reason: str


def evaluate_window(
    label: str,
    window_end: date,
    y: npt.NDArray[np.int64],
    p: npt.NDArray[np.float64],
    *,
    outcome_window_months: int,
    min_defaults: int,
) -> PerformanceWindow:
    """Metrics for one window, or a recorded refusal to compute them."""
    matured = add_months(window_end, outcome_window_months)
    n_defaults = int(y.sum())
    oe = observed_vs_expected(y, p)
    thin = n_defaults < min_defaults or n_defaults == y.size

    if thin:
        reason = (
            f"{n_defaults} defaults, below the {min_defaults} needed for a rank-based metric"
            if n_defaults < min_defaults
            else "window has no non-defaults, so discrimination is undefined"
        )
        log.warning("performance_window_not_evaluated", window=label, reason=reason)
        return PerformanceWindow(
            window=label,
            window_end=window_end,
            outcome_matured_at=matured,
            n_rows=int(y.size),
            n_defaults=n_defaults,
            observed_rate=oe["observed_rate"],
            expected_rate=oe["expected_rate"],
            # Calibration survives a thin window in a way ranking does not: an
            # observed-over-expected ratio is a comparison of two means and is
            # still meaningful, if wide, on few events.
            observed_over_expected=oe["ratio"],
            auc=None,
            gini=None,
            ks=None,
            brier=None,
            brier_reliability=None,
            evaluated=False,
            reason=reason,
        )

    brier = brier_decomposition(y, p)
    window = PerformanceWindow(
        window=label,
        window_end=window_end,
        outcome_matured_at=matured,
        n_rows=int(y.size),
        n_defaults=n_defaults,
        observed_rate=oe["observed_rate"],
        expected_rate=oe["expected_rate"],
        observed_over_expected=oe["ratio"],
        auc=auc(y, p),
        gini=gini(y, p),
        ks=ks_statistic(y, p).statistic,
        brier=brier.brier,
        brier_reliability=brier.reliability,
        evaluated=True,
        reason="",
    )
    log.info(
        "performance_window",
        window=label,
        n=window.n_rows,
        defaults=n_defaults,
        gini=round(window.gini or 0.0, 4),
        observed_over_expected=round(window.observed_over_expected or 0.0, 4),
    )
    return window


# Declared rather than inferred. The early windows of any monitoring run are
# the thin ones, so every nullable metric is null in the first rows and polars
# would infer a null column and then fail on the first real float.
SCHEMA = pl.Schema(
    {
        "window": pl.Utf8,
        "window_end": pl.Date,
        "outcome_matured_at": pl.Date,
        "n_rows": pl.Int64,
        "n_defaults": pl.Int64,
        "observed_rate": pl.Float64,
        "expected_rate": pl.Float64,
        "observed_over_expected": pl.Float64,
        "auc": pl.Float64,
        "gini": pl.Float64,
        "ks": pl.Float64,
        "brier": pl.Float64,
        "brier_reliability": pl.Float64,
        "evaluated": pl.Boolean,
        "reason": pl.Utf8,
    }
)


def performance_timeline(
    windows: list[tuple[str, date, npt.NDArray[np.int64], npt.NDArray[np.float64]]],
    *,
    outcome_window_months: int,
    min_defaults: int,
) -> pl.DataFrame:
    """Discrimination and calibration for every window, in window order."""
    rows = [
        evaluate_window(
            label,
            window_end,
            y,
            p,
            outcome_window_months=outcome_window_months,
            min_defaults=min_defaults,
        )
        for label, window_end, y, p in windows
        if y.size
    ]
    return pl.DataFrame([asdict(r) for r in rows], schema=SCHEMA)


def visible_at(timeline: pl.DataFrame, as_at: date) -> pl.DataFrame:
    """The timeline as it would have appeared on a real reporting date.

    Anything whose outcome window had not closed by ``as_at`` is dropped. This
    is what a monitoring pack could actually have shown on the day, and the gap
    between this and the full timeline is the blind spot the drift metrics
    exist to cover.
    """
    if timeline.is_empty():
        return timeline
    return timeline.filter(pl.col("outcome_matured_at") <= as_at)
