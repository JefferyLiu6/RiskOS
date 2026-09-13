"""Phase 5 — empirical Loss Given Default (build plan §7.9).

    LGD = actual_loss / UPB_at_default

**This is not clipped.** Freddie Mac's actual-loss measure is disclosed at
property disposition, not at the default moment, and nets zero-balance removal
UPB, delinquent accrued interest, sale proceeds, MI and non-MI recoveries, and
expenses. Values below 0 and above 1 are therefore economically real:

* **LGD < 0** (3.7% of cases): recoveries exceeded the balance. Median MI
  recovery of $7,965 and median sale proceeds of $138,652 against a median
  balance of $133,435. The property sold for more than was owed, or mortgage
  insurance covered more than the shortfall.
* **LGD > 1** (7.8%): accrued interest plus foreclosure expenses exceeded the
  balance. Median balance $72,995 against a median loss of $84,842.

Both are kept. ``np.clip(lgd, 0, 1)`` with no accompanying analysis is a
project-level failure.

**What IS excluded, and why it is not clipping.** A small number of loans have a
near-zero balance at disposition while still incurring foreclosure expenses. The
most extreme has a balance of **one cent** against $6,573 of expenses, giving a
ratio of 657,341. That is a degenerate denominator, not a 65-million-percent
loss: the ratio is uninformative when the quantity it divides by has rounded to
nothing. Dropping the three loans below a $100 balance moves the mean from
**42.62 to 0.4926** — a 99% change from removing 0.02% of observations.

The remedy is a materiality floor on the **denominator**, not a bound on the
ratio. Those are different operations: one removes observations where the
statistic is undefined in practice, the other alters observations that are
economically real. Both are registered as named assumptions with sensitivity
tests (LGD_001, LGD_002).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from riskos.config import DataConfig, data_config
from riskos.log import get_logger
from riskos.panel.config import panel_config

log = get_logger(__name__)

CREDIT_EVENTS = (
    "third_party_sale",
    "short_sale_or_charge_off",
    "reo_disposition",
    "whole_loan_sale",
)


@dataclass(frozen=True)
class LGDDistribution:
    """The raw distribution, reported before any treatment is applied."""

    n: int
    mean: float
    median: float
    share_below_zero: float
    share_above_one: float
    percentiles: dict[str, float]


def observations(cfg: DataConfig | None = None, upb_floor: float = 0.0) -> pl.DataFrame:
    """Every credit-event termination with a populated loss and balance.

    ``upb_floor`` is the denominator materiality floor. At 0.0 this returns the
    genuinely raw distribution, which is what the exhibit reports first.
    """
    cfg = cfg or data_config()
    perf = (cfg.paths.interim / "performance" / "*" / "*.parquet").as_posix()
    orig = (cfg.paths.interim / "origination" / "*" / "*.parquet").as_posix()
    events = ", ".join(f"'{e}'" for e in CREDIT_EVENTS)
    con = duckdb.connect()
    try:
        return con.execute(f"""
            SELECT p.loan_sequence_number,
                   p.monthly_reporting_period AS default_date,
                   p.termination_type,
                   p.actual_loss,
                   p.zero_balance_removal_upb AS upb_at_default,
                   p.net_sales_proceeds, p.mi_recoveries, p.non_mi_recoveries,
                   p.total_expenses, p.delinquent_accrued_interest,
                   o.original_ltv, o.property_state, o.credit_score, o.original_upb,
                   p.actual_loss / p.zero_balance_removal_upb AS lgd
            FROM read_parquet('{perf}') p
            JOIN read_parquet('{orig}') o USING (loan_sequence_number)
            WHERE p.termination_type IN ({events})
              AND p.actual_loss IS NOT NULL
              AND p.zero_balance_removal_upb > {upb_floor}
        """).pl()
    finally:
        con.close()


def describe(frame: pl.DataFrame) -> LGDDistribution:
    """Summarise without altering anything."""
    lgd = frame["lgd"].to_numpy()
    return LGDDistribution(
        n=int(lgd.size),
        mean=float(lgd.mean()),
        median=float(np.median(lgd)),
        share_below_zero=float((lgd < 0).mean()),
        share_above_one=float((lgd > 1).mean()),
        percentiles={
            f"p{p}": float(np.percentile(lgd, p)) for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)
        },
    )


def denominator_sensitivity(cfg: DataConfig | None = None) -> pl.DataFrame:
    """Mean LGD across candidate denominator floors (assumption LGD_001).

    The point of the exhibit is that the mean is wildly unstable between $0 and
    $100 and essentially flat thereafter, which identifies the problem as a
    handful of degenerate denominators rather than a modelling choice with real
    latitude.
    """
    raw = observations(cfg, upb_floor=0.0)
    total = raw.height
    rows = []
    for floor in (0.0, 100.0, 500.0, 1_000.0, 2_500.0, 5_000.0, 10_000.0, 25_000.0):
        kept = raw.filter(pl.col("upb_at_default") >= floor)["lgd"].to_numpy()
        rows.append(
            {
                "upb_floor": floor,
                "n_kept": int(kept.size),
                "share_dropped": 1.0 - kept.size / total,
                "mean_lgd": float(kept.mean()),
                "median_lgd": float(np.median(kept)),
                "p99_lgd": float(np.percentile(kept, 99)),
            }
        )
    return pl.DataFrame(rows)


def bounding_sensitivity(frame: pl.DataFrame) -> pl.DataFrame:
    """Unbounded against [0, 1]-bounded LGD (assumption LGD_002).

    Reported as a pair, never substituted silently. The bounded variant exists
    only because some downstream conventions require LGD in [0, 1]; it is not a
    correction, and the difference it makes to portfolio ECL is the number that
    matters.
    """
    lgd = frame["lgd"].to_numpy()
    bounded = np.clip(lgd, 0.0, 1.0)
    return pl.DataFrame(
        [
            {"variant": "unbounded", "mean_lgd": float(lgd.mean()), "n": int(lgd.size)},
            {"variant": "bounded_0_1", "mean_lgd": float(bounded.mean()), "n": int(bounded.size)},
            {
                "variant": "relative_difference",
                "mean_lgd": float(bounded.mean() / lgd.mean() - 1.0),
                "n": int(lgd.size),
            },
        ]
    )


def ltv_band(expr: pl.Expr) -> pl.Expr:
    """Original-LTV bands. Equity at origination is the primary LGD driver."""
    return (
        pl.when(expr <= 60)
        .then(pl.lit("<=60"))
        .when(expr <= 70)
        .then(pl.lit("61-70"))
        .when(expr <= 80)
        .then(pl.lit("71-80"))
        .when(expr <= 90)
        .then(pl.lit("81-90"))
        .when(expr <= 95)
        .then(pl.lit("91-95"))
        .otherwise(pl.lit(">95"))
    )


def segment(frame: pl.DataFrame, credibility_k: float, bounded: bool = False) -> pl.DataFrame:
    """Segment LGD by LTV band and state, shrinking thin cells to the mean.

    Credibility weighting: a segment with ``n`` observations is weighted
    ``n / (n + k)`` toward its own mean and the remainder toward the global
    mean. A state with four disposals should not be trusted to three decimal
    places, and shrinkage is the standard remedy — it degrades gracefully to the
    portfolio mean rather than to noise.
    """
    column = pl.col("lgd").clip(0.0, 1.0) if bounded else pl.col("lgd")
    global_mean = float(frame.select(column.mean()).item())
    return (
        frame.with_columns(ltv_band(pl.col("original_ltv")).alias("ltv_band"))
        .group_by(["ltv_band", "property_state"])
        .agg(n=pl.len(), raw_mean=column.mean(), raw_std=column.std())
        .with_columns(credibility=pl.col("n") / (pl.col("n") + credibility_k))
        .with_columns(
            shrunk_lgd=pl.col("credibility") * pl.col("raw_mean")
            + (1 - pl.col("credibility")) * global_mean
        )
        .with_columns(global_mean=pl.lit(global_mean))
        .sort(["ltv_band", "property_state"])
    )


def tail_investigation(cfg: DataConfig | None = None) -> pl.DataFrame:
    """Characteristics of each LGD band — the evidence behind the write-up."""
    raw = observations(cfg, upb_floor=0.0)
    return (
        raw.with_columns(
            band=pl.when(pl.col("lgd") > 5)
            .then(pl.lit("LGD > 5"))
            .when(pl.col("lgd") > 2)
            .then(pl.lit("LGD 2-5"))
            .when(pl.col("lgd") > 1)
            .then(pl.lit("LGD 1-2"))
            .when(pl.col("lgd") >= 0)
            .then(pl.lit("LGD 0-1"))
            .otherwise(pl.lit("LGD < 0"))
        )
        .group_by("band")
        .agg(
            n=pl.len(),
            median_upb=pl.col("upb_at_default").median(),
            min_upb=pl.col("upb_at_default").min(),
            median_loss=pl.col("actual_loss").median(),
            median_mi_recovery=pl.col("mi_recoveries").median(),
            median_proceeds=pl.col("net_sales_proceeds").median(),
            median_expenses=pl.col("total_expenses").median(),
        )
        .sort("median_upb", descending=True)
    )


def estimate(
    cfg: DataConfig | None = None,
    upb_floor: float | None = None,
    credibility_k: float | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, LGDDistribution]:
    """Full LGD estimation: raw description, segments, and the observation set."""
    assumptions = _lgd_assumptions()
    floor = upb_floor if upb_floor is not None else assumptions["upb_floor"]
    k = credibility_k if credibility_k is not None else assumptions["credibility_k"]

    raw = observations(cfg, upb_floor=0.0)
    raw_summary = describe(raw)
    log.warning(
        "lgd_raw_distribution",
        n=raw_summary.n,
        mean=round(raw_summary.mean, 4),
        median=round(raw_summary.median, 4),
        note="mean is dominated by degenerate denominators; see LGD_001",
    )
    usable = raw.filter(pl.col("upb_at_default") >= floor)
    log.info(
        "lgd_denominator_floor_applied",
        floor=floor,
        dropped=raw.height - usable.height,
        mean_before=round(raw_summary.mean, 4),
        mean_after=round(float(usable["lgd"].to_numpy().mean()), 4),
    )
    return usable, segment(usable, k), raw_summary


def _lgd_assumptions() -> dict[str, float]:
    """Read the registered values rather than hardcoding them (build plan rule 5)."""
    from riskos.config import CONF_DIR, load_yaml

    register = {a["id"]: a for a in load_yaml(CONF_DIR / "assumptions.yaml")["assumptions"]}
    return {
        "upb_floor": float(register["LGD_001"]["value"]["upb_floor"]),
        "credibility_k": float(register["LGD_003"]["value"]),
    }


def write_exhibits(out_dir: Path, cfg: DataConfig | None = None) -> LGDDistribution:
    """Persist the distribution, the tail investigation, and both sensitivities."""
    out_dir.mkdir(parents=True, exist_ok=True)
    usable, segments, raw_summary = estimate(cfg)
    raw = observations(cfg, upb_floor=0.0)
    pl.DataFrame(
        [{"metric": k, "value": v} for k, v in raw_summary.percentiles.items()]
        + [
            {"metric": "mean_raw", "value": raw_summary.mean},
            {"metric": "median_raw", "value": raw_summary.median},
            {"metric": "share_below_zero", "value": raw_summary.share_below_zero},
            {"metric": "share_above_one", "value": raw_summary.share_above_one},
            {"metric": "n", "value": float(raw_summary.n)},
        ]
    ).write_csv(out_dir / "lgd_raw_distribution.csv")
    tail_investigation(cfg).write_csv(out_dir / "lgd_tail_investigation.csv")
    denominator_sensitivity(cfg).write_csv(out_dir / "lgd_denominator_sensitivity.csv")
    bounding_sensitivity(usable).write_csv(out_dir / "lgd_bounding_sensitivity.csv")
    segments.write_csv(out_dir / "lgd_segments.csv")
    raw.select(["lgd", "upb_at_default", "original_ltv", "property_state"]).write_parquet(
        out_dir / "lgd_observations.parquet"
    )
    log.info("wrote_lgd_exhibits", path=str(out_dir), segments=segments.height)
    return raw_summary


def default_definition_events() -> tuple[str, ...]:
    """Credit events, read from the locked definition rather than redeclared."""
    return tuple(panel_config().default_definition.credit_event_terminations)
