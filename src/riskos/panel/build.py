"""Phase 2 — observation panel, monthly risk set, and labels.

Design (build plan §7.2): an observation-cohort panel. At each quarter-end t,
every eligible loan contributes one row, labelled 1 if it enters default in the
12 months after t. Separately, a monthly risk set gives one row per loan-month
while the loan is alive, for the discrete-time hazard model in Phase 5.

The two matrices are built from the same event flags, so the 12-month model and
the hazard model cannot disagree about what a default is.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import duckdb

from riskos.config import DataConfig, data_config
from riskos.log import get_logger
from riskos.panel import plots
from riskos.panel.config import FeatureConfig, PanelConfig, feature_config, panel_config

log = get_logger(__name__)

MANIFEST_NAME = "panel_manifest.json"
FAR_FUTURE = "9999-12-01"


def _sql_list(values: tuple[str, ...] | list[str]) -> str:
    """Render a Python sequence as a SQL IN-list."""
    inner = ", ".join(f"'{v}'" for v in values)
    return f"({inner})" if inner else "('__none__')"


@dataclass
class Exclusion:
    stage: str
    reason: str
    loans: int | None = None
    rows: int | None = None


@dataclass
class PanelReport:
    observation_dates: int
    panel_rows_eligible: int
    panel_rows_observable: int
    panel_rows_in_split: int
    panel_rows_sampled: int
    sampling_rate: float
    risk_set_rows: int
    default_rate: float
    exclusions: list[Exclusion] = field(default_factory=list)


def _register_views(con: duckdb.DuckDBPyConnection, cfg: DataConfig, pcfg: PanelConfig) -> None:
    """Event flags per loan-month, then first-event dates per loan."""
    dd = pcfg.default_definition
    perf = (cfg.paths.interim / "performance" / "*" / "*.parquet").as_posix()
    orig = (cfg.paths.interim / "origination" / "*" / "*.parquet").as_posix()

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW flagged AS
        SELECT
            loan_sequence_number,
            monthly_reporting_period AS period,
            current_actual_upb, current_loan_delinquency_status, loan_age,
            remaining_months_to_legal_maturity, current_interest_rate,
            current_deferred_upb, modification_flag,
            borrower_assistance_status_code,
            (
                COALESCE(TRY_CAST({dd.dpd_status_column} AS INTEGER), -1) >= {dd.dpd_min_months}
                OR {dd.dpd_status_column} IN {_sql_list(dd.dpd_also_default_statuses)}
                OR termination_type IN {_sql_list(dd.credit_event_terminations)}
            ) AS is_default,
            termination_type = 'prepaid_or_matured' AS is_prepaid,
            termination_type = 'reperforming_loan_securitization' AS is_other_exit,
            termination_type = 'defect_prior_to_credit_event' AS is_defect
        FROM read_parquet('{perf}');
    """)
    con.execute("""
        CREATE OR REPLACE TEMP VIEW loan_events AS
        SELECT loan_sequence_number,
            MIN(period) AS first_month,
            MAX(period) AS last_month,
            MIN(CASE WHEN is_default    THEN period END) AS default_month,
            MIN(CASE WHEN is_prepaid    THEN period END) AS prepaid_month,
            MIN(CASE WHEN is_other_exit THEN period END) AS other_exit_month,
            MAX(CASE WHEN is_defect THEN 1 ELSE 0 END)   AS has_defect,
            -- Assistance status in the month the loan first hit the default
            -- definition. 'F' = forbearance: 90+ DPD under a payment-relief
            -- programme rather than an economic credit event. Diagnostic only,
            -- never a feature - it describes the outcome window.
            --
            -- The value is wrapped in a struct before arg_min because DuckDB's
            -- arg_min ignores rows whose VALUE is null. Passing the bare column
            -- would skip a first default month with no assistance code and
            -- return the code from a LATER default month instead - a
            -- look-ahead that falsely flagged 1,600 loans as forbearance.
            -- A struct is never null, so the earliest default row always wins
            -- and a missing code correctly stays missing.
            (arg_min({'code': borrower_assistance_status_code}, period)
                FILTER (WHERE is_default))['code'] AS default_assistance_code
        FROM flagged GROUP BY 1;
    """)
    con.execute(f"CREATE OR REPLACE TEMP VIEW origination AS SELECT * FROM read_parquet('{orig}');")


def _register_splits(con: duckdb.DuckDBPyConnection, pcfg: PanelConfig) -> None:
    """Allocate each loan to exactly one split by seeded hash.

    Loan-disjoint by construction, which is the Phase 2 acceptance criterion:
    a loan cannot appear on both sides of a split boundary.
    """
    alloc = pcfg.splits.loan_allocation
    cases = "\n".join(
        f"WHEN bucket >= {lo} AND bucket < {hi} THEN '{name}'"
        for name, lo, hi in alloc.boundaries()
    )
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW loan_split AS
        SELECT loan_sequence_number, bucket,
               CASE {cases} END AS assigned_split
        FROM (
            SELECT loan_sequence_number,
                   hash(loan_sequence_number || '{alloc.seed}') % 1000 AS bucket
            FROM loan_events
        );
    """)


def _register_panel(con: duckdb.DuckDBPyConnection, pcfg: PanelConfig, fcfg: FeatureConfig) -> None:
    """Quarter-end observations, 12-month forward label, split filter."""
    spec = pcfg.panel
    start, end = f"{spec.start_year}-03-01", f"{spec.end_year}-12-01"
    orig_cols = ", ".join(f"o.{c}" for c in fcfg.observation_features["origination"])
    state_cols = ", ".join(f"f.{c}" for c in fcfg.observation_features["loan_state"])
    windows = "\n".join(
        f"WHEN s.assigned_split = '{name}' AND YEAR(f.period) BETWEEN "
        f"{lo.split('-')[0]} AND {hi.split('-')[0]} THEN s.assigned_split"
        for name, (lo, hi) in pcfg.splits.windows().items()
    )
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW panel_eligible AS
        SELECT
            f.loan_sequence_number,
            f.period AS observation_date,
            {state_cols},
            {orig_cols},
            (f.period + INTERVAL {spec.horizon_months} MONTH) AS window_end,
            e.default_month, e.last_month, e.default_assistance_code,
            LEAST(COALESCE(e.prepaid_month, DATE '{FAR_FUTURE}'),
                  COALESCE(e.other_exit_month, DATE '{FAR_FUTURE}')) AS exit_month,
            e.prepaid_month,
            CASE {windows} ELSE 'unassigned' END AS split
        FROM flagged f
        JOIN loan_events e USING (loan_sequence_number)
        JOIN loan_split  s USING (loan_sequence_number)
        JOIN origination o USING (loan_sequence_number)
        WHERE e.has_defect = 0
          AND MONTH(f.period) IN (3, 6, 9, 12)
          AND f.period BETWEEN DATE '{start}' AND DATE '{end}'
          AND (e.default_month    IS NULL OR f.period < e.default_month)
          AND (e.prepaid_month    IS NULL OR f.period < e.prepaid_month)
          AND (e.other_exit_month IS NULL OR f.period < e.other_exit_month);
    """)
    # Observability: the 12-month outcome must be *seen*, not assumed. A row
    # survives only if the loan defaults in the window, leaves the book in the
    # window, or is still reported at the window end. Everything else is dropped
    # rather than imputed (build plan §7.2).
    con.execute("""
        CREATE OR REPLACE TEMP VIEW panel_labelled AS
        SELECT * EXCLUDE (default_month, last_month, exit_month, prepaid_month, window_end,
                          default_assistance_code),
            CASE WHEN default_month IS NOT NULL AND default_month <= window_end
                 THEN 1 ELSE 0 END AS default_12m,
            CASE WHEN prepaid_month IS NOT NULL AND prepaid_month <= window_end
                 THEN TRUE ELSE FALSE END AS prepaid_in_window,
            CASE WHEN default_month IS NOT NULL AND default_month <= window_end
                      AND default_assistance_code = 'F'
                 THEN TRUE ELSE FALSE END AS default_in_forbearance
        FROM panel_eligible
        WHERE (default_month IS NOT NULL AND default_month <= window_end)
           OR (exit_month <= window_end)
           OR (last_month >= window_end);
    """)


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    result = con.execute(sql).fetchone()
    return int(result[0]) if result and result[0] is not None else 0


def _register_risk_set(con: duckdb.DuckDBPyConnection, fcfg: FeatureConfig) -> None:
    """One row per loan-month while alive, truncated at the first event.

    The event month itself is retained and carries the outcome, so the hazard
    model sees the month in which default or prepayment occurred.
    """
    orig_cols = ", ".join(f"o.{c}" for c in fcfg.observation_features["origination"])
    state_cols = ", ".join(f"f.{c}" for c in fcfg.observation_features["loan_state"])
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW risk_set AS
        SELECT
            f.loan_sequence_number,
            f.period AS observation_date,
            {state_cols},
            {orig_cols},
            CASE WHEN f.is_default    THEN 'default'
                 WHEN f.is_prepaid    THEN 'prepaid'
                 WHEN f.is_other_exit THEN 'other_exit'
                 ELSE 'alive' END AS outcome
        FROM flagged f
        JOIN loan_events e USING (loan_sequence_number)
        JOIN origination o USING (loan_sequence_number)
        WHERE e.has_defect = 0
          AND f.period <= LEAST(
                COALESCE(e.default_month,    DATE '{FAR_FUTURE}'),
                COALESCE(e.prepaid_month,    DATE '{FAR_FUTURE}'),
                COALESCE(e.other_exit_month, DATE '{FAR_FUTURE}'));
    """)


def _register_sample(con: duckdb.DuckDBPyConnection, pcfg: PanelConfig, rate: float) -> None:
    """Deterministic sample, uniform in the loan/date hash.

    A uniform rate preserves each observation date's share of the panel, which
    is what `stratify_by: observation_date` requires. The hash makes it
    reproducible from the seed alone, with no sampled-key table to carry around.
    """
    seed = pcfg.sampling.seed
    keep = (
        "TRUE"
        if rate >= 1.0
        else (
            f"hash(loan_sequence_number || observation_date::VARCHAR || '{seed}') % 1000000"
            f" < {round(rate * 1_000_000)}"
        )
    )
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW panel_sampled AS
        SELECT * FROM panel_labelled
        WHERE split <> 'unassigned' AND {keep};
    """)


def _write(con: duckdb.DuckDBPyConnection, view: str, out_dir: Path) -> None:
    """Write a view to Parquet, partitioned by observation year."""
    out_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        COPY (SELECT *, YEAR(observation_date) AS observation_year FROM {view})
        TO '{out_dir.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (observation_year),
         OVERWRITE_OR_IGNORE 1);
    """)
    log.info("wrote_partitioned_parquet", view=view, path=str(out_dir))


def _default_rate_by_quarter(con: duckdb.DuckDBPyConnection, out_csv: Path) -> None:
    """The Phase 2 acceptance exhibit: default rate by observation quarter."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        COPY (
            SELECT observation_date,
                   YEAR(observation_date) AS year,
                   QUARTER(observation_date) AS quarter,
                   COUNT(*) AS n,
                   SUM(default_12m) AS n_default,
                   SUM(default_12m)::DOUBLE / COUNT(*) AS default_rate,
                   AVG(CASE WHEN prepaid_in_window THEN 1.0 ELSE 0.0 END) AS prepay_rate
            FROM panel_sampled GROUP BY 1, 2, 3 ORDER BY 1
        ) TO '{out_csv.as_posix()}' (HEADER, DELIMITER ',');
    """)
    log.info("wrote_exhibit_data", path=str(out_csv))


def _exclusions(con: duckdb.DuckDBPyConnection, rate: float) -> list[Exclusion]:
    """Count what was dropped and why (Phase 2 acceptance criterion)."""
    defect_loans = _scalar(con, "SELECT COUNT(*) FROM loan_events WHERE has_defect = 1")
    eligible = _scalar(con, "SELECT COUNT(*) FROM panel_eligible")
    observable = _scalar(con, "SELECT COUNT(*) FROM panel_labelled")
    unassigned = _scalar(con, "SELECT COUNT(*) FROM panel_labelled WHERE split = 'unassigned'")
    return [
        Exclusion(
            stage="loan",
            reason="confirmed underwriting or servicing defect (code 96); outcome unobservable, "
            "excluded rather than counted as good",
            loans=defect_loans,
        ),
        Exclusion(
            stage="panel_row",
            reason="insufficient forward observation: the 12-month outcome is not visible and is "
            "dropped, never imputed",
            rows=eligible - observable,
        ),
        Exclusion(
            stage="panel_row",
            reason="observation date falls in 2010-2014, which belongs to no split by design",
            rows=unassigned,
        ),
        Exclusion(
            stage="panel_row",
            reason=f"stratified sample at rate {rate:.4f} to reach the configured target",
            rows=round((observable - unassigned) * (1 - rate)),
        ),
    ]


def run(cfg: DataConfig | None = None) -> PanelReport:
    """Build the panel and the risk set, write both, and report what was dropped."""
    cfg = cfg or data_config()
    pcfg, fcfg = panel_config(), feature_config()
    con = duckdb.connect()

    _register_views(con, cfg, pcfg)
    _register_splits(con, pcfg)
    _register_panel(con, pcfg, fcfg)
    _register_risk_set(con, fcfg)

    in_split = _scalar(con, "SELECT COUNT(*) FROM panel_labelled WHERE split <> 'unassigned'")
    rate = min(1.0, pcfg.sampling.target / in_split) if in_split else 1.0
    _register_sample(con, pcfg, rate)
    log.info("sampling", eligible=in_split, target=pcfg.sampling.target, rate=round(rate, 6))

    _write(con, "panel_sampled", cfg.paths.panel / "panel")
    _write(con, "risk_set", cfg.paths.panel / "risk_set")
    _default_rate_by_quarter(con, Path("reports/figures/default_rate_by_quarter.csv"))
    plots.default_rate_by_quarter()

    report = PanelReport(
        observation_dates=_scalar(
            con, "SELECT COUNT(DISTINCT observation_date) FROM panel_sampled"
        ),
        panel_rows_eligible=_scalar(con, "SELECT COUNT(*) FROM panel_eligible"),
        panel_rows_observable=_scalar(con, "SELECT COUNT(*) FROM panel_labelled"),
        panel_rows_in_split=in_split,
        panel_rows_sampled=_scalar(con, "SELECT COUNT(*) FROM panel_sampled"),
        sampling_rate=rate,
        risk_set_rows=_scalar(con, "SELECT COUNT(*) FROM risk_set"),
        default_rate=float(
            con.execute("SELECT AVG(default_12m::DOUBLE) FROM panel_sampled").fetchone()[0]  # type: ignore[index]
        ),
        exclusions=_exclusions(con, rate),
    )
    manifest = cfg.paths.panel / MANIFEST_NAME
    manifest.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    log.info(
        "panel_complete",
        rows=report.panel_rows_sampled,
        risk_set_rows=report.risk_set_rows,
        default_rate=round(report.default_rate, 5),
    )
    con.close()
    return report
