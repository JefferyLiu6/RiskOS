"""Command-line entry points for the implemented RiskOS pipeline stages."""

from __future__ import annotations

from pathlib import Path

import typer

from riskos.config import data_config
from riskos.log import configure

app = typer.Typer(add_completion=False, help="RiskOS pipeline.")


@app.callback()
def main(verbose: bool = False, json_logs: bool = False) -> None:
    """Configure logging for every subcommand."""
    configure(level="DEBUG" if verbose else "INFO", json_output=json_logs)


@app.command()
def status() -> None:
    """Report what is present on disk and what blocks the next step."""
    from riskos.ingest import discover as disco

    cfg = data_config()
    report = disco.discover(cfg)
    expected = cfg.vintages.all
    missing = report.missing_years(expected)
    typer.echo(f"raw dir:        {cfg.paths.raw}")
    typer.echo(f"vintages:       {len(report.complete_years)}/{len(expected)} complete")
    if missing:
        typer.echo(f"missing:        {', '.join(str(y) for y in missing)}")
    typer.echo(f"layout version: {cfg.layout.version or 'NOT RECORDED — see User Guide'}")
    typer.echo(f"zbc mapping:    {'populated' if cfg.zero_balance_codes.is_ready else 'EMPTY'}")


@app.command()
def ingest() -> None:
    """Phase 1 — parse raw SFLLD files to validated Parquet."""
    from riskos.ingest.run import run

    report = run()
    typer.echo(f"ingested {len(report.vintages)} vintages, {report.total_rows:,} rows")


@app.command()
def macro(refresh: bool = False) -> None:
    """Phase 1 — pull and cache FRED macro series. Uses the cache unless --refresh."""
    from riskos.ingest.macro import pull

    frames = pull(data_config(), refresh=refresh)
    typer.echo(
        f"{len(frames)} macro series available ({sum(f.height for f in frames.values()):,} observations)"
    )


@app.command()
def panel() -> None:
    """Phase 2 — observation panel, monthly risk set, labels."""
    from riskos.panel.build import run

    report = run()
    typer.echo(
        f"panel {report.panel_rows_sampled:,} rows "
        f"(default rate {report.default_rate:.3%}), "
        f"risk set {report.risk_set_rows:,} rows"
    )


@app.command()
def features() -> None:
    """Phase 3 — binning, WOE, IV screening (reported, not persisted)."""
    import polars as pl

    from riskos.features import binning

    train_df = pl.read_parquet("data/panel/panel/*/*.parquet").filter(pl.col("split") == "train")
    process, names = binning.fit(train_df)
    for f in binning.information_values(process, names):
        mark = "keep" if f.selected else "drop"
        typer.echo(f"{mark:>4}  {f.name:<38} IV {f.iv:7.4f}  {f.band:<20} {f.reason}")


@app.command()
def train() -> None:
    """Phase 3 — fit the WOE scorecard champion candidate."""
    from riskos.models.train_scorecard import run

    metrics = run()
    typer.echo(metrics.select(["split", "n", "auc", "gini", "ks", "observed_over_expected"]))


@app.command()
def challenger() -> None:
    """Phase 4 — fit the LightGBM challenger, calibrate both, apply the rubric."""
    from riskos.models.train_challenger import run

    comparison = run()
    typer.echo(comparison.head(20))


@app.command()
def evaluate() -> None:
    """Phase 3/4 — show ranking, calibration, and drift from saved results."""
    import polars as pl

    path = "reports/figures/champion_challenger_metrics.csv"
    if not Path(path).exists():
        typer.echo("no comparison found; run `riskos train` then `riskos challenger`", err=True)
        raise typer.Exit(code=2)
    comparison = (
        pl.read_csv(path)
        .filter(pl.col("calibration") == "uncalibrated")
        .select("model", "split", "auc", "observed_over_expected", "score_psi", "psi_band")
        .rename({"observed_over_expected": "O/E"})
    )
    typer.echo("Saved results, uncalibrated models (no training run).")
    typer.echo("AUC: ranking; O/E: observed / expected defaults (ideal 1); PSI: score drift.")
    with pl.Config(tbl_cols=6, tbl_width_chars=100, fmt_str_lengths=24, float_precision=4):
        typer.echo(comparison)
    typer.echo(f"Full metrics and calibration variants: {path}")


@app.command()
def hazard() -> None:
    """Phase 5 — discrete-time hazard with competing risks, and reconciliation."""
    from riskos.models.train_hazard import run

    out = run()
    r = out["reconciliation"]
    typer.echo(
        f"hazard-implied 12m PD {r['hazard_implied_pd_12m']:.4%} vs panel observed "
        f"{r['panel_observed_default_rate_12m']:.4%} (ratio {r['ratio_hazard_to_observed']:.3f})"
    )


@app.command()
def ecl() -> None:
    """Phase 5 — LGD, EAD, staging, ECL, scenarios, macro backtest."""
    from riskos.ecl.run import run

    report = run()
    typer.echo(
        f"portfolio ECL ${report['total_ecl'] / 1e6:,.2f}M on "
        f"${report['total_ead'] / 1e9:,.2f}B "
        f"({report['coverage']:.4%}); probability-weighted "
        f"${report['probability_weighted_ecl'] / 1e6:,.2f}M"
    )


@app.command()
def monitor() -> None:
    """Phase 6 — PSI/CSI, calibration decay, alert records."""
    from riskos.monitor.run import run

    report = run()
    typer.echo(f"{report['windows']} windows, {report['rules_evaluated']} rules")
    for entry in report["models"]:
        gap = entry["detection_gap"]
        typer.echo(
            f"\n{entry['model_id']} ({entry['model_family']}): "
            f"{entry['alerts_fired']} alerts, {entry['breaches']} breaches, "
            f"worst stress score PSI {entry['max_stress_score_psi']:.4f}"
        )
        if gap["months_of_warning"] is None:
            typer.echo("  no leading/lagging breach pair; detection gap not measurable")
        else:
            typer.echo(
                f"  leading {gap['leading_metric']} breached {gap['leading_window']} "
                f"(knowable {gap['leading_detectable_at']}); lagging "
                f"{gap['lagging_metric']} breached {gap['lagging_window']} "
                f"(knowable {gap['lagging_detectable_at']}) "
                f"- {gap['months_of_warning']} months of warning"
            )
        if gap["caveat"]:
            typer.echo(f"  caveat: {gap['caveat']}")
        for rule_id, rate in sorted(entry["rule_false_positive_rates"].items()):
            if rate is not None and rate > 0.25:
                typer.echo(f"  {rule_id}: fires on {rate:.0%} of development-sample windows")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Phase 6 — serve the inventoried, approved PD model over HTTP."""
    import uvicorn

    # Fail here, with the inventory's reason, rather than inside the server's
    # startup where the message would be buried in a traceback.
    from riskos.serve.scoring import ServingError, load_served_model

    try:
        model = load_served_model()
    except ServingError as exc:
        typer.echo(f"refusing to serve: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"serving {model.manifest.model_id} v{model.manifest.version} on {host}:{port}")
    uvicorn.run("riskos.serve.app:app", host=host, port=port, log_level="info")


@app.command()
def registry() -> None:
    """Phase 6 — reconcile the model inventory against the artefacts on disk."""
    import polars as pl

    from riskos.registry import inventory, reconcile, summary

    inv = inventory()
    discrepancies = reconcile()
    typer.echo(
        f"{len(inv.models)} models inventoried, {len(inv.deployed)} deployed, "
        f"{len(discrepancies)} discrepancies"
    )
    frame = summary(discrepancies)
    Path("governance").mkdir(parents=True, exist_ok=True)
    frame.write_csv("governance/inventory_reconciliation.csv")
    with pl.Config(fmt_str_lengths=200, tbl_width_chars=200):
        for row in frame.to_dicts():
            typer.echo(f"  [{row['severity']:>6}] {row['check']}: {row['detail']}")
    if any(d.severity == "high" for d in discrepancies):
        typer.echo(
            "\nhigh-severity discrepancies mean the inventory does not describe what is on disk",
            err=True,
        )
        raise typer.Exit(code=1)


@app.command()
def report() -> None:
    """Phase 7 — validation report and model card, generated from the artefacts."""
    from riskos.report import run

    summary = run()
    typer.echo(f"wrote {summary['report']} ({summary['report_bytes']:,} bytes)")
    typer.echo(f"wrote {summary['model_card']}")
    typer.echo(
        f"{summary['sections']} sections, {summary['findings']} findings, "
        f"{summary['models']} models; portfolio ECL {summary['portfolio_ecl']} "
        f"({summary['coverage']} coverage)"
    )
    if summary["missing_artefacts"]:
        typer.echo(
            f"artefacts missing: {', '.join(summary['missing_artefacts'])} (named in Appendix A)",
            err=True,
        )


if __name__ == "__main__":  # pragma: no cover
    app()
