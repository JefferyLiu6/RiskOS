"""Assemble the validation report and the model card, and write them."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from riskos.config import PROJECT_ROOT
from riskos.log import get_logger
from riskos.report import sections
from riskos.report.render import md_table, money, pct, ratio, squash, text
from riskos.report.sources import Sources, load_sources

log = get_logger(__name__)

REPORTS = PROJECT_ROOT / "reports"
REPORT_NAME = "validation_report.md"
CARD_NAME = "model_card.md"

SECTIONS = (
    sections.title_block,
    sections.executive_summary,
    sections.scope,
    sections.data,
    sections.development,
    sections.performance,
    sections.selection,
    sections.lifetime_pd,
    sections.lgd,
    sections.ecl,
    sections.macro,
    sections.monitoring,
    sections.governance,
    sections.findings,
    sections.assumptions,
    sections.conclusion,
    sections.reproducibility,
)


def build_report(s: Sources) -> str:
    return "\n".join(fn(s) for fn in SECTIONS)


def build_ablation_note(s: Sources) -> str:
    frame = s.csv("delinquency_ablation")
    if frame is None:
        return "# Delinquency ablation\n\nNot run. Requires the licensed panel and saved bundles.\n"
    rows = frame.filter(pl.col("split") == "oot_stress")
    blocks = [
        "# Delinquency ablation",
        "Retrospective sensitivity study on the existing evaluation splits. "
        "It is not a fresh holdout or a causal estimate of delinquency's effect.",
        "![Crisis performance with and without delinquency](figures/delinquency_ablation.png)",
        "## Crisis period: all observations",
    ]
    for group, heading in (("all", ""), ("current", "## Crisis period: current loans only")):
        if heading:
            blocks.append(heading)
        blocks.append(
            md_table(
                ["Model", "Variant", "Rows", "AUC", "Observed / expected"],
                [
                    [
                        r["model"],
                        r["variant"],
                        f"{r['n']:,}",
                        ratio(r["auc"], 4),
                        ratio(r["observed_over_expected"], 3),
                    ]
                    for r in rows.filter(pl.col("group") == group).to_dicts()
                ],
            )
        )
    blocks += [
        "## Interpretation",
        "Removing delinquency reduces whole-crisis ranking performance for both families. "
        "The current-only cohort separates ranking within performing loans from "
        "ranking across delinquency states. Both variants still under-predict crisis defaults. "
        "Other behavioural covariates remain; this is not an origination-only experiment.",
        "## Method and reproduction",
        "Baselines are reloaded fitted bundles. The scorecard variant reuses the same "
        "training-only univariate bins and repeats feature selection and coefficient fitting "
        "without current delinquency. The LightGBM variant repeats the configured grid and "
        "validation early stopping without that feature. All metrics are uncalibrated. "
        "The same observations are used for each paired comparison; other splits and "
        "30-60 DPD cohorts are in the CSV.",
        "Run `make ablate` with the licensed panel and saved baseline bundles, then "
        "`make report`. This writes separate study outputs and does not replace the main models.",
        "[All results](figures/delinquency_ablation.csv) · "
        "[Run protocol, input hashes, fitted features, and grid](figures/delinquency_ablation_manifest.json) · "
        "[Implementation](../src/riskos/models/ablation.py)",
    ]
    return "\n\n".join(blocks) + "\n"


def _performance_rows(s: Sources, label: str) -> str:
    cc = s.csv("champion_challenger_metrics")
    if cc is None:
        return sections._missing(s, "champion_challenger_metrics")
    rows = cc.filter((pl.col("model") == label) & (pl.col("calibration") == "uncalibrated"))
    return md_table(
        ["Split", "Gini", "O/E", "Score PSI"],
        [
            [
                r["split"],
                ratio(r["gini"], 3),
                ratio(r["observed_over_expected"]),
                ratio(r["score_psi"], 4) if r["score_psi"] is not None else "—",
            ]
            for r in rows.to_dicts()
        ],
    )


def build_model_card(s: Sources) -> str:
    """One card per inventoried model, in the shape a reviewer expects to read."""
    inv = s.inventory
    out = [
        "# RiskOS — Model Cards",
        "",
        f"Generated {date.today().isoformat()} from governance/model_inventory.yaml, the bundle "
        "manifests, the Phase 4 comparison table and the findings register.",
        "",
        f"> {sections.SCOPE_STATEMENT}",
        "",
    ]
    for m in inv["models"]:
        a = m["approval"]
        out += [f"## {m['id']} — {m['name']}", ""]
        facts = [
            ["Family", m["family"]],
            ["Inventory status", m["status"]],
            ["Tier", f"{m['tier']} — {squash(m['tier_rationale'])}"],
            ["Owner", m["owner"]],
            ["Clearance", a["status"]],
            [
                "Cleared on / review due",
                f"{text(a.get('approved_on'))} / {text(a.get('review_due'))}",
            ],
            ["Monitored by", ", ".join(m.get("monitored_by") or []) or "none"],
        ]
        artefact = m.get("artefact")
        manifest = s.bundle_manifest(artefact)
        if manifest:
            facts += [
                ["Version", manifest["version"]],
                ["Training window", manifest["training_window"]],
                ["Training rows", f"{int(manifest['n_train']):,}"],
                ["Features", f"{len(manifest['features'])}: {', '.join(manifest['features'])}"],
                ["Created", manifest["created_at"][:10]],
            ]
        elif artefact:
            facts.append(["Artefact", str(artefact)])
        out += [md_table(["Field", "Value"], facts), ""]
        out += [f"**Purpose.** {squash(m['purpose'])}", ""]
        out += [f"**Intended use.** {squash(m['intended_use'])}", ""]
        out += [f"**Prohibited use.** {squash(m['prohibited_use'])}", ""]
        if a.get("conditions"):
            out += [f"**Clearance conditions.** {squash(a['conditions'])}", ""]
        label = sections.FAMILY_LABEL.get(m["family"])
        if label:
            out += ["**Performance, uncalibrated.**", "", _performance_rows(s, label), ""]
        lims = [s.finding(fid) for fid in m.get("limitations") or []]
        if lims:
            out += [
                "**Limitations (findings register).**",
                "",
                md_table(
                    ["ID", "Severity", "Status", "Title"],
                    [
                        [f["id"], f["severity"], f["status"], squash(f["title"])]
                        for f in lims
                        if f is not None
                    ],
                ),
                "",
            ]
    return "\n".join(out)


def run(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    s = load_sources(root)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report = build_report(s)
    card = build_model_card(s)
    (reports / REPORT_NAME).write_text(report, encoding="utf-8")
    (reports / CARD_NAME).write_text(card, encoding="utf-8")
    if (s.figures / "delinquency_ablation.csv").exists():
        from riskos.models.plots import delinquency_ablation

        study = s.csv("delinquency_ablation")
        assert study is not None
        delinquency_ablation(study, s.figures / "delinquency_ablation.png")
        (reports / "delinquency_ablation.md").write_text(build_ablation_note(s), encoding="utf-8")
    stage = s.csv("ecl_by_stage")
    summary = {
        "report": str(reports / REPORT_NAME),
        "model_card": str(reports / CARD_NAME),
        "sections": len(SECTIONS),
        "findings": len(s.findings),
        "models": len(s.inventory["models"]),
        "missing_artefacts": list(s.missing),
        "portfolio_ecl": money(float(stage["ecl"].sum())) if stage is not None else None,
        "coverage": pct(float(stage["ecl"].sum()) / float(stage["ead"].sum()), 3)
        if stage is not None
        else None,
        "report_bytes": len(report.encode("utf-8")),
    }
    log.info("report_written", **summary)
    return summary
