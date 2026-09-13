"""Reconcile the inventory against the artefacts and the findings register.

Each check below exists because there is a specific way an inventory goes wrong,
and every one of them is a way it goes wrong in the flattering direction: the
list says the estate is in better condition than it is. So the checks are
written to find the gap, not to confirm the list.

Severity here is about the inventory, not about the model. A high-severity
discrepancy means the record cannot be relied on to describe what is running.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import polars as pl

from riskos.config import PROJECT_ROOT, load_yaml
from riskos.log import get_logger
from riskos.models.persistence import BundleError, load_manifest
from riskos.monitor.config import MonitoringConfig, monitoring_config
from riskos.registry.inventory import InventoryConfig, ModelEntry, inventory

log = get_logger(__name__)

Severity = Literal["low", "medium", "high"]
SEVERITY_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}

MODELS_DIR = PROJECT_ROOT / "models"
FINDINGS_FILE = PROJECT_ROOT / "governance" / "findings_register.yaml"

# Directories under models/ that are model bundles rather than loose artefacts.
BUNDLE_MARKER = "manifest.json"


@dataclass(frozen=True)
class Discrepancy:
    """One place the inventory and reality disagree."""

    check: str
    model_id: str | None
    severity: Severity
    detail: str
    remediation: str


def _open_findings() -> dict[str, dict[str, object]]:
    register = load_yaml(FINDINGS_FILE)["findings"]
    return {f["id"]: f for f in register if f.get("status") == "open"}


def _bundles_on_disk(models_dir: Path) -> dict[str, Path]:
    """Every loadable bundle, keyed by the model id its own manifest declares."""
    found: dict[str, Path] = {}
    if not models_dir.exists():
        return found
    for path in sorted(models_dir.iterdir()):
        if path.is_dir() and (path / BUNDLE_MARKER).exists():
            try:
                found[load_manifest(path).model_id] = path
            except BundleError as exc:  # a malformed bundle is itself a finding
                log.warning("unreadable_bundle", path=str(path), error=str(exc))
    return found


def _check_artefact_exists(entry: ModelEntry, root: Path) -> list[Discrepancy]:
    if entry.artefact is None:
        if entry.is_deployed:  # unreachable via validation, kept as a belt
            return [
                Discrepancy(
                    check="artefact_missing",
                    model_id=entry.id,
                    severity="high",
                    detail=f"{entry.id} is {entry.status} but names no artefact",
                    remediation="Persist a scoring-ready artefact or change the status.",
                )
            ]
        return []
    path = root / entry.artefact
    if path.exists():
        return []
    return [
        Discrepancy(
            check="artefact_missing",
            model_id=entry.id,
            severity="high" if entry.is_deployed else "medium",
            detail=f"{entry.id} names {entry.artefact} which is not on disk",
            remediation=(
                "Re-run the phase that produces the artefact, or correct the inventory. "
                "A model recorded as in use with nothing to load cannot be scored, "
                "reproduced or reviewed."
            ),
        )
    ]


def _check_selected_model_is_deployable(inv: InventoryConfig, root: Path) -> list[Discrepancy]:
    """The model the rubric selected must be the model that can actually run.

    This is the check that catches a selection exercise whose conclusion never
    reached an artefact: the comparison is recorded, the winner is named, and
    the only thing on disk that can be loaded and scored is the model that lost.
    """
    record_path = root / "models" / "selection_record.json"
    if not record_path.exists():
        return []
    import json

    selected_family = json.loads(record_path.read_text(encoding="utf-8")).get("selected")
    matching = [m for m in inv.models if m.family == selected_family]
    if not matching:
        return [
            Discrepancy(
                check="selected_model_not_inventoried",
                model_id=None,
                severity="high",
                detail=f"the selection record names {selected_family!r}, which is not in the inventory",
                remediation="Add the selected model to the inventory or correct the record.",
            )
        ]
    entry = matching[0]
    if entry.artefact is not None and (root / entry.artefact).exists():
        return []
    return [
        Discrepancy(
            check="selected_model_not_deployable",
            model_id=entry.id,
            severity="high",
            detail=(
                f"the Phase 4 rubric selected {selected_family!r} ({entry.id}) but no "
                "scoring-ready artefact exists for it, so every downstream figure is "
                "produced by a model the rubric did not select"
            ),
            remediation=(
                "Persist a bundle for the selected model and re-run the downstream "
                "phases against it, or record explicitly that the runner-up is "
                "deployed and why."
            ),
        )
    ]


def _check_shadow_artefacts(inv: InventoryConfig, bundles: dict[str, Path]) -> list[Discrepancy]:
    """A loadable model nobody inventoried is the classic shadow model."""
    known = {m.id for m in inv.models}
    return [
        Discrepancy(
            check="uninventoried_artefact",
            model_id=model_id,
            severity="high",
            detail=f"{path} declares model id {model_id!r}, which is not in the inventory",
            remediation=(
                "Add it to the inventory or delete it. A model that can be loaded and "
                "scored, with no owner and no recorded intended use, is exactly what an "
                "inventory exists to prevent."
            ),
        )
        for model_id, path in bundles.items()
        if model_id not in known
    ]


def _check_config_drift(entry: ModelEntry, bundles: dict[str, Path]) -> list[Discrepancy]:
    """A bundle fitted against configuration that has since changed."""
    path = bundles.get(entry.id)
    if path is None:
        return []
    from riskos.models.persistence import config_fingerprint

    manifest = load_manifest(path)
    current = config_fingerprint(*manifest.config_fingerprint)
    drifted = sorted(
        name for name, digest in manifest.config_fingerprint.items() if current.get(name) != digest
    )
    if not drifted:
        return []
    return [
        Discrepancy(
            check="config_drift",
            model_id=entry.id,
            severity="medium",
            detail=f"{entry.id} was fitted against a different {', '.join(drifted)}",
            remediation=(
                "Re-fit, or record why the change does not invalidate the fit. A changed "
                "threshold does not necessarily invalidate a fitted model, but it does "
                "mean the model on disk was not built against what is in the repository."
            ),
        )
    ]


def _check_findings_are_recorded(entry: ModelEntry) -> list[Discrepancy]:
    """Every limitation a model cites must be a finding that actually exists."""
    out = []
    known = {f["id"] for f in load_yaml(FINDINGS_FILE)["findings"]}
    for ref in entry.limitations:
        if ref not in known:
            out.append(
                Discrepancy(
                    check="unknown_finding_reference",
                    model_id=entry.id,
                    severity="medium",
                    detail=f"{entry.id} cites {ref}, which is not in the findings register",
                    remediation=(
                        "Correct the reference. A citation to a finding that does not "
                        "exist reads as evidence that a limitation was documented when "
                        "it was not."
                    ),
                )
            )
    return out


def _check_high_severity_findings_are_disclosed(
    entry: ModelEntry, open_findings: dict[str, dict[str, object]]
) -> list[Discrepancy]:
    """An open high-severity finding must appear in the model's limitations.

    Deliberately does not attempt to guess which model a finding belongs to.
    It reports the ones that name the model explicitly, because a heuristic that
    silently mis-assigns findings would produce a reconciliation that looks
    thorough and is wrong.
    """
    out = []
    for fid, finding in open_findings.items():
        if finding.get("severity") not in ("high", "critical"):
            continue
        text = f"{finding.get('title', '')} {finding.get('description', '')}"
        if entry.id in text and fid not in entry.limitations:
            out.append(
                Discrepancy(
                    check="undisclosed_finding",
                    model_id=entry.id,
                    severity="high",
                    detail=f"{fid} is open, high severity, and names {entry.id} but is not "
                    "in its recorded limitations",
                    remediation="Add the finding to the model's limitations.",
                )
            )
    return out


def _check_monitoring_covers_tier_one(
    entry: ModelEntry, cfg: MonitoringConfig
) -> list[Discrepancy]:
    known_rules = {r.id for r in cfg.rules}
    if entry.tier == 1 and entry.is_deployed and not entry.monitored_by:
        return [
            Discrepancy(
                check="tier_one_unmonitored",
                model_id=entry.id,
                severity="medium",
                detail=(
                    f"{entry.id} is tier 1 and {entry.status} but no monitoring rule is "
                    "pointed at it"
                ),
                remediation=(
                    "Add rules to conf/monitoring.yaml covering it, or record in the "
                    "approval conditions why its tier does not require them. The gap is "
                    "reported rather than left implicit either way."
                ),
            )
        ]
    unknown = sorted(set(entry.monitored_by) - known_rules)
    if unknown:
        return [
            Discrepancy(
                check="unknown_monitoring_rule",
                model_id=entry.id,
                severity="low",
                detail=f"{entry.id} cites monitoring rules that do not exist: {unknown}",
                remediation="Correct the rule ids or add the rules.",
            )
        ]
    return []


def _check_approval_is_current(entry: ModelEntry, as_at: date) -> list[Discrepancy]:
    if not entry.approval.is_lapsed(as_at):
        return []
    return [
        Discrepancy(
            check="approval_lapsed",
            model_id=entry.id,
            severity="high" if entry.is_deployed else "medium",
            detail=(
                f"{entry.id} was due for review on {entry.approval.review_due} and is "
                f"still recorded as {entry.approval.status}"
            ),
            remediation="Re-review and re-approve, or move the model out of use.",
        )
    ]


def reconcile(
    inv: InventoryConfig | None = None,
    cfg: MonitoringConfig | None = None,
    *,
    root: Path = PROJECT_ROOT,
    as_at: date | None = None,
) -> list[Discrepancy]:
    """Every disagreement between the inventory, the artefacts, and the findings."""
    inv = inv or inventory()
    cfg = cfg or monitoring_config()
    as_at = as_at or date.today()
    bundles = _bundles_on_disk(root / "models")
    open_findings = _open_findings()

    out: list[Discrepancy] = []
    out += _check_shadow_artefacts(inv, bundles)
    out += _check_selected_model_is_deployable(inv, root)
    for entry in inv.models:
        out += _check_artefact_exists(entry, root)
        out += _check_config_drift(entry, bundles)
        out += _check_findings_are_recorded(entry)
        out += _check_high_severity_findings_are_disclosed(entry, open_findings)
        out += _check_monitoring_covers_tier_one(entry, cfg)
        out += _check_approval_is_current(entry, as_at)

    out.sort(key=lambda d: (-SEVERITY_ORDER[d.severity], d.model_id or "", d.check))
    log.info(
        "inventory_reconciled",
        models=len(inv.models),
        bundles=len(bundles),
        discrepancies=len(out),
        high=sum(d.severity == "high" for d in out),
    )
    return out


def summary(discrepancies: list[Discrepancy]) -> pl.DataFrame:
    schema = pl.Schema(
        {
            "check": pl.Utf8,
            "model_id": pl.Utf8,
            "severity": pl.Utf8,
            "detail": pl.Utf8,
            "remediation": pl.Utf8,
        }
    )
    if not discrepancies:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame([asdict(d) for d in discrepancies], schema=schema)
