"""Phase 6 — the model inventory and its reconciliation.

The inventory is the easy artefact to fake: a list of models is a list. These
tests are about the reconciliation, because that is the part that can tell the
difference between a list and a record of what is actually running. Each test
pins a way the record can flatter the estate.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
import yaml

from riskos.registry import inventory, reconcile, summary
from riskos.registry.inventory import InventoryConfig, ModelEntry

REAL = inventory()


def _entry(**overrides: object) -> ModelEntry:
    base: dict[str, object] = {
        "id": "T_001",
        "name": "test model",
        "purpose": "testing",
        "family": "toy",
        "status": "in_use",
        "owner": "someone",
        "tier": 1,
        "tier_rationale": "because",
        "intended_use": "tests",
        "prohibited_use": "anything else",
        "artefact": "models/scorecard_bundle",
        "monitored_by": ["MON-01"],
        "approval": {
            "status": "approved",
            "approved_on": date(2026, 1, 1),
            "approved_by": "someone",
            "review_due": date(2099, 1, 1),
        },
        "limitations": [],
    }
    return ModelEntry.model_validate(base | overrides)


def _config(*entries: ModelEntry) -> InventoryConfig:
    return InventoryConfig.model_validate(
        {
            "schema_": REAL.schema_.model_dump(),
            "tiering_rubric": REAL.tiering_rubric.model_dump(),
            "models": [e.model_dump() for e in entries],
        }
    )


# --- the inventory refuses records that cannot be true ------------------------


def test_a_model_in_use_with_no_artefact_is_rejected() -> None:
    """A model in use that cannot be loaded is not in use, it is a claim."""
    with pytest.raises(ValueError, match="names no artefact"):
        _entry(artefact=None)


def test_a_model_in_use_that_is_not_approved_is_rejected() -> None:
    with pytest.raises(ValueError, match="not approved"):
        _entry(approval={"status": "not_approved"})


def test_an_approval_with_no_review_date_is_rejected() -> None:
    """An approval that never expires is a permanent exemption, not an approval."""
    with pytest.raises(ValueError, match="never expires"):
        _entry(
            approval={
                "status": "approved",
                "approved_on": date(2026, 1, 1),
                "approved_by": "someone",
            }
        )


def test_an_approval_must_name_who_approved_it() -> None:
    with pytest.raises(ValueError, match="who approved it"):
        _entry(
            approval={
                "status": "approved",
                "approved_on": date(2026, 1, 1),
                "review_due": date(2099, 1, 1),
            }
        )


def test_conditional_approval_must_state_its_conditions() -> None:
    with pytest.raises(ValueError, match="state the conditions"):
        _entry(
            approval={
                "status": "approved_with_conditions",
                "approved_on": date(2026, 1, 1),
                "approved_by": "someone",
                "review_due": date(2099, 1, 1),
            }
        )


def test_duplicate_model_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate model ids"):
        _config(_entry(), _entry())


# --- reconciliation finds the gaps -------------------------------------------


def _checks(discrepancies: list) -> set[str]:
    return {d.check for d in discrepancies}


def test_a_missing_artefact_is_reported(tmp_path: Path) -> None:
    cfg = _config(_entry(artefact="models/not_here"))

    found = reconcile(cfg, root=tmp_path, as_at=date(2026, 1, 1))

    assert "artefact_missing" in _checks(found)
    assert any(d.severity == "high" for d in found if d.check == "artefact_missing")


def test_a_lapsed_approval_is_reported(tmp_path: Path) -> None:
    (tmp_path / "models" / "scorecard_bundle").mkdir(parents=True)
    cfg = _config(
        _entry(
            approval={
                "status": "approved",
                "approved_on": date(2020, 1, 1),
                "approved_by": "someone",
                "review_due": date(2021, 1, 1),
            }
        )
    )

    found = reconcile(cfg, root=tmp_path, as_at=date(2026, 1, 1))

    assert "approval_lapsed" in _checks(found)


def test_a_tier_one_model_with_no_monitoring_rule_is_reported(tmp_path: Path) -> None:
    (tmp_path / "models" / "scorecard_bundle").mkdir(parents=True)
    cfg = _config(_entry(monitored_by=[]))

    found = reconcile(cfg, root=tmp_path, as_at=date(2026, 1, 1))

    assert "tier_one_unmonitored" in _checks(found)


def test_a_citation_to_a_finding_that_does_not_exist_is_reported(tmp_path: Path) -> None:
    """Worse than no citation: it reads as evidence a limitation was documented."""
    (tmp_path / "models" / "scorecard_bundle").mkdir(parents=True)
    cfg = _config(_entry(limitations=["F-999"]))

    found = reconcile(cfg, root=tmp_path, as_at=date(2026, 1, 1))

    assert "unknown_finding_reference" in _checks(found)


def test_a_loadable_model_nobody_inventoried_is_reported(tmp_path: Path) -> None:
    """The shadow model: scoreable, with no owner and no recorded intended use."""
    bundle = tmp_path / "models" / "ghost"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": "GHOST_001",
                "model_family": "toy",
                "version": "1.0.0",
                "created_at": "2026-01-01T00:00:00+00:00",
                "features": [],
                "training_window": "x",
                "n_train": 1,
                "components": [],
                "config_fingerprint": {},
                "calibration_method": None,
                "format": "riskos-model-bundle/1",
            }
        ),
        encoding="utf-8",
    )
    cfg = _config(_entry(artefact="models/ghost"))

    found = reconcile(cfg, root=tmp_path, as_at=date(2026, 1, 1))

    assert "uninventoried_artefact" in _checks(found)


def test_a_selected_model_with_no_artefact_is_reported(tmp_path: Path) -> None:
    """The defect F-018 recorded: a selection whose conclusion never shipped."""
    models = tmp_path / "models"
    models.mkdir(parents=True)
    (models / "selection_record.json").write_text(json.dumps({"selected": "toy"}), encoding="utf-8")
    cfg = _config(_entry(status="candidate", artefact=None, approval={"status": "not_approved"}))

    found = reconcile(cfg, root=tmp_path, as_at=date(2026, 1, 1))

    assert "selected_model_not_deployable" in _checks(found)
    assert any(d.severity == "high" for d in found)


def test_a_clean_estate_produces_no_discrepancies(tmp_path: Path) -> None:
    (tmp_path / "models" / "scorecard_bundle").mkdir(parents=True)

    found = reconcile(_config(_entry()), root=tmp_path, as_at=date(2026, 1, 1))

    assert found == []
    assert summary(found).is_empty()


def test_discrepancies_sort_most_severe_first(tmp_path: Path) -> None:
    cfg = _config(_entry(artefact="models/gone", monitored_by=[]))

    frame = summary(reconcile(cfg, root=tmp_path, as_at=date(2026, 1, 1)))

    assert frame["severity"][0] == "high"


# --- the committed inventory ---------------------------------------------------


def test_the_committed_inventory_loads_and_every_model_has_an_owner() -> None:
    for model in REAL.models:
        assert model.owner
        assert model.intended_use.strip()
        assert model.prohibited_use.strip()
        assert model.tier_rationale.strip()


def test_every_required_field_named_in_the_schema_is_present_on_every_model() -> None:
    raw = yaml.safe_load(Path("governance/model_inventory.yaml").read_text(encoding="utf-8"))

    for entry in raw["models"]:
        missing = [f for f in raw["schema"]["required_fields"] if f not in entry]
        assert not missing, f"{entry.get('id')} is missing {missing}"


def test_the_real_estate_has_no_high_severity_discrepancies() -> None:
    """The check that would have caught F-018, run against the live repository.

    Kept as a regression test rather than a one-off: any future model whose
    selection outruns its persistence, or whose artefact goes missing, fails
    here rather than being noticed later.
    """
    high = [d for d in reconcile() if d.severity == "high"]

    assert not high, f"high-severity inventory discrepancies: {[d.detail for d in high]}"
