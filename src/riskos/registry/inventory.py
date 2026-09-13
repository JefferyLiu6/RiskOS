"""Typed model inventory.

The inventory holds judgements and nothing else. Everything derivable from an
artefact — feature list, training window, row counts, config fingerprints — is
read from the artefact at reconciliation time and deliberately not copied here,
because a copied fact is a fact that can go stale without anyone noticing.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from riskos.config import PROJECT_ROOT, Strict, load_yaml

GOVERNANCE_DIR = PROJECT_ROOT / "governance"
INVENTORY_FILE = "model_inventory.yaml"

Status = Literal["development", "candidate", "approved", "in_use", "retired"]
ApprovalStatus = Literal["not_approved", "approved", "approved_with_conditions", "approval_lapsed"]

# Statuses that assert the model is being relied on, and therefore must have a
# loadable artefact and a monitoring route.
DEPLOYED = ("approved", "in_use")


class Approval(Strict):
    status: ApprovalStatus
    approved_on: date | None = None
    approved_by: str | None = None
    review_due: date | None = None
    conditions: str = ""

    @model_validator(mode="after")
    def _an_approval_names_who_and_when(self) -> Approval:
        if self.status in ("approved", "approved_with_conditions"):
            if not (self.approved_on and self.approved_by):
                raise ValueError("an approval must record who approved it and when")
            if not self.review_due:
                raise ValueError(
                    "an approval with no review date never expires, which is not an "
                    "approval but a permanent exemption"
                )
        if self.status == "approved_with_conditions" and not self.conditions.strip():
            raise ValueError("approval_with_conditions must state the conditions")
        return self

    def is_lapsed(self, as_at: date) -> bool:
        return self.review_due is not None and self.review_due < as_at


class ModelEntry(Strict):
    id: str
    name: str
    purpose: str
    family: str
    status: Status
    owner: str
    tier: Literal[1, 2, 3]
    tier_rationale: str
    intended_use: str
    prohibited_use: str
    artefact: Path | None = None
    monitored_by: tuple[str, ...] = ()
    approval: Approval
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _a_deployed_model_is_approved_and_has_an_artefact(self) -> ModelEntry:
        if self.status in DEPLOYED:
            if self.approval.status == "not_approved":
                raise ValueError(f"{self.id} is recorded as {self.status} but not approved")
            if self.artefact is None:
                raise ValueError(
                    f"{self.id} is recorded as {self.status} but names no artefact; a model "
                    "in use that cannot be loaded is not in use, it is a claim"
                )
        return self

    @property
    def is_deployed(self) -> bool:
        return self.status in DEPLOYED


class TieringRubric(Strict):
    dimensions: dict[str, str]
    mapping: dict[str, str]


class InventorySchema(Strict):
    required_fields: tuple[str, ...]
    statuses: tuple[str, ...]
    tiers: tuple[int, ...]


class InventoryConfig(Strict):
    schema_: InventorySchema
    tiering_rubric: TieringRubric
    models: tuple[ModelEntry, ...]

    @model_validator(mode="after")
    def _ids_are_unique(self) -> InventoryConfig:
        ids = [m.id for m in self.models]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate model ids in the inventory: {sorted(ids)}")
        return self

    def by_id(self, model_id: str) -> ModelEntry | None:
        return next((m for m in self.models if m.id == model_id), None)

    @property
    def deployed(self) -> tuple[ModelEntry, ...]:
        return tuple(m for m in self.models if m.is_deployed)


@lru_cache(maxsize=1)
def inventory(governance_dir: Path = GOVERNANCE_DIR) -> InventoryConfig:
    raw = load_yaml(governance_dir / INVENTORY_FILE)
    # `schema` is a pydantic-reserved name on the model, so it is renamed on the
    # way in rather than renaming the key a reviewer reads in the YAML.
    raw["schema_"] = raw.pop("schema")
    return InventoryConfig.model_validate(raw)
