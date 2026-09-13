"""Model loading gated on the inventory, and the scoring path itself.

Kept free of FastAPI so the gating and the scoring can be tested without an HTTP
client, and so the same code path serves a CLI or a batch job unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import numpy.typing as npt
import polars as pl

from riskos.config import PROJECT_ROOT
from riskos.features import binning
from riskos.log import get_logger
from riskos.models.persistence import (
    BundleError,
    BundleManifest,
    config_fingerprint,
    load_manifest,
)
from riskos.models.scorecard import Scorecard
from riskos.registry.inventory import InventoryConfig, ModelEntry, inventory

log = get_logger(__name__)

# The one model family this service knows how to score. The challenger has a
# bundle too, but it is `candidate` in the inventory and would be refused by the
# gate below regardless; when it is approved, a loader is added here and the
# inventory decides which one is served.
SERVED_FAMILY = "woe_scorecard"
TOP_DRIVERS = 5


class ServingError(RuntimeError):
    """The inventory, the approval, or the artefact does not permit serving."""


@dataclass(frozen=True)
class Governance:
    """What a consumer of a score is told about where it came from."""

    model_id: str
    version: str
    family: str
    training_window: str
    inventory_status: str
    tier: int
    approval_status: str
    approved_on: date | None
    review_due: date | None
    conditions: str
    limitations: tuple[str, ...]
    config_drift: tuple[str, ...]


@dataclass
class ServedModel:
    """A loaded, approved model plus everything needed to score a request."""

    card: Scorecard
    process: Any
    manifest: BundleManifest
    entry: ModelEntry
    loaded_at: datetime
    config_drift: tuple[str, ...]
    # Fitted variable set of the binning process, and which of those are
    # categorical. optbinning needs every fitted variable present in the input,
    # so a request carrying only the model's features is padded with nulls
    # for the rest; the model's own features never read those columns.
    fitted_variables: tuple[str, ...] = field(default_factory=tuple)
    categorical: frozenset[str] = field(default_factory=frozenset)

    @property
    def features(self) -> list[str]:
        return list(self.manifest.features)

    def governance(self) -> Governance:
        a = self.entry.approval
        return Governance(
            model_id=self.manifest.model_id,
            version=self.manifest.version,
            family=self.manifest.model_family,
            training_window=self.manifest.training_window,
            inventory_status=self.entry.status,
            tier=self.entry.tier,
            approval_status=a.status,
            approved_on=a.approved_on,
            review_due=a.review_due,
            conditions=a.conditions.strip(),
            limitations=tuple(self.entry.limitations),
            config_drift=self.config_drift,
        )


def _select_entry(inv: InventoryConfig) -> ModelEntry:
    """The single PD model the inventory says is in use. Anything else refuses."""
    candidates = [m for m in inv.models if m.family == SERVED_FAMILY and m.status == "in_use"]
    if not candidates:
        raise ServingError(
            f"no {SERVED_FAMILY} model is recorded as in_use in the inventory; a model that "
            "is not in use is not served"
        )
    if len(candidates) > 1:
        raise ServingError(
            f"inventory records more than one {SERVED_FAMILY} model as in_use: "
            f"{[m.id for m in candidates]}; the service will not guess"
        )
    return candidates[0]


def _require_current_approval(entry: ModelEntry, as_at: date) -> None:
    if entry.approval.status not in ("approved", "approved_with_conditions"):
        raise ServingError(
            f"{entry.id} has approval status {entry.approval.status!r}; a model is served "
            "because second line signed an approval, not because it has a bundle"
        )
    if entry.approval.is_lapsed(as_at):
        raise ServingError(
            f"{entry.id} was due for review on {entry.approval.review_due} and has not been "
            "re-approved; a lapsed approval is not an approval"
        )


def load_served_model(
    inv: InventoryConfig | None = None,
    *,
    root: Path = PROJECT_ROOT,
    as_at: date | None = None,
) -> ServedModel:
    """Load the model the inventory says is in use, or refuse with a reason."""
    inv = inv or inventory()
    as_at = as_at or date.today()
    entry = _select_entry(inv)
    _require_current_approval(entry, as_at)

    assert entry.artefact is not None  # guaranteed by inventory validation for in_use
    directory = root / entry.artefact
    try:
        manifest = load_manifest(directory)
    except BundleError as exc:
        raise ServingError(f"{entry.id}: {exc}") from exc
    if manifest.model_id != entry.id:
        raise ServingError(
            f"inventory entry {entry.id} names {entry.artefact}, whose manifest declares "
            f"{manifest.model_id!r}; the record and the artefact disagree about what this is"
        )

    card, process, _ = Scorecard.from_bundle(directory)
    current = config_fingerprint(*manifest.config_fingerprint)
    drift = tuple(sorted(k for k, v in manifest.config_fingerprint.items() if current.get(k) != v))
    served = ServedModel(
        card=card,
        process=process,
        manifest=manifest,
        entry=entry,
        loaded_at=datetime.now(UTC),
        config_drift=drift,
        fitted_variables=tuple(process.variable_names),
        categorical=frozenset(process.categorical_variables or ()),
    )
    log.info(
        "serving_model_loaded",
        model_id=manifest.model_id,
        version=manifest.version,
        inventory_status=entry.status,
        approval=entry.approval.status,
        review_due=str(entry.approval.review_due),
        config_drift=list(drift) or "none",
    )
    return served


@dataclass(frozen=True)
class ScoreResult:
    request_id: str
    pd_12m: list[float]
    score: list[float]
    principal_drivers: list[list[dict[str, float | str]]]


def _frame(model: ServedModel, rows: list[dict[str, Any]]) -> pl.DataFrame:
    """A polars frame in the shape the binning process was fitted on.

    Missing features are refused; null features are accepted. The distinction
    is deliberate: the scorecard treats a missing value as its own bin, so a
    null is a legitimate input carrying information. A feature absent from the
    request altogether is a malformed request, and silently nulling it would
    score the loan as if the field had been checked and found empty.
    """
    for i, row in enumerate(rows):
        missing = [f for f in model.features if f not in row]
        if missing:
            raise ServingError(f"loan {i}: request is missing required features {missing}")

    columns: dict[str, pl.Series] = {}
    for name in model.fitted_variables:
        values = [row.get(name) for row in rows]
        dtype = pl.String if name in model.categorical else pl.Float64
        columns[name] = pl.Series(name, values, dtype=dtype, strict=False)
    # The transform reads a label column it never uses for scoring.
    columns["default_12m"] = pl.Series("default_12m", [0] * len(rows), dtype=pl.Int32)
    return pl.DataFrame(columns)


def score(model: ServedModel, rows: list[dict[str, Any]]) -> ScoreResult:
    """Score one or more loans. Explanations are per loan, never per batch.

    The principal drivers are centred on the training-population baseline
    stored in the bundle, so a loan scored alone reports exactly the drivers it
    would report in a batch of a thousand (defect R-003). That property is
    asserted in the tests rather than assumed.
    """
    if not rows:
        raise ServingError("nothing to score")
    request_id = uuid4().hex[:12]
    frame = _frame(model, rows)
    woe, _ = binning.transform(model.process, frame, model.features)
    pd_12m: npt.NDArray[np.float64] = model.card.predict_proba(woe)
    points = model.card.score(woe)
    drivers = model.card.principal_drivers(woe, top=TOP_DRIVERS)
    # Audit trail: what was produced and by which model, never the loan itself.
    log.info(
        "scored",
        request_id=request_id,
        model_id=model.manifest.model_id,
        version=model.manifest.version,
        n=len(rows),
        pd_mean=round(float(pd_12m.mean()), 6),
        pd_max=round(float(pd_12m.max()), 6),
    )
    return ScoreResult(
        request_id=request_id,
        pd_12m=[float(p) for p in pd_12m],
        score=[float(s) for s in points],
        principal_drivers=[
            [{"feature": d["feature"], "points": round(float(d["points"]), 4)} for d in loan]
            for loan in drivers
        ],
    )
