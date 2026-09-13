"""FastAPI surface over the inventory-gated scorer.

Thin by design: request and response shapes, the startup gate, and four
endpoints. Everything that decides what gets served and how it is scored lives
in ``riskos.serve.scoring`` and is tested there.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date, datetime

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from riskos.log import get_logger
from riskos.serve import scoring

log = get_logger(__name__)


class Loan(BaseModel):
    """One loan, as observed at a point in its life.

    Every field is nullable on purpose. The scorecard bins a missing value as its
    own category - 8% of loans have no DTI and that absence is informative - so
    ``null`` is a legitimate input, not an error. A field that is absent from
    the JSON altogether is different and is refused: see ``scoring._frame``.
    The endpoints dump the request with ``exclude_unset=True`` so pydantic's
    defaults do not quietly convert an absent field into a null one.

    The field set is the scorecard's fitted feature list, and the service checks
    at startup that this schema and the loaded manifest agree; a bundle refitted
    with different features fails to start rather than silently scoring the
    wrong columns.
    """

    model_config = ConfigDict(extra="forbid")

    current_loan_delinquency_status: str | None = Field(
        None, description="Months delinquent as reported, e.g. '0', '1', '2'"
    )
    credit_score: int | None = None
    original_interest_rate: float | None = None
    original_ltv: int | None = None
    original_cltv: int | None = None
    mi_percentage: int | None = None
    original_loan_term: int | None = None
    number_of_borrowers: int | None = None
    property_state: str | None = None
    original_dti: int | None = None
    loan_age: int | None = None
    channel: str | None = None
    original_upb: float | None = None
    current_actual_upb: float | None = None
    property_type: str | None = None


class Driver(BaseModel):
    feature: str
    points: float


class GovernanceBlock(BaseModel):
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
    limitations: list[str]
    config_drift: list[str]


class LoanScore(BaseModel):
    pd_12m: float = Field(description="Twelve-month probability of default")
    score: float = Field(description="Points score; higher is safer")
    principal_drivers: list[Driver]


class ScoreResponse(BaseModel):
    request_id: str
    result: LoanScore
    governance: GovernanceBlock


class BatchScoreResponse(BaseModel):
    request_id: str
    results: list[LoanScore]
    governance: GovernanceBlock


class Health(BaseModel):
    status: str
    model_id: str
    version: str
    loaded_at: datetime
    approval_status: str
    review_due: date | None
    config_drift: list[str]


def _governance(model: scoring.ServedModel) -> GovernanceBlock:
    g = model.governance()
    return GovernanceBlock(
        model_id=g.model_id,
        version=g.version,
        family=g.family,
        training_window=g.training_window,
        inventory_status=g.inventory_status,
        tier=g.tier,
        approval_status=g.approval_status,
        approved_on=g.approved_on,
        review_due=g.review_due,
        conditions=g.conditions,
        limitations=list(g.limitations),
        config_drift=list(g.config_drift),
    )


def _results(result: scoring.ScoreResult) -> list[LoanScore]:
    return [
        LoanScore(
            pd_12m=p,
            score=s,
            principal_drivers=[
                Driver(feature=str(d["feature"]), points=float(d["points"])) for d in drivers
            ],
        )
        for p, s, drivers in zip(result.pd_12m, result.score, result.principal_drivers, strict=True)
    ]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model the inventory says is in use, or refuse to start."""
    model = scoring.load_served_model()
    schema_fields = set(Loan.model_fields)
    if schema_fields != set(model.features):
        raise scoring.ServingError(
            "request schema and loaded model disagree on features: "
            f"schema-only {sorted(schema_fields - set(model.features))}, "
            f"model-only {sorted(set(model.features) - schema_fields)}"
        )
    app.state.model = model
    yield


app = FastAPI(
    title="RiskOS PD scoring",
    version="0.1.0",
    description=(
        "Twelve-month PD from the inventoried, approved scorecard. Illustrative and "
        "educational; not for lending, underwriting, capital or provisioning decisions."
    ),
    lifespan=lifespan,
)


def _model(request: Request) -> scoring.ServedModel:
    model: scoring.ServedModel | None = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return model


@app.get("/health", response_model=Health)
def health(request: Request) -> Health:
    model = _model(request)
    return Health(
        status="degraded" if model.config_drift else "ok",
        model_id=model.manifest.model_id,
        version=model.manifest.version,
        loaded_at=model.loaded_at,
        approval_status=model.entry.approval.status,
        review_due=model.entry.approval.review_due,
        config_drift=list(model.config_drift),
    )


@app.get("/model", response_model=GovernanceBlock)
def model_info(request: Request) -> GovernanceBlock:
    return _governance(_model(request))


@app.post("/score", response_model=ScoreResponse)
def score_one(loan: Loan, request: Request) -> ScoreResponse:
    model = _model(request)
    try:
        result = scoring.score(model, [loan.model_dump(exclude_unset=True)])
    except scoring.ServingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ScoreResponse(
        request_id=result.request_id,
        result=_results(result)[0],
        governance=_governance(model),
    )


@app.post("/score/batch", response_model=BatchScoreResponse)
def score_batch(loans: list[Loan], request: Request) -> BatchScoreResponse:
    model = _model(request)
    try:
        result = scoring.score(model, [loan.model_dump(exclude_unset=True) for loan in loans])
    except scoring.ServingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return BatchScoreResponse(
        request_id=result.request_id,
        results=_results(result),
        governance=_governance(model),
    )
