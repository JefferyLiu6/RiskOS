"""Phase 6 — serving is gated on the inventory, and scoring is a property of the loan.

The synthetic tests build a real binning process and a real scorecard on
generic columns, save them as a bundle, and point a hand-built inventory at it.
That exercises every gate and every scoring invariant without loan data. The
tests that hit the real bundle over HTTP are marked ``needs_data`` and skip on a
fresh clone.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from riskos.features import binning
from riskos.models.scorecard import Scorecard
from riskos.panel.config import feature_config
from riskos.registry.inventory import InventoryConfig, inventory
from riskos.serve import scoring

RNG = np.random.default_rng(20260913)
REAL = inventory()
PANEL_GLOB = "data/panel/panel/*/*.parquet"
needs_data = pytest.mark.skipif(
    not Path("data/panel/panel").exists() or not Path("models/scorecard_bundle").exists(),
    reason="needs the built panel and the scorecard bundle",
)


# --- a toy model with a real binning process --------------------------------


@pytest.fixture(scope="module")
def toy_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A fitted scorecard on generic columns, saved as a bundle under models/."""
    n = 6_000
    u = RNG.normal(0.0, 1.0, n)
    v = RNG.normal(0.0, 1.0, n)
    k = RNG.choice(["p", "q", "r"], n)
    logit = -3.0 + 1.2 * u - 0.8 * v + np.where(k == "r", 0.9, 0.0)
    y = RNG.binomial(1, 1.0 / (1.0 + np.exp(-logit)))
    frame = pl.DataFrame({"u": u, "v": v, "k": k, "default_12m": y})

    cfg = feature_config().binning
    process = binning.build_process(["u", "v", "k"], frame, cfg)
    process.fit(frame.select(["u", "v", "k"]).to_pandas(), y)
    woe, _ = binning.transform(process, frame, ["u", "v", "k"])
    card = Scorecard(["u", "v", "k"])
    card.fit(woe, y)

    root = tmp_path_factory.mktemp("estate")
    card.save_bundle(root / "models" / "toy_bundle", process, model_id="TOY_001")
    return root


def _entry(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "TOY_001",
        "name": "toy",
        "purpose": "tests",
        "family": scoring.SERVED_FAMILY,
        "status": "in_use",
        "owner": "someone",
        "tier": 1,
        "tier_rationale": "because",
        "intended_use": "tests",
        "prohibited_use": "anything else",
        "artefact": "models/toy_bundle",
        "monitored_by": ["MON-01"],
        "approval": {
            "status": "approved_with_conditions",
            "approved_on": date(2026, 1, 1),
            "approved_by": "someone",
            "review_due": date(2027, 1, 1),
            "conditions": "illustrative only",
        },
        "limitations": ["F-005"],
    }
    return base | overrides


def _inventory(*entries: dict[str, object]) -> InventoryConfig:
    return InventoryConfig.model_validate(
        {
            "schema_": REAL.schema_.model_dump(),
            "tiering_rubric": REAL.tiering_rubric.model_dump(),
            "models": list(entries),
        }
    )


@pytest.fixture
def served(toy_bundle: Path) -> scoring.ServedModel:
    return scoring.load_served_model(_inventory(_entry()), root=toy_bundle, as_at=date(2026, 6, 1))


# --- the gate ------------------------------------------------------------------


def test_the_in_use_approved_model_is_served(served: scoring.ServedModel) -> None:
    assert served.manifest.model_id == "TOY_001"
    assert served.features == ["u", "v", "k"]
    assert served.categorical == {"k"}


def test_a_candidate_model_is_refused(toy_bundle: Path) -> None:
    """A bundle on disk is not a reason to serve. An approval is."""
    inv = _inventory(_entry(status="candidate", approval={"status": "not_approved"}))

    with pytest.raises(scoring.ServingError, match="not in use"):
        scoring.load_served_model(inv, root=toy_bundle, as_at=date(2026, 6, 1))


def test_a_lapsed_approval_is_refused(toy_bundle: Path) -> None:
    with pytest.raises(scoring.ServingError, match="lapsed approval is not an approval"):
        scoring.load_served_model(_inventory(_entry()), root=toy_bundle, as_at=date(2030, 1, 1))


def test_two_models_in_use_is_refused_rather_than_guessed(toy_bundle: Path) -> None:
    inv = _inventory(_entry(), _entry(id="TOY_002"))

    with pytest.raises(scoring.ServingError, match="will not guess"):
        scoring.load_served_model(inv, root=toy_bundle, as_at=date(2026, 6, 1))


def test_an_artefact_naming_a_different_model_is_refused(toy_bundle: Path) -> None:
    """The inventory says one thing, the manifest says another. Neither is trusted."""
    inv = _inventory(_entry(id="TOY_999"))

    with pytest.raises(scoring.ServingError, match="disagree about what this is"):
        scoring.load_served_model(inv, root=toy_bundle, as_at=date(2026, 6, 1))


def test_a_missing_artefact_is_refused_with_the_bundle_error(toy_bundle: Path) -> None:
    inv = _inventory(_entry(artefact="models/absent"))

    with pytest.raises(scoring.ServingError, match="no model bundle"):
        scoring.load_served_model(inv, root=toy_bundle, as_at=date(2026, 6, 1))


def test_governance_block_carries_provenance_and_limitations(
    served: scoring.ServedModel,
) -> None:
    g = served.governance()

    assert g.model_id == "TOY_001"
    assert g.approval_status == "approved_with_conditions"
    assert g.conditions == "illustrative only"
    assert g.limitations == ("F-005",)
    assert g.review_due == date(2027, 1, 1)


# --- scoring invariants ----------------------------------------------------------


def _loans(n: int) -> list[dict[str, object]]:
    return [
        {"u": float(RNG.normal()), "v": float(RNG.normal()), "k": str(RNG.choice(["p", "q", "r"]))}
        for _ in range(n)
    ]


def test_a_loan_scored_alone_matches_the_same_loan_in_a_batch(
    served: scoring.ServedModel,
) -> None:
    """Defect R-003, asserted at the serving boundary.

    The PD is trivially batch-independent. The explanation is not, unless it is
    centred on a baseline fixed at fit time - and a borrower's stated reasons
    for a decision must not depend on who else was scored that second.
    """
    loans = _loans(40)
    batch = scoring.score(served, loans)
    alone = scoring.score(served, [loans[11]])

    assert alone.pd_12m[0] == pytest.approx(batch.pd_12m[11], abs=1e-12)
    assert alone.score[0] == pytest.approx(batch.score[11], abs=1e-9)
    assert alone.principal_drivers[0] == batch.principal_drivers[11]


def test_a_null_feature_is_scored_as_its_own_bin(served: scoring.ServedModel) -> None:
    loan = _loans(1)[0]
    loan["v"] = None

    result = scoring.score(served, [loan])

    assert 0.0 < result.pd_12m[0] < 1.0


def test_an_absent_feature_is_refused_by_name(served: scoring.ServedModel) -> None:
    """Absent is not null. Nulling it would score a field nobody supplied."""
    loan = _loans(1)[0]
    del loan["k"]

    with pytest.raises(scoring.ServingError, match=r"missing required features \['k'\]"):
        scoring.score(served, [loan])


def test_an_empty_request_is_refused(served: scoring.ServedModel) -> None:
    with pytest.raises(scoring.ServingError, match="nothing to score"):
        scoring.score(served, [])


def test_drivers_are_the_top_features_by_absolute_points(served: scoring.ServedModel) -> None:
    result = scoring.score(served, _loans(3))

    for drivers in result.principal_drivers:
        magnitudes = [abs(float(d["points"])) for d in drivers]
        assert magnitudes == sorted(magnitudes, reverse=True)
        assert len(drivers) == 3  # only three features exist


def test_scored_pds_match_the_scorecard_scored_directly(
    served: scoring.ServedModel,
) -> None:
    """The serving path adds padding columns; it must not change the number."""
    loans = _loans(25)
    frame = pl.DataFrame(
        {
            "u": [x["u"] for x in loans],
            "v": [x["v"] for x in loans],
            "k": [x["k"] for x in loans],
            "default_12m": [0] * 25,
        }
    )
    woe, _ = binning.transform(served.process, frame, served.features)
    direct = served.card.predict_proba(woe)

    via_serving = np.array(scoring.score(served, loans).pd_12m)

    np.testing.assert_allclose(via_serving, direct, atol=1e-12)


# --- the real bundle over HTTP -----------------------------------------------------


@needs_data
def test_the_api_serves_the_inventoried_model_and_reproduces_the_direct_path() -> None:
    from fastapi.testclient import TestClient

    from riskos.serve.app import Loan, app

    panel = pl.read_parquet(PANEL_GLOB).filter(pl.col("split") == "oot_stress").head(30)
    card, process, _ = Scorecard.from_bundle(Path("models/scorecard_bundle"))
    woe, _ = binning.transform(process, panel, card.features)
    direct = card.predict_proba(woe)
    loans = [{k: v for k, v in row.items() if k in card.features} for row in panel.to_dicts()]

    assert set(Loan.model_fields) == set(card.features)

    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["model_id"] == REAL.by_id("RISKOS_PD_001").id  # type: ignore[union-attr]

        batch = client.post("/score/batch", json=loans)
        assert batch.status_code == 200, batch.text
        served = np.array([r["pd_12m"] for r in batch.json()["results"]])
        np.testing.assert_allclose(served, direct, atol=1e-12)
        assert batch.json()["governance"]["approval_status"] == "approved_with_conditions"
        assert "F-005" in batch.json()["governance"]["limitations"]

        one = client.post("/score", json=loans[3]).json()
        assert one["result"]["pd_12m"] == pytest.approx(served[3], abs=1e-12)


@needs_data
def test_the_api_distinguishes_null_from_absent_and_refuses_unknown_fields() -> None:
    from fastapi.testclient import TestClient

    from riskos.serve.app import app

    panel = pl.read_parquet(PANEL_GLOB).filter(pl.col("split") == "oot_stress").head(1)
    card, _, _ = Scorecard.from_bundle(Path("models/scorecard_bundle"))
    loan = {k: v for k, v in panel.to_dicts()[0].items() if k in card.features}

    with TestClient(app) as client:
        nulled = {**loan, "original_dti": None}
        assert client.post("/score", json=nulled).status_code == 200

        absent = {k: v for k, v in loan.items() if k != "credit_score"}
        response = client.post("/score", json=absent)
        assert response.status_code == 422
        assert "credit_score" in response.json()["detail"]

        assert client.post("/score", json={**loan, "postal_code": "1"}).status_code == 422
