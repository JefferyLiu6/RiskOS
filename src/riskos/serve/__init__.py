"""Phase 6 — serving the approved PD model.

Serving is the point at which a model stops being an analysis and starts
producing numbers other systems act on, so it is also the point at which the
governance record has to bind. The rule here is simple and enforced in code:
**the service serves whatever the inventory says is in use, and refuses anything
else.** It does not take a bundle path. It asks ``governance/model_inventory.yaml``
for the PD model with status ``in_use``, checks that the approval is current, loads
the artefact that entry names, and confirms the artefact's own manifest agrees
about which model it is.

A candidate model, an unapproved model, a lapsed approval, or an artefact whose
manifest names a different model id all fail at startup rather than at the first
request. Every response carries the model id and version that produced it, the
inventory and approval status, and the findings the model is recorded as
carrying, so a consumer cannot use the number without seeing its provenance.
"""

from riskos.serve.scoring import ServedModel, ServingError, load_served_model, score

__all__ = ["ServedModel", "ServingError", "load_served_model", "score"]
