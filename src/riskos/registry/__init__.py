"""Phase 6 — the model inventory and its reconciliation against reality.

OSFI E-23 expects an institution to know what models it has, what each is for,
who owns it, and what condition it is in. The part that is easy to produce is the
list. The part that makes the list worth anything is the reconciliation: an
inventory nobody checks against the artefacts on disk records what somebody
intended, not what is running.

``reconcile`` reads three independent sources — the inventory, the model
artefacts, and the findings register — and reports every place they disagree.
The checks are deliberately the unflattering ones: a model in use with no
loadable artefact, an artefact nobody inventoried, a bundle fitted against
configuration that has since changed, an open high-severity finding missing from
a model's recorded limitations, a tier-1 model with no monitoring rule pointed at
it, and an approval past its review date.
"""

from riskos.registry.inventory import (
    InventoryConfig,
    ModelEntry,
    inventory,
)
from riskos.registry.reconcile import Discrepancy, reconcile, summary

__all__ = [
    "Discrepancy",
    "InventoryConfig",
    "ModelEntry",
    "inventory",
    "reconcile",
    "summary",
]
