"""Phase 6 — ongoing monitoring: drift, performance decay, and alert records.

The organising distinction in this package is between what can be measured at
scoring time and what cannot.

* **Drift** — score PSI and feature CSI — compares the scored population with
  the development sample. Both sides are inputs, so a drift number exists the
  moment a window is scored.
* **Performance** — discrimination and calibration — needs outcomes. The label
  is a 12-month forward default flag, so the outcome for an observation dated
  ``T`` is unknown until ``T+12m``.

That asymmetry is the entire justification for drift monitoring. If outcomes
arrived instantly nobody would watch input distributions; they would watch the
default rate. They do not arrive instantly, so for a year the only available
evidence that a model has stopped working is that the population it is scoring
has stopped looking like the population it was fitted on.

``riskos.monitor.run`` reports both timelines on the same axis and measures the
gap between them, which is the number that says whether drift monitoring is
worth running on this model.
"""

from riskos.monitor.alerts import Alert, evaluate_rules
from riskos.monitor.config import MonitoringConfig, Rule, monitoring_config
from riskos.monitor.drift import DriftWindow, drift_timeline
from riskos.monitor.performance import PerformanceWindow, performance_timeline

__all__ = [
    "Alert",
    "DriftWindow",
    "MonitoringConfig",
    "PerformanceWindow",
    "Rule",
    "drift_timeline",
    "evaluate_rules",
    "monitoring_config",
    "performance_timeline",
]
