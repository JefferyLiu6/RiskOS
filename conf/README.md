# Configuration

| File | Purpose |
| --- | --- |
| `data.yaml`, `panel.yaml` | Input layout, labels, sampling, and time splits |
| `features.yaml`, `models.yaml` | Features, fitting settings, and comparison rubric |
| `monitoring.yaml` | Drift and performance thresholds |
| `assumptions.yaml`, `scenarios.yaml` | Numeric assumptions and ECL scenarios |

Saved bundles fingerprint the exact training configuration bytes, including
comments. The YAML snapshots are retained for those checks. Chronological comments
are development notes; the experiments are presented as retrospective studies.
See [validation coverage](../docs/validation.md) for their interpretation.
