# Reference index

Start with the [presentation walkthrough](demo.md) or the [project overview](../README.md).
The phase documents below are detailed development references; you do not need to
read them in sequence. [Setup and reproduction](reproduce.md) covers execution.

[Validation coverage and limitations](validation.md) summarises the empirical and automated checks.

## Core experiment

| Question | Read | Implementation |
| --- | --- | --- |
| What data and default definition were used? | [Ingest](01-ingest.md) | [src/riskos/ingest/](../src/riskos/ingest/) |
| How are future information and overlapping loans kept out? | [Panel](02-panel.md) | [src/riskos/panel/](../src/riskos/panel/) |
| How does the baseline predict default? | [Scorecard](03-scorecard.md) | [src/riskos/features/](../src/riskos/features/), [src/riskos/models/train_scorecard.py](../src/riskos/models/train_scorecard.py) |
| Does a more flexible model help? | [Challenger](04-challenger.md), [selection memo](../reports/model_selection_memo.md) | [src/riskos/models/train_challenger.py](../src/riskos/models/train_challenger.py) |
| Why did drift monitoring miss the failure? | [Monitoring, §§1–4](06-monitoring.md) | [src/riskos/monitor/](../src/riskos/monitor/), [src/riskos/metrics/](../src/riskos/metrics/) |

[src/riskos/cli.py](../src/riskos/cli.py) connects the commands to these modules. Tests live in [tests/](../tests/).

## Optional extensions

| Topic | Reference | Implementation |
| --- | --- | --- |
| Provisioning background and vocabulary | [Business context](00-business-context.md) | Background reading |
| Lifetime hazards, LGD, ECL, and macro scenarios | [ECL](05-ecl.md) | [src/riskos/models/train_hazard.py](../src/riskos/models/train_hazard.py), [src/riskos/ecl/](../src/riskos/ecl/) |
| Inventory reconciliation and scoring API | [Monitoring, registry, and serving](06-monitoring.md) | [src/riskos/registry/](../src/riskos/registry/), [src/riskos/serve/](../src/riskos/serve/) |
| Full validation narrative and model cards | [Validation report](../reports/validation_report.md), [model card](../reports/model_card.md) | [src/riskos/report/](../src/riskos/report/) |

## Config: open only what you need

| Files | Purpose |
| --- | --- |
| `conf/data.yaml`, `conf/panel.yaml` | Input layout, labels, sampling, and time splits |
| `conf/features.yaml`, `conf/models.yaml` | Candidate features, fitting settings, and selection rubric |
| `conf/monitoring.yaml` | Drift and performance thresholds |
| `conf/assumptions.yaml`, `conf/scenarios.yaml` | Cross-cutting numeric assumptions and ECL scenarios |

The YAML files remain separate because they configure different pipeline steps.
Their historical timing claims are unverified; see [configuration provenance](../conf/README.md).

## Evidence: inputs versus generated outputs

- **Maintained registers:** `governance/findings_register.yaml` records findings and
  status; `governance/model_inventory.yaml` records model metadata and limitations.
- **Generated evidence:** `reports/` contains aggregate tables, figures, the selection
  memo, model cards, and validation report. `governance/alert_register.csv` and
  `governance/inventory_reconciliation.csv` are generated control outputs.
- **Local inputs and fitted artifacts:** `data/`, `models/`, and MLflow files are
  ignored by git. A fresh clone contains saved evidence, not a runnable fitted model.

Use the generated report when you need the full audit trail. To update evidence,
run the relevant pipeline step and then `make report`; avoid hand-editing generated
results. Existing output paths are retained so code, tests, and report links agree.
