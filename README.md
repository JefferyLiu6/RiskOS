# RiskOS

**An IFRS 9 expected-credit-loss system with the model governance OSFI E-23 asks
for.** PD, LGD and lifetime ECL on 950,000 U.S. mortgages, trained on 1999–2006,
then made to face 2008 alongside the controls that were supposed to catch it.

![Calibration collapses to nearly 5x under-prediction through 2008 while score PSI never leaves the stable band](reports/figures/monitoring_blind_spot.png)

## What it shows

- **A good classifier was a bad risk model.** Through the 2008 crisis both PD
  models kept about 80% of their ranking power while predicting a third of the
  defaults that happened. ECL is PD × LGD × EAD, so that is a threefold
  understatement of the provision. A model chosen on AUC would have passed.
- **The standard drift control could not see it.** Score PSI stayed under 0.012
  for both model families while calibration collapsed. PSI compares
  distributions of inputs and outputs; what changed was the relationship between
  them. Recorded as a high-severity finding against the monitoring design, not
  hidden behind a green chart.
- **The macro overlay recovers half and cannot recover the rest.** Fitted on
  pre-crisis data and fed the real 2008–09 economy, it under-predicts by 2×. The
  estimation sample contains no house-price decline, so no estimator can recover
  the sensitivity. That sits in the executive summary, not an appendix.
- **Governance that runs, not governance that is written.** A model inventory
  reconciled against the artefacts on disk, which caught that the model the
  selection rubric chose had never been persisted. A monitoring rulebook
  committed before the run, with its false-positive rate measured on the
  development sample and published rather than tuned away. A scoring API that
  serves only what the inventory says is approved, and returns the approval
  conditions and open findings with every score.
- **Two critical defects found by the project's own checks.** Every loan was
  provisioned on another loan's PD after a join reordered rows; the mean never
  moved, and run-to-run irreproducibility exposed it. A gitignore pattern had
  kept the entire modelling package out of version control.

**32 findings recorded at discovery, 14 open. 27 registered assumptions, each
bound to the code by a test. 381 tests.** The validation report and model cards
are generated from the artefacts by `make report`; no figure in them is typed by
hand.

## See it in sixty seconds

1. [`reports/validation_report.md`](reports/validation_report.md) — the
   executive summary is the first screen.
2. [`governance/findings_register.yaml`](governance/findings_register.yaml) —
   what went wrong, when it was found, what was done, and which diagnoses turned
   out to be wrong.
3. [`governance/model_inventory.yaml`](governance/model_inventory.yaml) — tiers,
   owners, approvals with conditions and review dates.
4. [`docs/06-monitoring.md`](docs/06-monitoring.md) — why the chart above looks
   the way it does.

The scoring service returns provenance with every number. A real loan from the
crisis period, one payment late:

```json
{
  "result": {
    "pd_12m": 0.1051,
    "score": 548.9,
    "principal_drivers": [
      {"feature": "current_loan_delinquency_status", "points": -93.69},
      {"feature": "number_of_borrowers", "points": -11.66},
      {"feature": "original_interest_rate", "points": 10.43}
    ]
  },
  "governance": {
    "model_id": "RISKOS_PD_001",
    "version": "1.0.0",
    "approval_status": "approved_with_conditions",
    "review_due": "2027-08-31",
    "limitations": ["F-004", "F-005", "F-015", "F-018"],
    "config_drift": []
  }
}
```

## How it is built

Seven phases, each runnable from the command line, each with its own chapter.

| Phase | What it does | Headline |
| --- | --- | --- |
| 1 · Ingest | Layout-driven parser from a version-stamped User Guide; casts abort on any failure, no tolerance threshold | 19 vintages · 950,000 loans · 57.4M loan-months |
| 2 · Panel | Observation-cohort design: every loan alive at each quarter-end, labelled on default in the next 12 months; loan-disjoint out-of-time splits; leakage control as config plus tests | 3.44M panel rows · 54.7M risk-set rows · crisis peak 3.17% in 2009Q1 |
| 3 · Scorecard | WOE logistic with monotone bins, IV screening with a 0.9 leakage tripwire, collinearity pruning, wrong-sign elimination | 15 features · Gini 0.787 in-sample, 0.638 in the crisis · O/E 1.00 → 3.21 |
| 4 · Challenger | LightGBM on a fixed six-point grid, same pool and constraints as the scorecard, isotonic and Platt calibration, TreeSHAP vs points; rubric committed before the fit | LightGBM 0.4035 vs 0.3648; both score near zero on calibration |
| 5 · ECL | Competing-risk discrete-time hazards, unclipped LGD with a denominator floor, staged and discounted ECL, two-covariate macro overlay with Newey–West intervals | $34.0M on $15.91B (0.214%) · weighted $60.1M · backtest ratio 2.04 |
| 6 · Governance | Six-rule monitoring rulebook with owners and actions; inventory reconciled against disk; serving gated on the approval record | 64 windows · both models · PSI blind (F-015) · selected model had no artefact (F-018) |
| 7 · Report | Validation report and model cards generated from the artefacts | 17 sections, zero hand-typed figures |

## Governance as code

The controls are data and tests, not documents.

- **Assumptions register.** Every numeric assumption lives in
  `conf/assumptions.yaml` with a source, a materiality statement and a
  sensitivity test. Tests bind the register to the constants in code, to the
  monitoring rulebook, and to a fitted artefact; a value that drifts fails the
  build.
- **Findings register.** Recorded when found, not assembled at the end.
  Remediated entries keep their original text, and three findings record a
  first diagnosis that was wrong alongside the one that was right.
- **Pre-committed rules.** The selection rubric and the monitoring thresholds
  were committed to git before the comparison and the run they govern, and were
  applied as written when they turned out to be imperfect, with the sensitivity
  reported instead.
- **Inventory reconciliation.** `riskos registry` checks the inventory against
  the artefacts and the findings register: a model in use with no loadable
  artefact, an artefact nobody inventoried, a bundle fitted against changed
  config, an undisclosed high-severity finding, a tier-1 model with no monitoring
  rule, a lapsed approval.
- **Gated serving.** `riskos serve` takes no model path. It serves the model the
  inventory records as in use, refuses a candidate or a lapsed approval, and
  attaches the approval conditions and recorded limitations to every response.

## Why this shape, and what it is not

The revised OSFI Guideline E-23 on model risk management takes effect in May
2027 and extends to every model carrying risk at a federally regulated
institution, with explicit attention to AI and machine learning. The inventory,
risk tiering, approval records, committed monitoring thresholds and
reconciliation here are that lifecycle, built as code and tests.

> The underlying portfolio is U.S. residential mortgage data, because comparable
> public Canadian loan-level default and loss data is not available. The project
> applies IFRS 9 concepts and OSFI Guideline E-23 as a methodological and
> governance framework relevant to Canadian financial institutions. It does not
> represent a regulatory implementation, does not reproduce any institution's ECL
> system, and makes no claim of OSFI compliance.

The review is a **developer validation with simulated second-line review**. One
person cannot be organisationally independent of their own work, and that is
recorded as a condition on every tier-1 approval rather than waived.

## What is still open

The findings register separates conclusions from unfinished work. The open
items that imply work: the scorecard has not yet been reported without
delinquency status (F-004); lifetime PD beyond 96 months is an extrapolation
(F-008); the SICR test needs a delinquency transition model to be more than a
backstop (F-011, F-014); there is no leading indicator for the failure mode that
matters most (F-015); and the CSI rules need time-structural features split out
(F-016). The hazard and LGD models are tier 1 and unmonitored, which the
reconciliation reports rather than hides.

## Guide book

One chapter per phase, explaining the domain as well as the code.

- [Business context](docs/00-business-context.md) — why this system exists, what
  it costs to get wrong, and who is accountable. **Read first.**
- [Phase 1 — Ingest and validate](docs/01-ingest.md)
- [Phase 2 — Panel, risk set, and labels](docs/02-panel.md)
- [Phase 3 — The WOE scorecard champion](docs/03-scorecard.md)
- [Phase 4 — Challenger, calibration, and selection](docs/04-challenger.md)
- [Phase 5 — Hazard, LGD, ECL, and scenarios](docs/05-ecl.md)
- [Phase 6 — Monitoring, registry, and serving](docs/06-monitoring.md)
- Phase 7 — [`reports/validation_report.md`](reports/validation_report.md) and
  [`reports/model_card.md`](reports/model_card.md), generated by `make report`

## Reproducing it

Everything a reviewer needs to read is committed: the aggregate CSVs and figures
under `reports/figures/`, the governance registers, the generated report. The
licensed loan-level data is not, so the pipeline itself needs the data.

**Setup**

```bash
make setup     # uv sync
make test      # 381 tests; those that need the data skip with a reason
make lint      # ruff, mypy --strict
```

**Data.** Access is manual. Freddie Mac distributes the Single-Family Loan-Level
Dataset through Clarity Data Intelligence, which requires free registration.
There is no scraper in this repository and there will not be one.

1. Register and accept the SFLLD licence terms.
2. Download the **sample** dataset (50,000 loans per vintage year) for vintages
   **1999–2012** and **2015–2019**. Each vintage has an origination file and a
   performance file, pipe-delimited, no header row.
3. Download the *Single Family Loan-Level Dataset General User Guide* that ships
   with it.
4. Unpack into `data/raw/`. The whole `data/` tree is gitignored.
5. Record the User Guide version and column layout in `conf/data.yaml`. Ingest
   refuses to run while `layout.version` is null; column positions come from the
   guide, never from memory.

FRED macro series need a free API key in `.env` (see `.env.example`).

**Pipeline**

```bash
make ingest      # Phase 1
make panel       # Phase 2
make train       # Phase 3 scorecard
make challenger  # Phase 4 LightGBM, calibration, selection
make hazard      # Phase 5 hazards and reconciliation
make ecl         # Phase 5 LGD, ECL, scenarios, backtest
make monitor     # Phase 6 rulebook over both models
make registry    # Phase 6 inventory reconciliation (exits non-zero on a high-severity gap)
make serve       # Phase 6 scoring API, gated on the inventory
make report      # Phase 7 validation report and model cards
```

**Layout**

```
conf/          typed YAML config — every numeric assumption lives here, never in code
governance/    model inventory, findings register, alert register, reconciliation
docs/          guide book, one chapter per phase
src/riskos/    ingest, panel, features, models, metrics, ecl, monitor, registry, serve, report
tests/         381 tests, including leakage assertions and register-to-code bindings
reports/       generated validation report and model cards, selection memo, figures
data/          gitignored
```

## Intended use

Illustrative and educational. **Not** for real lending or underwriting decisions,
**not** for regulatory capital or provisioning, and not applicable to any
Canadian portfolio without redevelopment on Canadian data.
