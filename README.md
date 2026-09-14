# RiskOS

**End-to-end credit loss modelling on residential mortgages: build PD/LGD/ECL models, test them through the 2008 crisis, and put monitoring and inventory checks around them.**

Portfolio of ~950,000 U.S. mortgages (Freddie Mac sample). Methods follow **IFRS 9** expected-credit-loss ideas and common **model-risk** practices (inventory, findings, monitoring thresholds). This is a learning project — not a bank system and not a regulatory filing.

![Through 2008, predicted defaults fell far short of what occurred, while a common score-drift check (PSI) stayed in the “stable” range](reports/figures/monitoring_blind_spot.png)

## What this shows

| Result | Meaning |
| --- | --- |
| Ranking held, levels did not | Both PD models kept most of their ranking power in the crisis, but predicted only about one-third of the defaults that occurred — so provisions based on those PDs would have been far too low |
| PSI did not flag it | Score PSI stayed very low (“stable”) while calibration broke. That gap is logged as a monitoring-design finding (F-015) |
| Macro overlay is incomplete | A pre-crisis macro overlay given the real 2008–09 economy still under-predicted defaults by roughly 2× |
| Checks found concrete bugs | Including a join that assigned another loan’s PD after a row reorder, and a gitignore mistake that had excluded the modelling package from version control |

Findings are logged as they appear (32 total; some remain open on purpose as limitations). Assumptions live in config and are checked by tests. The validation report is generated from pipeline outputs (`make report`).

## What was built

| Area | Contents |
| --- | --- |
| Modelling | WOE logistic PD scorecard, LightGBM challenger, calibration, hazard, LGD, staged/discounted ECL |
| Evaluation | Discrimination, calibration (O/E), crisis window, scenario overlay |
| Controls | Monitoring rules, findings register, model inventory, reconciliation vs artefacts on disk |
| Serving | Small scoring API that only loads the inventory’s designated model and returns listed limitations |

### Pipeline

| Phase | What it does |
| --- | --- |
| 1 · Ingest | Parse and validate loan-level files |
| 2 · Panel | Build observation dataset with out-of-time splits and leakage checks |
| 3 · Scorecard | Fit champion PD model |
| 4 · Challenger | Fit LightGBM, calibrate, compare under a rubric fixed before fitting |
| 5 · ECL | Lifetime hazards, LGD, ECL, macro scenarios, backtest |
| 6 · Controls | Monitor, reconcile inventory, optional scoring service |
| 7 · Report | Generate validation report and model cards |

Detail and scale numbers: [docs](docs/00-business-context.md) and [`reports/validation_report.md`](reports/validation_report.md).

## Quick links

1. [`reports/validation_report.md`](reports/validation_report.md) — start with the executive summary  
2. [`governance/findings_register.yaml`](governance/findings_register.yaml) — issues found and status  
3. [`governance/model_inventory.yaml`](governance/model_inventory.yaml) — models, owners, limitations  
4. [`docs/06-monitoring.md`](docs/06-monitoring.md) — monitoring design and the PSI gap  

Example API response (crisis-period loan, one payment late). `approval_status` here means **developer clearance for this project**, not an independent bank model approval:

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

## How controls work in this repo

- **Assumptions** — numeric assumptions in `conf/assumptions.yaml`; tests fail if code drifts from them  
- **Findings** — recorded when found (including cases where the first diagnosis was wrong)  
- **Pre-committed rules** — selection rubric and monitoring thresholds fixed in git before the runs they govern  
- **Inventory check** — `riskos registry` compares inventory ↔ artefacts ↔ findings and exits non-zero on serious gaps  
- **Serving** — `riskos serve` does not take an arbitrary model path; it serves what the inventory marks as in use  

These are project disciplines inspired by model-risk practice (including themes in OSFI Guideline E-23). They are not a claim of institutional compliance.

## Scope and limits

> **Data:** U.S. conforming residential mortgages. Public Canadian loan-level default/loss data is not available in comparable form, so this does not cover Canadian-specific products (e.g. CMHC-insured mortgages, HELOCs) or a Canadian lender’s book.  
> **Framework:** IFRS 9-style ECL measurement and common model-risk controls. **Not** a regulatory implementation, **not** any institution’s ECL system, **no** claim of OSFI compliance.  
> **Review:** One author. This is developer validation with a *simulated* second-line review — not organisationally independent validation. Inventory “approval” fields mean illustrative clearance by the author, with conditions and review dates recorded explicitly.

Open limitations that imply further work include stronger monitoring for the PSI blind spot (F-015), SICR/lifetime-PD design, and monitoring for hazard/LGD (currently unmonitored tier-1 entries — the registry reports that).

## Docs

- [Business context](docs/00-business-context.md) — why loss provisioning matters (**read first**)  
- [Ingest](docs/01-ingest.md) · [Panel](docs/02-panel.md) · [Scorecard](docs/03-scorecard.md) · [Challenger](docs/04-challenger.md) · [ECL](docs/05-ecl.md) · [Monitoring](docs/06-monitoring.md)  
- Generated: [`reports/validation_report.md`](reports/validation_report.md) · [`reports/model_card.md`](reports/model_card.md)

## Reproduce

Committed: aggregate CSVs/figures under `reports/`, governance registers, generated report. Not committed: licensed loan-level data.

```bash
make setup     # uv sync
make test      # unit/integration tests (data-dependent tests skip with a reason)
make lint      # ruff, mypy --strict
```

**Data.** Register for Freddie Mac SFLLD via Clarity Data Intelligence (free). Download sample vintages **1999–2012** and **2015–2019**, unpack to `data/raw/`, set layout version in `conf/data.yaml`. No scraper in this repo. FRED series need a key in `.env` (see `.env.example`).

```bash
make ingest && make panel && make train && make challenger
make hazard && make ecl && make monitor && make registry
make serve     # local scoring API
make report    # refresh validation report and model cards
```

```
conf/          typed YAML config (assumptions live here)
governance/    inventory, findings, alerts, reconciliation outputs
docs/          phase write-ups
src/riskos/    ingest → panel → models → ecl → monitor → registry → serve → report
tests/
reports/       generated report, model cards, figures
data/          gitignored (local/raw only)
```

## Intended use

Educational only. Not for lending, underwriting, regulatory capital, or provisioning. Not transferable to a Canadian portfolio without redevelopment on Canadian data.
