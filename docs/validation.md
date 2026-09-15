# Validation and limitations

The project evaluates behavioural mortgage default models across time periods
and tests the implementation with controlled inputs. [Results](../README.md#what-the-experiment-found)
and [reproduction instructions](reproduce.md) are available separately.

## Validation coverage

| Check | Evidence | Inputs |
| --- | --- | --- |
| Out-of-time model comparison | [Ranking and calibration metrics](../reports/figures/champion_challenger_metrics.csv) | Mortgage panel |
| Dependence on current delinquency | [Ablation study](../reports/delinquency_ablation.md) | Same panel, refitted variants |
| Leakage and cohort separation | [Leakage tests](../tests/test_leakage.py) | Config and locally built panel |
| Loan-to-PD alignment and loss calculation | [ECL integration tests](../tests/test_ecl_integration.py) | Synthetic inputs; runs in CI |
| Outcome availability and alert timing | [Monitoring tests](../tests/test_monitoring.py) | Synthetic inputs; runs in CI |
| Recorded model and artifact consistency | [Registry checks](../src/riskos/registry/reconcile.py) | Inventory and local artifacts |

`make test` runs the suite. Checks that require unavailable local artifacts skip
with a reason. CI exercises the synthetic integration path and saved-result
checks; rebuilding the empirical experiment requires licensed data.

## Integration safeguards

The ECL integration tests fit hazards, enrich observations through SQL, project
per-loan probabilities, assign stages, calculate discounted losses, run scenarios,
and check report totals. Synthetic training data, LGD segments, and macro inputs
replace the licensed inputs. The tests also exercise the ECL command-line entry point.

The [alignment guard](../src/riskos/panel/alignment.py) requires unique, non-null
loan/date keys and preserves their sequence through enrichment. The tests verify
that each loan receives the same probability and loss when scored in a portfolio,
in reverse order, or alone. Reordered observations and duplicate join matches
raise errors. LGD joins preserve the input order and reject duplicate segment keys.

A neutral macro scenario must reconcile to independently calculated base losses,
including staging and discounting. Macro predictions must remain unchanged when
unrelated evaluation quarters are added. These checks protect the fixes recorded
in R-012 and R-014 in the [findings register](../governance/findings_register.yaml).

## Model limitations

| Area | Limitation |
| --- | --- |
| Default models | Both families under-predict crisis defaults. Current delinquency contributes materially to ranking; the ablation retains other behavioural covariates |
| Monitoring | Default outcomes take 12 months to mature. Low score drift does not establish calibration, and feature-drift rules produce frequent development-sample alarms |
| Stage 3 | The empirical panel excludes already-defaulted observations. Synthetic branch checks do not provide empirical validation of impaired-loan losses |
| SICR | At a ratio threshold of 2.0, 1,524 of 1,556 Stage 2 loans enter through the delinquency backstop; only 32 are incremental. Forward-looking sensitivity remains limited |
| Lifetime hazard | Older loan ages are extrapolated beyond the fitting range; the model lacks delinquency transition dynamics |
| Macro overlay | The fit has 24 quarterly observations and no house-price declines. Corrected mean crisis observed/expected defaults is 1.79 |
| Scenario ranges | Coefficient-endpoint sensitivities are not joint prediction intervals covering model and scenario uncertainty |

The comparison rubric is recorded in configuration. The delinquency ablation
reuses the existing evaluation splits and is a retrospective sensitivity study.
Its [manifest](../reports/figures/delinquency_ablation_manifest.json) records input
hashes, source hashes, row counts, and fitting settings for traceability.

## Scope

This is a single-author educational project using U.S. mortgage data and
simplified IFRS 9-style loss mechanics. Inventory clearance is illustrative;
independent validation and use in lending or financial reporting are outside
this project's scope. The [full report](../reports/validation_report.md) describes
methods, sensitivity analyses, and open findings.
