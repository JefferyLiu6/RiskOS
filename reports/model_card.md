# RiskOS — Model Cards

Generated 2026-09-15 from governance/model_inventory.yaml, the bundle manifests, the Phase 4 comparison table and the findings register.

> The underlying portfolio is U.S. residential mortgage data, because comparable public Canadian loan-level default and loss data is not available. Canadian-specific products (for example CMHC-insured mortgages and HELOCs) are out of scope. The project applies IFRS 9-style ECL measurement and common model-risk practices (inventory, findings, monitoring thresholds; themes also discussed in OSFI Guideline E-23). It is not a regulatory implementation, does not reproduce any institution's ECL system, and makes no claim of OSFI compliance. Inventory 'approval' fields mean illustrative developer clearance by the project author, not an independent model-risk committee decision.

## RISKOS_PD_001 — 12-month PD scorecard (WOE logistic)

| Field | Value |
| --- | --- |
| Family | woe_scorecard |
| Inventory status | in_use |
| Tier | 1 — Feeds ECL directly, which is a reported financial figure. Not autonomous — nothing in this project acts on a customer — but materiality alone is sufficient for tier 1 under the rubric above. |
| Owner | project_author |
| Clearance | approved_with_conditions |
| Cleared on / review due | 2026-08-31 / 2027-08-31 |
| Monitored by | MON-01, MON-02, MON-03, MON-04, MON-05, MON-06 |
| Version | 1.0.0 |
| Training window | 1999Q1-2006Q4 |
| Training rows | 2,046,874 |
| Features | 15: current_loan_delinquency_status, credit_score, original_interest_rate, original_ltv, original_cltv, mi_percentage, original_loan_term, number_of_borrowers, property_state, original_dti, loan_age, channel, original_upb, current_actual_upb, property_type |
| Created | 2026-08-31 |

**Purpose.** Twelve-month probability of default for a seasoned residential mortgage, used as the stage-1 input to the IFRS 9 ECL calculation and as the champion candidate in the Phase 4 comparison.

**Intended use.** Illustrative and educational. Ranking and level estimation of 12-month default risk on U.S. conforming residential mortgages originated 1999-2006, observed at a point in their life.

**Prohibited use.** Not for real lending or underwriting decisions, not for regulatory capital or provisioning, and not applicable to any Canadian portfolio without redevelopment on Canadian data. Not valid outside the origination vintages and product type it was fitted on, and per F-005 not reliable for level estimation under regime change at all.

**Clearance conditions.** Cleared by the project author for illustrative use only, on the condition that the calibration failure in F-005 is stated wherever an output is reported. This is not independent validation; tier 1 would ordinarily require that. The gap is recorded here rather than waived.

**Performance, uncalibrated.**

| Split | Gini | O/E | Score PSI |
| --- | --- | --- | --- |
| train | 0.787 | 1.00 | — |
| validation_in_time | 0.784 | 1.03 | 0.0002 |
| oot_stress | 0.638 | 3.21 | 0.0019 |
| oot_benign | 0.535 | 3.12 | 0.2128 |

**Limitations (findings register).**

| ID | Severity | Status | Title |
| --- | --- | --- | --- |
| F-004 | medium | open | Delinquency status at the observation date dominates the scorecard |
| F-005 | high | open | Scorecard materially under-predicts default under regime change |
| F-015 | high | open | Score PSI is blind to the 2008 regime change, so drift monitoring gives zero months of warning on the failure it exists to catch |
| F-018 | high | remediated | The model selected by the Phase 4 rubric has no scoring-ready artefact, so every downstream figure comes from the model that lost |

## RISKOS_PD_002 — 12-month PD challenger (LightGBM)

| Field | Value |
| --- | --- |
| Family | lightgbm |
| Inventory status | candidate |
| Tier | 1 — Same use and same materiality as RISKOS_PD_001. Tier does not fall because a model is a challenger; a challenger selected by the rubric is a model awaiting deployment, not an experiment. |
| Owner | project_author |
| Clearance | not_approved |
| Cleared on / review due | — / — |
| Monitored by | MON-01, MON-02, MON-03, MON-04, MON-05, MON-06 |
| Version | 1.0.0 |
| Training window | 1999Q1-2006Q4 |
| Training rows | 2,046,874 |
| Features | 27: credit_score, first_time_homebuyer_flag, mi_percentage, number_of_units, occupancy_status, original_cltv, original_dti, original_upb, original_ltv, original_interest_rate, channel, prepayment_penalty_flag, amortization_type, property_state, property_type, loan_purpose, original_loan_term, number_of_borrowers, super_conforming_flag, harp_indicator, interest_only_flag, current_actual_upb, current_loan_delinquency_status, loan_age, remaining_months_to_legal_maturity, current_deferred_upb, modification_flag |
| Created | 2026-09-12 |

**Purpose.** Gradient-boosted challenger to RISKOS_PD_001, selected by the Phase 4 rubric at 0.4035 against 0.3648.

**Intended use.** As RISKOS_PD_001. Selected on the recorded rubric but not deployed.

**Prohibited use.** As RISKOS_PD_001, and additionally not to be reported as the model behind any published figure until a scoring-ready artefact exists and has been monitored. See F-018.

**Clearance conditions.** An artefact now exists (F-018 remediated) and the model is monitored on the same rulebook as RISKOS_PD_001, but it remains not cleared for use. A loadable artefact is not a validated model. Independent validation has not been performed. Clearance would also have to address F-006: the rubric's calibration dimension scores this model 0.062 out of 1.0.

**Performance, uncalibrated.**

| Split | Gini | O/E | Score PSI |
| --- | --- | --- | --- |
| train | 0.832 | 1.00 | — |
| validation_in_time | 0.787 | 1.03 | 0.0002 |
| oot_stress | 0.665 | 2.80 | 0.0004 |
| oot_benign | 0.564 | 2.73 | 0.1407 |

**Limitations (findings register).**

| ID | Severity | Status | Title |
| --- | --- | --- | --- |
| F-006 | high | open | Both PD candidates fail the calibration dimension under stress |
| F-007 | low | open | The selection rubric measured the wrong proxy for explainability |
| F-018 | high | remediated | The model selected by the Phase 4 rubric has no scoring-ready artefact, so every downstream figure comes from the model that lost |

## RISKOS_HAZ_001 — Discrete-time hazard with competing risks

| Field | Value |
| --- | --- |
| Family | cloglog_discrete_hazard |
| Inventory status | in_use |
| Tier | 1 — Lifetime PD is the stage-2 ECL input and drives the staging test itself, so an error moves both the provision and which loans sit in which stage. |
| Owner | project_author |
| Clearance | approved_with_conditions |
| Cleared on / review due | 2026-09-08 / 2027-09-08 |
| Monitored by | none |
| Artefact | models/hazard_fit.json |

**Purpose.** Monthly default and prepayment hazards, used to build the lifetime PD term structure the IFRS 9 stage-2 measurement and the SICR test require.

**Intended use.** Lifetime default projection over a loan's remaining term, and the origination-vintage comparator used by the SICR test.

**Prohibited use.** Not to be projected beyond the fitted age range without stating the extrapolation. Per F-014 the lifetime projection lacks a transition model, so a loan's path between delinquency states is not modelled.

**Clearance conditions.** Cleared by the project author for illustrative use with F-008 and F-014 stated alongside any lifetime figure. NOT currently monitored: no rule in conf/monitoring.yaml applies to it, which is a gap against its tier and is reported by reconciliation rather than left implicit.

**Limitations (findings register).**

| ID | Severity | Status | Title |
| --- | --- | --- | --- |
| F-008 | high | open | Lifetime PD beyond 96 months is extrapolated, not fitted |
| F-011 | high | open | The hazard specification cannot detect SICR, so IFRS 9 staging collapses to the DPD backstop |
| F-014 | high | open | A state-conditional hazard cannot be projected forward without a delinquency transition model |

## RISKOS_LGD_001 — Segment LGD with shrinkage

| Field | Value |
| --- | --- |
| Family | segment_mean_shrunk |
| Inventory status | in_use |
| Tier | 1 — Multiplies directly into ECL. A segment LGD that is wrong by 10 points moves the provision by the same proportion as a PD error of the same size. |
| Owner | project_author |
| Clearance | approved_with_conditions |
| Cleared on / review due | 2026-09-08 / 2027-09-08 |
| Monitored by | none |
| Artefact | reports/figures/lgd_segments.csv |

**Purpose.** Loss given default by LTV band and state, shrunk toward the portfolio mean, supplying the LGD term of the ECL calculation.

**Intended use.** Portfolio-level and segment-level LGD for the ECL engine.

**Prohibited use.** Not a loan-level loss forecast. Segment means with shrinkage describe a group, and applying one to an individual property overstates what the estimate supports.

**Clearance conditions.** Cleared by the project author with the denominator and bounding sensitivities reported alongside any LGD figure. Not monitored, same gap as RISKOS_HAZ_001.
