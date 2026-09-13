# RiskOS — Model Validation Report

**Generated:** 2026-09-13 from the pipeline artefacts  
**Models in scope:** RISKOS_PD_001 v1.0.0, RISKOS_PD_002 v1.0.0, RISKOS_HAZ_001 vn/a, RISKOS_LGD_001 vn/a  
**Review type:** developer validation with simulated second-line review  
**Author and reviewer:** project author (see §2 on independence)  

> The underlying portfolio is U.S. residential mortgage data, because comparable public Canadian loan-level default and loss data is not available. The project applies IFRS 9 concepts and OSFI Guideline E-23 as a methodological and governance framework relevant to Canadian financial institutions. It does not represent a regulatory implementation, does not reproduce any institution's ECL system, and makes no claim of OSFI compliance.

## 1. Executive summary

**Decision.** RISKOS_PD_001 is approved with conditions. RISKOS_HAZ_001 is approved with conditions. RISKOS_LGD_001 is approved with conditions. No model in this inventory is approved without conditions, and none is approved for real lending, underwriting, capital or provisioning decisions.

- **RISKOS_PD_001** (12-month PD scorecard (WOE logistic)): inventory status `in_use`, approval `approved_with_conditions`, review due 2027-08-31.
- **RISKOS_PD_002** (12-month PD challenger (LightGBM)): inventory status `candidate`, approval `not_approved`.
- **RISKOS_HAZ_001** (Discrete-time hazard with competing risks): inventory status `in_use`, approval `approved_with_conditions`, review due 2027-09-08.
- **RISKOS_LGD_001** (Segment LGD with shrinkage): inventory status `in_use`, approval `approved_with_conditions`, review due 2027-09-08.

**The number.** Portfolio ECL as at the last training date is $34.0M on $15.91B of exposure, a coverage ratio of 0.214%. Probability-weighted across three macro scenarios it is $60.1M. The severe-stress scenario alone spans $45.6M to $236.9M at 95% confidence, a factor of 5.2 end to end. That width is the correct representation of what the estimation sample can support, not a presentational weakness.

**Three findings qualify every figure in this report.**
1. **Calibration fails under regime change (F-005, F-006).** Through 2007-2009 the scorecard predicts 0.68% where 2.17% occurred, an observed-over-expected ratio of 3.21; the challenger's is 2.80. Discrimination held (Gini 0.638 and 0.665). A model selected on AUC would have passed while understating the provision roughly threefold.
2. **The macro overlay recovers half of that and cannot recover the rest (F-009).** Fitted on the pre-crisis window and fed the realised 2008-2009 economy, it under-predicts the default rate by a mean factor of 2.04. The estimation sample contains no house-price decline, so no estimator can recover the sensitivity. This is the central methodological limitation of the project.
3. **Drift monitoring is blind to this failure mode (F-015).** Over the same crisis quarters the worst score PSI is RISKOS_PD_001 0.0113, RISKOS_PD_002 0.0112, an order of magnitude inside the stable band, for both model families. The only control that detected the deterioration was calibration monitoring, which is structurally twelve months late.

**Findings register.** 32 findings, 14 open, 8 of them high or critical. Most open findings are conclusions this project exists to report rather than defects awaiting a fix; §13 distinguishes the two. Two critical defects (R-012, R-013) were found and remediated before this report.

## 2. Scope, independence, and intended use

**Scope.** The underlying portfolio is U.S. residential mortgage data, because comparable public Canadian loan-level default and loss data is not available. The project applies IFRS 9 concepts and OSFI Guideline E-23 as a methodological and governance framework relevant to Canadian financial institutions. It does not represent a regulatory implementation, does not reproduce any institution's ECL system, and makes no claim of OSFI compliance.

**Independence.** This project is built by one person. It cannot claim the organisational independence OSFI E-23 expects between model development and model validation. The review artefact is therefore a *developer validation with simulated second-line review*: the same discipline of pre-committed rules, a findings register populated at discovery, approvals with conditions and review dates, and reconciliation of the inventory against what is on disk — without the independent reviewer. Every tier-1 approval in §12 records this as a condition, not a waiver.

**Intended use.** Illustrative and educational. Not for real lending or underwriting decisions, not for regulatory capital or provisioning, and not applicable to any Canadian portfolio without redevelopment on Canadian data.

**Risk tiering.** Every model that feeds the ECL calculation is tier 1, because the ECL figure is the one a reader would be tempted to quote.

| Tier | Definition |
| --- | --- |
| 1 | Material to a reported financial figure OR acts autonomously on customers. Requires independent validation before approval, ongoing monitoring against committed thresholds, and annual review. |
| 2 | Feeds a material figure indirectly, or supports a decision a human makes. Requires documented developer validation, ongoing monitoring, and review every two years. |
| 3 | Exploratory or internal-analysis only, with no path to a reported figure or a customer decision. Documented, inventoried, and not monitored. |

## 3. Data, target, and experimental design

**Source.** Freddie Mac Single-Family Loan-Level Dataset (SFLLD), sample variant, manual access. Column layout transcribed from *Single Family Loan-Level Dataset General User Guide, Release 47, July 2026*; ingest refuses to run without a version stamp. Raw and derived loan-level data are never committed.

**Vintages.** Core 1999-2012 and benign contrast 2015-2019. 19 vintages ingested: 950,000 loans, 57,422,329 loan-months, per the ingest manifest.

**Default definition (locked before results).** A loan is in default if it is 90 or more days past due, or if it terminates via a credit event: third-party sale, short sale, REO disposition, note sale, or charge-off. Credit-event terminations: third_party_sale, short_sale_or_charge_off, reo_disposition, whole_loan_sale. Excluded and not counted as good: defect_prior_to_credit_event. The whole-loan-sale mapping is a judgement rather than a transcription and is recorded as F-002.

**Design.** Observation-cohort panel: at each quarter-end, every loan alive and not already 90+ days past due is labelled on whether it defaults in the next twelve months. Rows whose outcome window is not observable are dropped, never imputed. Prepayment is labelled zero and flagged as a competing risk. Splits are out-of-time and loan-disjoint by seeded hash (F-003 records the sample-size cost).

| Split | Observation window | Rows | Defaults | Default rate |
| --- | --- | --- | --- | --- |
| train | 1999-Q1 to 2006-Q4 | 2,046,874 | 13,513 | 0.660% |
| validation_in_time | 1999-Q1 to 2006-Q4 | 517,080 | 3,497 | 0.676% |
| oot_stress | 2007-Q1 to 2009-Q4 | 342,603 | 7,446 | 2.173% |
| oot_benign | 2015-Q1 to 2019-Q4 | 536,028 | 5,416 | 1.010% |
| oot_benign_ex_forbearance | — | 533,056 | 2,444 | 0.458% |

**F-001, COVID forbearance.** The benign split's default rate of 1.010% is inflated by CARES Act forbearance: borrowers permitted to stop paying register as 90+ days past due without a credit event, and 2019 observation windows close in 2020. Excluding forbearance-driven defaults the rate is 0.458%. The definition was not amended; every benign-split metric is reported both ways.

## 4. Model development

**Candidate pool.** 27 features survive leakage control; both model families receive the same pool (conf/models.yaml). For the scorecard, 18 pass the Information Value floor. 2 tripped the 0.9 leakage tripwire (current_loan_delinquency_status, credit_score) and were cleared individually with written evidence rather than by raising the ceiling; F-004 records that delinquency status, while legitimate, is mechanically close to the target.

| Feature | IV | Band | Bins | Selected |
| --- | --- | --- | --- | --- |
| current_loan_delinquency_status | 1.6840 | suspiciously_strong | 2 | True |
| credit_score | 1.1543 | suspiciously_strong | 6 | True |
| original_interest_rate | 0.4254 | strong | 6 | True |
| original_ltv | 0.4245 | strong | 6 | True |
| original_cltv | 0.3747 | strong | 6 | True |
| mi_percentage | 0.3149 | strong | 5 | True |
| original_loan_term | 0.2127 | medium | 4 | True |
| remaining_months_to_legal_maturity | 0.2104 | medium | 4 | True |
| number_of_borrowers | 0.1766 | medium | 2 | True |
| property_state | 0.1643 | medium | 6 | True |

**Collinearity pruning** at a 0.95 ceiling on the WOE matrix removed remaining_months_to_legal_maturity (r = 0.959 with original_loan_term).

**Wrong-sign elimination** removed first_time_homebuyer_flag, loan_purpose, iteratively and worst first: a positive coefficient on a WOE feature awards more points for worse credit. Final scorecard: **15 features**, all coefficients negative, monotone constraints verified in the fitted bins.

| Feature | Coefficient | Points per WOE unit | Sign OK |
| --- | --- | --- | --- |
| current_loan_delinquency_status | -0.7778 | 22.44 | True |
| number_of_borrowers | -0.7560 | 21.81 | True |
| property_state | -0.6787 | 19.58 | True |
| credit_score | -0.6423 | 18.53 | True |
| original_loan_term | -0.3946 | 11.39 | True |
| loan_age | -0.3459 | 9.98 | True |
| original_dti | -0.3434 | 9.91 | True |
| channel | -0.3263 | 9.41 | True |
| original_interest_rate | -0.3259 | 9.40 | True |
| property_type | -0.2462 | 7.10 | True |
| original_ltv | -0.2027 | 5.85 | True |
| original_cltv | -0.1858 | 5.36 | True |
| original_upb | -0.1663 | 4.80 | True |
| current_actual_upb | -0.0795 | 2.29 | True |
| mi_percentage | -0.0541 | 1.56 | True |

**Challenger.** LightGBM on raw features with native categorical handling, a fixed grid of 6 combinations (num_leaves [15, 31, 63], max_depth [4, 6]), early stopping on in-time validation, and the same monotone constraints as the scorecard on 5 features. No unbounded search.

| max_depth | num_leaves | Validation AUC | Best iteration |
| --- | --- | --- | --- |
| 6 | 15 | 0.8937 | 178 |
| 4 | 31 | 0.8936 | 193 |
| 4 | 63 | 0.8936 | 193 |
| 4 | 15 | 0.8935 | 202 |
| 6 | 31 | 0.8931 | 133 |
| 6 | 63 | 0.8929 | 134 |

Selected hyperparameters: max_depth=6, num_leaves=15. Constraints silently ignored: none.

## 5. Performance: discrimination held, calibration collapsed

**All four splits, both models, uncalibrated.** Read discrimination and calibration together: the finding is in the difference between how they degrade.

| Model | Split | Rows | Defaults | AUC | Gini | Observed | Predicted | O/E | Brier | Reliability | Score PSI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scorecard | train | 2,046,874 | 13,513 | 0.894 | 0.787 | 0.660% | 0.662% | 1.00 | 0.00599 | 8.29e-07 | — |
| scorecard | validation_in_time | 517,080 | 3,497 | 0.892 | 0.784 | 0.676% | 0.654% | 1.03 | 0.00622 | 1.39e-06 | 0.0002 |
| scorecard | oot_stress | 342,603 | 7,446 | 0.819 | 0.638 | 2.173% | 0.676% | 3.21 | 0.01973 | 5.35e-04 | 0.0019 |
| scorecard | oot_benign | 536,028 | 5,416 | 0.768 | 0.535 | 1.010% | 0.324% | 3.12 | 0.00974 | 8.21e-05 | 0.2128 |
| lightgbm | train | 2,046,874 | 13,513 | 0.916 | 0.832 | 0.660% | 0.659% | 1.00 | 0.00555 | 1.31e-06 | — |
| lightgbm | validation_in_time | 517,080 | 3,497 | 0.894 | 0.787 | 0.676% | 0.659% | 1.03 | 0.00590 | 5.23e-07 | 0.0002 |
| lightgbm | oot_stress | 342,603 | 7,446 | 0.832 | 0.665 | 2.173% | 0.775% | 2.80 | 0.01865 | 4.34e-04 | 0.0004 |
| lightgbm | oot_benign | 536,028 | 5,416 | 0.782 | 0.564 | 1.010% | 0.370% | 2.73 | 0.00947 | 6.76e-05 | 0.1407 |

**Recalibration on the stress split.** Isotonic and Platt are fitted on in-time validation. Both are monotone, so ranking is unchanged; both act on the level, and the level they learned is the pre-crisis one.

| Model | Calibration | O/E | Gini | Brier | Reliability |
| --- | --- | --- | --- | --- | --- |
| scorecard | uncalibrated | 3.21 | 0.6382 | 0.01973 | 5.35e-04 |
| scorecard | isotonic | 3.03 | 0.6366 | 0.01943 | 4.67e-04 |
| scorecard | platt | 3.11 | 0.6382 | 0.01971 | 5.21e-04 |
| lightgbm | uncalibrated | 2.80 | 0.6648 | 0.01865 | 4.34e-04 |
| lightgbm | isotonic | 2.76 | 0.6634 | 0.01864 | 4.47e-04 |
| lightgbm | platt | 2.75 | 0.6648 | 0.01865 | 4.25e-04 |

Recalibration corrects a level that is systematically wrong on data you have. It cannot correct a level that is wrong because the economy changed. This table is the evidence that the Phase 5 macro overlay is necessary rather than decorative (F-006).

**Reliability by predicted-PD decile, Scorecard, 2007-2009.** Wilson intervals. Every decile observes more default than it predicts; the miscalibration is systematic, not concentrated.

| Decile | Rows | Defaults | Predicted | Observed | CI low | CI high | Within CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 34,263 | 26 | 0.038% | 0.076% | 0.052% | 0.111% | False |
| 2 | 34,259 | 84 | 0.067% | 0.245% | 0.198% | 0.303% | False |
| 3 | 34,259 | 149 | 0.096% | 0.435% | 0.371% | 0.510% | False |
| 4 | 34,261 | 222 | 0.131% | 0.648% | 0.568% | 0.739% | False |
| 5 | 34,261 | 358 | 0.176% | 1.045% | 0.943% | 1.158% | False |
| 6 | 34,262 | 456 | 0.236% | 1.331% | 1.215% | 1.458% | False |
| 7 | 34,260 | 602 | 0.325% | 1.757% | 1.623% | 1.902% | False |
| 8 | 34,260 | 805 | 0.470% | 2.350% | 2.195% | 2.515% | False |
| 9 | 34,258 | 1,030 | 0.737% | 3.007% | 2.831% | 3.193% | False |
| 10 | 34,260 | 3,714 | 4.484% | 10.841% | 10.516% | 11.174% | False |

**Reliability by predicted-PD decile, LightGBM, 2007-2009.** Wilson intervals. Every decile observes more default than it predicts; the miscalibration is systematic, not concentrated.

| Decile | Rows | Defaults | Predicted | Observed | CI low | CI high | Within CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 34,265 | 30 | 0.027% | 0.088% | 0.061% | 0.125% | False |
| 2 | 34,256 | 73 | 0.047% | 0.213% | 0.170% | 0.268% | False |
| 3 | 34,260 | 115 | 0.070% | 0.336% | 0.280% | 0.403% | False |
| 4 | 34,261 | 189 | 0.101% | 0.552% | 0.479% | 0.636% | False |
| 5 | 34,260 | 276 | 0.141% | 0.806% | 0.716% | 0.906% | False |
| 6 | 34,260 | 434 | 0.196% | 1.267% | 1.154% | 1.391% | False |
| 7 | 34,261 | 597 | 0.278% | 1.743% | 1.609% | 1.887% | False |
| 8 | 34,259 | 808 | 0.423% | 2.359% | 2.203% | 2.525% | False |
| 9 | 34,260 | 1,097 | 0.731% | 3.202% | 3.021% | 3.394% | False |
| 10 | 34,261 | 3,827 | 5.739% | 11.170% | 10.841% | 11.508% | False |

## 6. Champion-challenger selection

**The rubric was committed before the challenger was fitted** (conf/models.yaml, verifiable in git history). Weights: calibration 0.35, stability 0.25, discrimination 0.2, explainability 0.15, latency 0.05. Calibration is weighted highest because ECL is a money number.

| candidate | calibration (w=0.35) | stability (w=0.25) | discrimination (w=0.2) | explainability (w=0.15) | latency (w=0.05) | weighted_total |
| --- | --- | --- | --- | --- | --- | --- |
| lightgbm | 0.0617 | 0.4371 | 0.6648 | 0.6000 | 0.9930 | 0.4035 |
| scorecard | 0.0000 | 0.1486 | 0.6382 | 1.0000 | 0.9998 | 0.3648 |

**Decision recorded:** lightgbm scores 0.4035 against 0.3648 for scorecard.

**F-007 sensitivity.** The rubric scored SHAP as approximate attribution (0.6); measurement shows TreeSHAP reconstructs the margin exactly. The rule was applied as written and the sensitivity reported: equalising explainability at 1.0 for both candidates leaves lightgbm selected.

| Candidate | As committed | Explainability equalised |
| --- | --- | --- |
| scorecard | 0.3648 | 0.3648 |
| lightgbm | 0.4035 | 0.4635 |

**Explanation comparison** on 10 stress-period borrowers: top-driver agreement 100%, top-three overlap 0.73. Reconstruction error: scorecard points 1.14e-13, SHAP 1.15e-14. Both mechanisms are exact; the genuine difference is that points are absolute, on a human scale and enumerable in advance, while SHAP is relative to a population baseline in log-odds.

**The condition that qualifies the decision.** Both candidates score near zero on the most heavily weighted dimension. The rubric selected the less-bad of two models that both fail the criterion that matters most for provisioning (F-006). The selected model is recorded in the inventory as a candidate and is not approved; see §12 and F-018.

## 7. Lifetime PD: the hazard term structure

**Specification.** Two cause-specific discrete-time hazards, default and prepayment, complementary log-log link, loan-age baseline as band dummies, origination covariates plus mark-to-market LTV and amortisation ratio lagged one month (R-009), standard errors clustered by calendar month. Prepayment is a competing risk: lifetime PD is the sum of marginal defaults each weighted by surviving both risks.

Fitted on 2,381,686 loan-months: 1,251 defaults and 45,543 prepayments.

**Empirical seasoning.** The reason the 12-month PD cannot be chained.

| Age band | Loan-months | Default hazard | Prepayment hazard |
| --- | --- | --- | --- |
| age_00_06 | 573,081 | 0.000154 | 0.0063 |
| age_06_12 | 444,636 | 0.000436 | 0.0171 |
| age_12_24 | 648,663 | 0.000581 | 0.0249 |
| age_24_36 | 377,906 | 0.000733 | 0.0249 |
| age_36_60 | 274,521 | 0.000962 | 0.0274 |
| age_60_96 | 62,879 | 0.000811 | 0.0194 |

| Age band | Fitted hazard ratio vs 0-6 months |
| --- | --- |
| age_06_12 | 2.78 |
| age_12_24 | 3.73 |
| age_24_36 | 4.85 |
| age_36_60 | 5.63 |
| age_60_96 | 4.69 |

**Term structure against naive chaining.** Chaining repeats the 12-month PD; the hazard integrates the seasoning curve and the competing risk.

| Month | Cumulative default (hazard) | Survival | Naive chained |
| --- | --- | --- | --- |
| 12 | 0.57% | 78.8% | 0.63% |
| 24 | 1.01% | 62.1% | 1.25% |
| 60 | 1.86% | 30.4% | 3.10% |
| 96 | 2.27% | 14.9% | 4.91% |
| 120 | 2.43% | 9.3% | 6.10% |
| 180 | 2.60% | 2.8% | 9.02% |
| 240 | 2.65% | 0.9% | 11.84% |
| 300 | 2.67% | 0.3% | 14.57% |

At month 300 chaining reports 14.57% against the hazard's 2.67%, an overstatement of 5.5x. Beyond 96 months the hazard is an extrapolation of the oldest estimable band (F-008).

**Reconciliation, the acceptance criterion.** Hazard-implied 12-month PD against the observed 12-month default rate on the same panel rows.

| Specification | Terms | Hazard-implied 12m PD | Observed | Ratio |
| --- | --- | --- | --- | --- |
| primary | 14 | 0.617% | 0.694% | 0.889 |
| delinquency_banded | 16 | 0.130% | 0.694% | 0.188 |

The delinquency-banded specification reproduces the empirical hazard in every state and still forecasts worse in aggregate, because projection holds the state fixed. It is reported as a diagnostic and not promoted; promoting it needs a delinquency transition model (F-012, F-014).

## 8. Loss given default

**Definition.** Actual loss divided by balance at default, from Freddie Mac's disclosed loss at disposition. **Not clipped to [0, 1].**

18,998 observed defaults with a disclosed loss. Median 0.46, interquartile 0.22-0.71. 3.7% of losses are below zero (recoveries exceeded the balance) and 8.0% above one (accrued interest and expenses exceeded it). Both are economically real and both are kept. The raw mean is 42.6, and the reason is a degenerate denominator, not a loss.

| Band | n | Median balance | Min balance | Median loss | Median expenses |
| --- | --- | --- | --- | --- | --- |
| LGD 0-1 | 16,774 | $148,014 | $1,013.25 | $59,369 | $11,378 |
| LGD < 0 | 711 | $133,435 | $245.31 | $-2,589 | $6,806 |
| LGD 1-2 | 1,478 | $72,995 | $111.09 | $84,842 | $17,252 |
| LGD 2-5 | 29 | $15,462 | $1,079.53 | $35,247 | $26,652 |
| LGD > 5 | 6 | $113 | $0.01 | $8,216 | $4,830 |

**Denominator floor (LGD_001, $1,000).** A materiality floor on the balance removes observations where the ratio is undefined in practice. Bounding the ratio would alter observations that are real; they are different operations.

| Balance floor | Kept | Dropped | Mean LGD | Median | p99 |
| --- | --- | --- | --- | --- | --- |
| $0 | 18,998 | 0.000% | 42.6217 | 0.4603 | 1.478 |
| $100 | 18,995 | 0.016% | 0.4926 | 0.4602 | 1.474 |
| $500 | 18,990 | 0.042% | 0.4877 | 0.4602 | 1.472 |
| $1,000 | 18,989 | 0.047% | 0.4877 | 0.4602 | 1.472 |
| $2,500 | 18,976 | 0.116% | 0.4869 | 0.4598 | 1.464 |
| $5,000 | 18,964 | 0.179% | 0.4865 | 0.4595 | 1.455 |
| $10,000 | 18,952 | 0.242% | 0.4858 | 0.4593 | 1.446 |
| $25,000 | 18,744 | 1.337% | 0.4827 | 0.4579 | 1.382 |

**Bounded variant (LGD_002), reported alongside and not applied:** unbounded 0.4877; bounded_0_1 0.4729.

**Segmentation.** 304 cells of LTV band by state, each shrunk toward the portfolio mean of 0.4877 with credibility n / (n + 50) (LGD_003).

## 9. Expected credit loss and staging

**As at the last training date, 112,266 loans, $15.91B exposure.** Each loan is projected through the fitted hazards over its own remaining term for 12-month, lifetime and origination-vintage PD (F-010). Stage 1 is measured over twelve months, stages 2 and 3 over the remaining lifetime, discounted at the loan's own rate mid-period.

| Stage | Loans | EAD | Mean PD | Mean LGD | ECL | Share of ECL | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 110,710 | $15.71B | 0.45% | 0.499 | $31.6M | 93.0% | 0.201% |
| 2 | 1,556 | $201.0M | 6.24% | 0.498 | $2.4M | 7.0% | 1.181% |

**Portfolio ECL $34.0M, coverage 0.214%.** Stage 2 holds 1.3% of exposure and 7.0% of the provision at a mean PD 14x stage 1. That ratio is the most direct evidence the loan-level alignment is correct: before R-012 was fixed, every loan was provisioned on another loan's PD and the ratio was diluted to about a third of this.

**Staging sensitivity (ECL_001, threshold 2.0, backstop 30 days).**

| SICR threshold | Stage 1 | Stage 2 | Stage 3 | Stage 2 share |
| --- | --- | --- | --- | --- |
| 1.5 | 110,142 | 2,124 | 0 | 1.89% |
| 2.0 | 110,710 | 1,556 | 0 | 1.39% |
| 2.5 | 110,741 | 1,525 | 0 | 1.36% |
| 3.0 | 110,742 | 1,524 | 0 | 1.36% |
| 4.0 | 110,742 | 1,524 | 0 | 1.36% |

1,524 loans are the days-past-due backstop; the ratio test adds 32 at the registered threshold. The test is live but barely, because the hazard's only inputs that change after origination are age and the mark-to-market covariates, and on a 1999-2006 book house prices only rose (F-011).

## 10. Macroeconomic overlay and the crisis backtest

**Two-stage by design.** The loan-level model carries idiosyncratic risk; the cycle enters as a shift in logit space from a portfolio-level regression of the quarterly default rate on unemployment and year-on-year house-price change. At most two covariates, Newey-West standard errors, intervals propagated into ECL as a range.

**Estimation window.** Requested 1999Q1-2006Q4; effective 2001Q1-2006Q4, 24 observations, because the house-price index begins 2000Q1 and the year-on-year change needs four prior quarters. The register recorded 32 until F-019 corrected it.

| Window | n | Term | Coefficient | CI low | CI high | Significant | R² |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1999-2006 | 24 | const | -5.3826 | -6.1135 | -4.6517 | True | 0.513 |
| 1999-2006 | 24 | unemployment | 0.1195 | -0.0085 | 0.2475 | False | 0.513 |
| 1999-2006 | 24 | hpi_yoy | -0.0193 | -0.0279 | -0.0108 | True | 0.513 |
| 1999-2019 | 56 | const | -5.1685 | -6.1740 | -4.1630 | True | 0.586 |
| 1999-2019 | 56 | unemployment | 0.1159 | -0.0390 | 0.2708 | False | 0.586 |
| 1999-2019 | 56 | hpi_yoy | -0.0424 | -0.0552 | -0.0295 | True | 0.586 |

**Scenarios** (conf/scenarios.yaml), with the ECL each implies.

| Scenario | Weight | Unemployment | HPI y/y | PD multiplier | Extrapolates | ECL | 95% low | 95% high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0.50 | 6.0% | +2.0% | 1.31 | False | $44.2M | $37.4M | $52.3M |
| mild_recession | 0.35 | 8.0% | -5.0% | 1.91 | True | $63.5M | $39.6M | $101.2M |
| severe_stress | 0.15 | 10.0% | -20.0% | 3.24 | True | $105.2M | $45.6M | $236.9M |

**Probability-weighted ECL: $60.1M.** Staging is held at the base assignment across scenarios (R-008).

**Backtest.** The pre-crisis fit, fed the realised 2008-2009 economy, against the realised default rate.

| Quarter | Unemployment | HPI y/y | Predicted | Observed | Ratio |
| --- | --- | --- | --- | --- | --- |
| 2008-01 | 5.0% | -12.4% | 0.92% | 1.81% | 1.97 |
| 2008-04 | 5.3% | -15.5% | 1.02% | 2.23% | 2.19 |
| 2008-07 | 6.0% | -17.0% | 1.13% | 2.58% | 2.28 |
| 2008-10 | 6.9% | -18.3% | 1.28% | 2.96% | 2.31 |
| 2009-01 | 8.3% | -18.6% | 1.52% | 3.17% | 2.08 |
| 2009-04 | 9.3% | -16.8% | 1.66% | 3.13% | 1.89 |
| 2009-07 | 9.6% | -11.5% | 1.56% | 2.89% | 1.85 |
| 2009-10 | 9.9% | -5.2% | 1.44% | 2.55% | 1.77 |

**Mean ratio of observed to predicted: 2.04 (F-009).** The overlay reduces the Phase 3 shortfall and does not close it. The reason is structural: the estimation sample contains no house-price decline. The HPI coefficient is -0.0193 on the pre-crisis window and -0.0424 on the full history, a factor of 2.19, which accounts almost exactly for the shortfall. The full-history figure is in-sample by construction and isolates the mechanism; it does not vindicate the model.

## 11. Ongoing monitoring

**Design.** Six rules committed before the run (conf/monitoring.yaml), each with a metric, thresholds, severity, owner and required action. Leading indicators (score PSI, feature CSI) are knowable at scoring time; lagging indicators (observed/expected, Gini, Brier reliability) need the twelve-month outcome window to close, and every alert carries the date it could first have been raised.

| Rule | Metric | Warn | Breach | Severity at breach | Basis |
| --- | --- | --- | --- | --- | --- |
| MON-01 | score_psi | 0.1 | 0.25 | high | PSI_001 |
| MON-02 | max_feature_csi | 0.1 | 0.25 | medium | PSI_001 |
| MON-03 | n_features_csi_significant | 0 | 2 | medium | MON_007 |
| MON-04 | observed_over_expected | [0.8, 1.25] | [0.5, 2.0] | high | MON_001 |
| MON-05 | gini | 0.1 | 0.25 | high | MON_002 |
| MON-06 | brier_reliability | 1.0 | 4.0 | medium | MON_003 |

**Both PD models, 64 quarterly windows, 6 rules.**

| Model | Family | Windows | Worst stress score PSI | Alerts | Out-of-sample breaches | Months of warning | Leading rule |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RISKOS_PD_001 | woe_scorecard | 64 | 0.0113 | 262 | 105 | 12.0 | MON-02 |
| RISKOS_PD_002 | lightgbm | 64 | 0.0112 | 246 | 104 | 12.0 | MON-02 |

**F-015.** Score PSI stays an order of magnitude inside the stable band through the entire crisis, for both families, while calibration collapses. PSI compares input and output distributions; what changed in 2008 was the mapping from characteristics to default, which no input-side comparison can see. Drift monitoring gave zero months of warning on the failure it exists to catch.

**Rule false-positive rates on the development sample.** Every in-sample window is one on which the model is by construction working, so a firing there is a false positive. Thresholds were left as committed and the rates published rather than tuned (F-016, F-017).

| Model | Rule | In-sample firing rate |
| --- | --- | --- |
| RISKOS_PD_001 | MON-01 | 15.6% |
| RISKOS_PD_001 | MON-02 | 90.6% |
| RISKOS_PD_001 | MON-03 | 87.5% |
| RISKOS_PD_001 | MON-04 | 15.6% |
| RISKOS_PD_001 | MON-05 | 0.0% |
| RISKOS_PD_001 | MON-06 | 76.7% |
| RISKOS_PD_002 | MON-01 | 9.4% |
| RISKOS_PD_002 | MON-02 | 90.6% |
| RISKOS_PD_002 | MON-03 | 87.5% |
| RISKOS_PD_002 | MON-04 | 12.5% |
| RISKOS_PD_002 | MON-05 | 0.0% |
| RISKOS_PD_002 | MON-06 | 46.7% |

**Detection-gap caveat, as reported by the run:** rule MON-02 also fires on 91% of development-sample windows, where the model is by construction working. Its early firing is not evidence of detection; treat the warning it bought as zero until the rule is re-specified.

## 12. Governance: inventory, reconciliation, and serving

**Inventory** (governance/model_inventory.yaml). Judgements only; everything derivable from an artefact is read from the artefact at reconciliation time.

| Model | Family | Status | Tier | Approval | Review due | Monitored by | Limitations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RISKOS_PD_001 | woe_scorecard | in_use | 1 | approved_with_conditions | 2027-08-31 | MON-01, MON-02, MON-03, MON-04, MON-05, MON-06 | F-004, F-005, F-015, F-018 |
| RISKOS_PD_002 | lightgbm | candidate | 1 | not_approved | — | MON-01, MON-02, MON-03, MON-04, MON-05, MON-06 | F-006, F-007, F-018 |
| RISKOS_HAZ_001 | cloglog_discrete_hazard | in_use | 1 | approved_with_conditions | 2027-09-08 | none | F-008, F-011, F-014 |
| RISKOS_LGD_001 | segment_mean_shrunk | in_use | 1 | approved_with_conditions | 2027-09-08 | none | none |

**RISKOS_PD_001 — conditions of approval.** Approved for illustrative use only, on the explicit condition that the calibration failure recorded as F-005 is stated wherever an output is reported. A developer validation with a simulated second-line review is not independent validation, and tier 1 would ordinarily require the latter; that gap is a limitation of this review, recorded here rather than waived.

**RISKOS_PD_002 — conditions of approval.** An artefact now exists (F-018 remediated) and the model is monitored on the same rulebook as RISKOS_PD_001, but it remains not approved. A model that can be loaded is not thereby validated, and nothing in this project has performed the independent validation a tier 1 rating would require. Approval would also have to address F-006: the rubric's own calibration dimension scores this model 0.062 out of 1.0.

**RISKOS_HAZ_001 — conditions of approval.** Approved for illustrative use with F-008 and F-014 stated alongside any lifetime figure. NOT currently monitored: no rule in conf/monitoring.yaml applies to it, which is a gap against its tier and is reported by reconciliation rather than left implicit.

**RISKOS_LGD_001 — conditions of approval.** Approved with the denominator and bounding sensitivities reported alongside any LGD figure. Not monitored, same gap as RISKOS_HAZ_001.

**Reconciliation** (`riskos registry`) checks the inventory against the artefacts on disk and the findings register: a model in use with no loadable artefact, an artefact nobody inventoried, a bundle fitted against changed configuration, a cited finding that does not exist, an open high-severity finding absent from a model's limitations, a tier-1 model with no monitoring rule, an approval past its review date.

| Severity | Check | Model | Detail |
| --- | --- | --- | --- |
| medium | tier_one_unmonitored | RISKOS_HAZ_001 | RISKOS_HAZ_001 is tier 1 and in_use but no monitoring rule is pointed at it |
| medium | tier_one_unmonitored | RISKOS_LGD_001 | RISKOS_LGD_001 is tier 1 and in_use but no monitoring rule is pointed at it |

The first reconciliation returned a high-severity discrepancy: the rubric-selected model had no scoring-ready artefact, so every downstream figure had come from the runner-up (F-018, remediated). The remaining discrepancies are genuine gaps, reported rather than closed.

**Serving.** The scoring service takes no model path. It serves the model the inventory records as in use, refuses a candidate, an unapproved or lapsed approval, or an artefact whose manifest disagrees with the inventory, and returns the model id, version, approval status, conditions and recorded limitations with every score.

## 13. Findings register

A review with no findings is not credible. Findings are recorded when found, not assembled at the end, and remediated entries keep their original text with the remediation appended. F-prefixed findings are raised against the models and data; R-prefixed findings are defects found by code review after a phase was reported complete.

| Severity | Open | Remediated | Total |
| --- | --- | --- | --- |
| critical | 0 | 2 | 2 |
| high | 8 | 7 | 15 |
| medium | 4 | 8 | 12 |
| low | 2 | 1 | 3 |

**Open findings that are conclusions, not defects.** F-001 (a policy intervention in the data), F-003 (a deliberate design cost), F-005, F-006 and F-009 (the calibration and overlay results the project exists to report), F-014 (disclosed, with the correct reduced-form choice in place) and F-017 (a known-defective low-severity threshold) carry an accepted residual risk. **Open findings that imply work:** F-004 (report the variant without delinquency), F-008 (estimate the long-horizon hazard), F-011 (a specification using delinquency state), F-015 (a leading indicator that can see a changing relationship), F-016 (split time-structural features out of CSI).

| ID | Severity | Status | Phase | Title |
| --- | --- | --- | --- | --- |
| F-001 | high | open | 2 | COVID-19 payment forbearance contaminates the OOT-benign split |
| F-002 | medium | open | 1 | Zero balance code 15 mapped to a credit event by inference, not transcription |
| F-003 | medium | open | 2 | Loan-disjoint splits discard observations and reduce OOT sample size |
| F-004 | medium | open | 3 | Delinquency status at the observation date dominates the scorecard |
| F-005 | high | open | 3 | Scorecard materially under-predicts default under regime change |
| R-001 | high | remediated | 3 | Look-ahead in the forbearance flag caused by DuckDB arg_min null semantics |
| R-002 | medium | remediated | 3 | CSI was blind to changes in missingness |
| R-003 | medium | remediated | 3 | Principal-driver explanations depended on the scoring batch |
| R-004 | low | remediated | 3 | CLI logging flags had no effect |
| F-006 | high | open | 4 | Both PD candidates fail the calibration dimension under stress |
| F-007 | low | open | 4 | The selection rubric measured the wrong proxy for explainability |
| F-008 | high | open | 5 | Lifetime PD beyond 96 months is extrapolated, not fitted |
| F-009 | high | open | 5 | The macro overlay under-predicts the crisis by half, and the reason is structural |
| F-010 | medium | remediated | 5 | The end-to-end ECL run uses a portfolio-constant PD, degenerating the SICR sensitivity |
| R-005 | high | remediated | 5 | Hazard imputation used the scoring batch median rather than the fitting median |
| R-006 | medium | remediated | 5 | Competing-risk convention was unstated and asymmetric |
| R-007 | high | remediated | 5 | Phase 5 was reported as substantially done while unreachable from the CLI |
| R-008 | high | remediated | 5 | Scenario ECL bypassed staging, lifetime PD and discounting |
| F-011 | high | open | 5 | The hazard specification cannot detect SICR, so IFRS 9 staging collapses to the DPD backstop |
| F-012 | medium | remediated | 5 | Delinquency state cannot enter the hazard model as a linear term |
| F-013 | medium | remediated | 5 | The prepayment hazard does not converge |
| R-009 | high | remediated | 5 | Terminal-state contamination — the risk set's own balance revealed its own outcome |
| F-014 | high | open | 5 | A state-conditional hazard cannot be projected forward without a delinquency transition model |
| R-010 | high | remediated | 5 | The Phase 5 acceptance criterion had been failing since the time-varying covariates were added |
| R-011 | medium | remediated | 5 | An assumption was cited in code while absent from the register |
| R-012 | critical | remediated | 5 | Per-loan PDs were permuted against the portfolio, so every loan was provisioned on another loan's probability of default |
| R-013 | critical | remediated | 5 | A gitignore pattern kept the entire modelling package out of version control |
| F-015 | high | open | 6 | Score PSI is blind to the 2008 regime change, so drift monitoring gives zero months of warning on the failure it exists to catch |
| F-016 | medium | open | 6 | CSI against a pooled multi-year reference measures portfolio seasoning, not drift, and fires on 91% of development-sample windows |
| F-017 | low | open | 6 | A relative threshold on the Brier reliability term is unusable because the development-sample baseline is near zero |
| F-018 | high | remediated | 6 | The model selected by the Phase 4 rubric has no scoring-ready artefact, so every downstream figure comes from the model that lost |
| F-019 | medium | remediated | 6 | The macro overlay is fitted on 24 quarters, not the 32 the register, the scenario config and the module docstring all assert |

## 14. Assumptions register

27 registered assumptions (convention 5, empirical 1, judgement 21). Every numeric assumption lives in conf/assumptions.yaml with a source, a materiality statement and a sensitivity test; tests bind the register to the constants in code and to the monitoring rulebook, and F-019 added a test binding it to a fitted artefact.

| ID | Source | Description |
| --- | --- | --- |
| PSI_001 | convention | PSI/CSI stability bands |
| PSI_002 | judgement | Epsilon substituted for an empty PSI/CSI bin |
| PSI_003 | convention | Number of PSI bins, taken from training-set score deciles |
| CAL_001 | convention | Confidence level for binomial intervals on the reliability curve |
| CAL_002 | judgement | Number of bins for the reliability curve and Brier decomposition |
| PANEL_001 | judgement | Observation-cohort panel sampling target |
| PANEL_002 | judgement | Allocation of loans to splits |
| PANEL_003 | judgement | Treatment of COVID-19 forbearance in the default label |
| LGD_001 | judgement | Materiality floor on the LGD denominator (UPB at default) |
| LGD_002 | judgement | Optional [0, 1] bounded LGD variant, reported alongside the unbounded estimate |
| LGD_003 | judgement | Credibility constant for shrinking thin LGD segments to the portfolio mean |
| EAD_001 | convention | Exposure at default for an amortising mortgage |
| HAZ_001 | judgement | Loan-level sample size for fitting the discrete-time hazard model |
| HAZ_002 | judgement | Loan-age bands forming the hazard baseline |
| HAZ_003 | judgement | Hazard model fitting window restricted to 1999-2006 |
| HAZ_004 | judgement | House-price fallback when a loan's origination predates the HPI series |
| HAZ_005 | empirical | Delinquency-state bands for the hazard specification |
| MACRO_001 | judgement | Macro overlay estimation window and specification |
| ECL_001 | judgement | Significant Increase in Credit Risk (SICR) threshold for IFRS 9 stage 2 |
| ECL_002 | convention | Discounting convention for expected credit loss |
| MON_001 | judgement | Observed-over-expected bands for the calibration monitoring rule (MON-04) |
| MON_002 | judgement | Proportional Gini decline from the development sample (MON-05) |
| MON_003 | judgement | Relative increase in the Brier reliability term (MON-06) |
| MON_004 | judgement | Minimum window size before a monitoring alert is raised or a metric computed |
| MON_005 | judgement | Monitoring window frequency and the outcome-observability lag |
| MON_006 | judgement | In-sample firing rate above which a rule is reported as unfit to claim a detection |
| MON_007 | judgement | Count of significantly drifting inputs that triggers the breadth rule (MON-03) |

## 15. Conclusion and conditions

- **RISKOS_PD_001**: approved with conditions on 2026-08-31, review due 2027-08-31. Limitations carried: F-004, F-005, F-015, F-018.
- **RISKOS_PD_002**: not approved. Limitations carried: F-006, F-007, F-018.
- **RISKOS_HAZ_001**: approved with conditions on 2026-09-08, review due 2027-09-08. Limitations carried: F-008, F-011, F-014.
- **RISKOS_LGD_001**: approved with conditions on 2026-09-08, review due 2027-09-08. Limitations carried: none.

**What would change these decisions.** An independent second-line review, which this project cannot supply. A leading indicator able to see a changing relationship between characteristics and default (F-015). A delinquency transition model, which unblocks both the better-specified hazard and the SICR test (F-014, F-011). None of these changes the central limitation: a stress overlay estimated on a sample containing no stress will under-predict stress, and the correct output is the range and the sentence, not a point.

## Appendix A. Reproducibility

Every table above is regenerated from CSV, JSON and YAML artefacts under reports/figures/, models/ and governance/. No figure is typed into this document.

| Step | Command |
| --- | --- |
| Ingest and validate | `make ingest` |
| Panel, risk set, labels | `make panel` |
| Scorecard champion | `make train` |
| Challenger, calibration, selection | `make challenger` |
| Hazard and reconciliation | `make hazard` |
| LGD, ECL, scenarios, backtest | `make ecl` |
| Monitoring | `make monitor` |
| Inventory reconciliation | `make registry` |
| This report and the model card | `make report` |
| Tests, lint, types | `make test`, `make lint` |

All artefacts read by this report were present.
