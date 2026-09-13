# Phase 5 — Hazard, LGD, ECL, and scenarios

**Status: runnable end to end.** Two cause-specific hazards, 304 LGD segments,
a staged and discounted ECL under three weighted scenarios, and a crisis
backtest that the overlay fails by half.

> Three things in this chapter matter more than the ECL number. First, the
> lifetime PD comes from a fitted hazard, not from repeating year one: chaining
> the 12-month PD out to 300 months would have overstated lifetime default by a
> factor of five. Second, the LGD is **not clipped** to [0, 1], because 11.7% of
> observed losses genuinely fall outside it. Third, the macro overlay, fed the
> real 2008–2009 economy, predicts half the defaults that happened — and the
> reason is structural, so no estimator fixes it. That last one is the central
> limitation of the whole project.

**Run it:** `uv run riskos hazard` then `uv run riskos ecl` (~10 minutes)

---

## 1. What the phase has to produce

```
ECL  =  Σ over loans of   PD × LGD × EAD × discount factor
```

Phases 3 and 4 produced a 12-month PD. IFRS 9 needs more than that, because
staging decides the horizon:

| Stage | Trigger here | PD applied |
| --- | --- | --- |
| 1 | Nothing below fired | 12-month |
| 2 | Lifetime PD more than 2.0× its origination value (ECL_001), **or** 30+ days past due (backstop) | Lifetime |
| 3 | 90+ days past due or a credit event | Lifetime |

So the phase needs a **lifetime** PD, which means a term structure; an **LGD**
from actual recoveries; an **EAD**, which for an amortising mortgage is the
balance (EAD_001); a **discount** at the loan's own rate, mid-period (ECL_002);
and a way to condition all of it on the **economy** under weighted scenarios.

Every threshold above is in `conf/assumptions.yaml` with a source and a
sensitivity test, and the SICR threshold is swept rather than asserted.

---

## 2. The hazard model: where the year-five PD comes from

A 12-month PD says nothing about year five. The tempting shortcut is to chain it
— assume next year looks like this year — and the empirical seasoning curve says
exactly why that is wrong:

| Loan age | Monthly default hazard | Monthly prepayment hazard |
| --- | --- | --- |
| 0–6 months | 0.000154 | 0.0063 |
| 6–12 | 0.000436 | 0.0171 |
| 12–24 | 0.000581 | 0.0249 |
| 24–36 | 0.000733 | 0.0249 |
| 36–60 | **0.000962** | 0.0274 |
| 60–96 | 0.000811 | 0.0194 |

Default risk rises for the first four years, peaks, then eases. Prepayment runs
twenty to thirty times higher than default at every age. A model that ignores
either shape is wrong about lifetime loss.

**Specification.** Two cause-specific discrete-time hazards, default and
prepayment, with a complementary log-log link (the discrete analogue of
proportional hazards), a loan-age baseline as band dummies (HAZ_002), origination
covariates, two time-varying covariates (mark-to-market LTV and the amortisation
ratio, both lagged one month — see R-009), and standard errors clustered by
calendar month because everyone experiences 2008 together. Fitted on a
220,000-loan sample, about 2.4 million loan-months (HAZ_001), restricted to
1999–2006 to match the 12-month model (HAZ_003).

**Prepayment is a competing risk, not censoring.** A borrower who refinances
cannot later default, so:

```
S(m)         = Π over j ≤ m of (1 − h_default(j)) · (1 − h_prepay(j))
lifetime PD  = Σ over m of  h_default(m) · S(m−1)
```

The lifetime PD is a sum of marginal defaults, each weighted by surviving *both*
risks to that month. One minus a default-only survival would credit defaults to
loans that had already prepaid. The convention for the joint-event mass is named
and selectable (R-006); the two options differ by under 1%.

### The term structure, and what chaining would have said

| Month | Cumulative default (hazard) | Survival | Naive chained 12m PD |
| --- | --- | --- | --- |
| 12 | 0.57% | 78.8% | 0.63% |
| 60 | 1.86% | 30.4% | 3.10% |
| 96 | 2.27% | 14.9% | 4.91% |
| 300 | **2.67%** | 0.3% | **14.57%** |

By month 300 only 0.3% of the original cohort is still there to default;
prepayment has removed almost everything. Chaining the 12-month PD ignores that
and reports 14.6%, **5.5 times** the hazard-implied figure. That is the number an
interviewer asking "where does the year-five PD come from" is checking you have
not produced.

### Reconciliation: the acceptance criterion

The hazard-implied 12-month PD on the panel is compared with the observed
12-month default rate on the same rows. The ratio is **0.889** on 35,613 rows.
Not 1.0, and the gap is explained: the primary specification excludes
delinquency state (see §3), so it averages over transitions it does not model.

**This criterion had been silently failing** for a week after the time-varying
covariates were added (R-010). The reconciliation read panel rows that did not
carry the new covariates, raised, and the phase read as complete because nothing
tested that `make hazard` ran to the end. The 0.89 quoted in an earlier finding
had come from a stale artefact. Fixed, regenerated, and recorded.

---

## 3. What delinquency does to a hazard model

Delinquency status is the strongest predictor in the dataset (F-004). Adding it
to the hazard as a linear term made the model degenerate — the intercept fell
from −9.3 to −27.4 and the PD for a current loan collapsed to 0.001%.

The first diagnosis was wrong, and the register keeps it (F-012). The real cause
was **target contamination**: default is *defined* as reaching 90 days past due,
and the covariate was taken in the same month, so on the fitting window the
monthly hazard was exactly 1.000000 for every row at 3+ months delinquent and
exactly 0.000000 for every current row. Perfect separation. The same defect class
had already appeared once as R-009, where the current balance in the month of
termination is zero and so revealed the termination.

Both fixes are the same fix: **lag the covariate one month**, so it means "state
at the start of the month being predicted" both when fitted and when scored.
With that, the state enters as band dummies (HAZ_005) and the banded model
reproduces the empirical hazard in every state to within 0.04%.

**And it still cannot be used.** Its 12-month reconciliation ratio is **0.188**
— it under-predicts by a factor of five — where the crude primary specification
gets 0.889. The better monthly model gives the worse forecast, because
projection holds the delinquency state fixed: a current loan is projected as
permanently current at a hazard of 0.000017 a month, when in reality 4.5% of
current loans enter delinquency within a year and the path to default runs
through that transition. Fixing it needs a **delinquency transition model**, a
monthly Markov matrix over current / 30–59 / 60–89, absorbing into default and
prepayment, projected alongside the age path. That is F-014, open, and it blocks
the SICR improvement F-011 still needs.

---

## 4. LGD: not clipped, and why that is a claim worth defending

```
LGD = actual_loss / UPB at default
```

18,998 observed defaults with a disclosed loss. Median LGD **0.46**. And:

| | Share | What it means |
| --- | --- | --- |
| LGD < 0 | 3.7% | Recoveries exceeded the balance: the property sold for more than was owed, or MI covered more than the shortfall |
| LGD > 1 | 8.0% | Accrued interest plus foreclosure expenses exceeded the balance |

Both are economically real and both are kept. `np.clip(lgd, 0, 1)` with no
analysis would have been a project-level failure.

**What is excluded, and why it is not clipping.** The raw mean is **42.6**. Three
loans have a balance at disposition below $100 while still incurring foreclosure
expenses; the most extreme is a balance of one cent against $6,573 of costs, a
ratio of 657,341. Dropping those three — 0.02% of observations — moves the mean
from 42.6 to **0.49**. That is a degenerate denominator, not a 65-million-percent
loss. The remedy is a materiality floor on the **denominator** (LGD_001, $1,000),
which removes observations where the ratio is undefined in practice. Bounding the
**ratio** would alter observations that are real. They are different operations,
and the bounded variant is reported alongside (LGD_002): it lowers the mean by
3.0%.

**Segmentation.** Six LTV bands by state, 304 cells, each shrunk toward the
portfolio mean with credibility *n / (n + 50)* (LGD_003). Florida at 61–70 LTV
on 355 loans keeps 88% of its own mean of 0.60; DC on three loans keeps 6% of
its 0.16 and lands at 0.47.

---

## 5. The ECL, as at 2006-12-01

Portfolio of 112,266 training-window loans on the last training date, scored
with the hazard model for 12-month, lifetime and origination-vintage PD, each
loan projected over its own remaining term (F-010).

| Stage | Loans | EAD | Mean PD | Mean LGD | ECL | Share of ECL |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 110,710 | $15.71B | 0.45% | 0.499 | $31.6M | 93.0% |
| 2 | 1,556 | $0.20B | **6.24%** | 0.498 | $2.4M | 7.0% |
| 3 | 0 | — | — | — | — | — |
| **Total** | 112,266 | **$15.91B** | | | **$34.0M** | **0.214%** |

Stage 2 holds 1.3% of exposure and 7% of the provision, with a mean PD
fourteen times stage 1. That ratio is the most direct evidence the loan-level
alignment is correct — see R-012 below for what it looked like when it was not.

### The staging sensitivity

| SICR threshold | Stage 2 loans |
| --- | --- |
| 1.5 | 2,124 |
| **2.0** | **1,556** |
| 2.5 | 1,525 |
| 3.0 and above | 1,524 |

1,524 loans are the 30-DPD backstop. The ratio test itself adds **32** at the
registered threshold and 600 at 1.5. The test is live, but only just, and the
reason is F-011: the hazard's only inputs that change after origination are age
and the two mark-to-market covariates, so on a 1999–2006 book where house prices
only rose, very few loans look riskier than the day they were written.

---

## 6. The macro overlay, and the half it misses

The loan-level model carries idiosyncratic risk. The cycle comes from a
portfolio-level regression of the logit of the quarterly default rate on
unemployment and year-on-year house-price change, applied as a shift in logit
space:

```
PD_scenario = sigmoid( logit(PD_base) + δ )
```

At most two covariates, Newey–West standard errors with four lags, intervals
propagated into ECL as a range (MACRO_001). Fitted on the pre-2007 window, HPI
is significant at −0.019; unemployment is **not** (+0.12, interval spanning
zero). R² 0.51.

**On 24 quarters, not 32.** The window is requested as 1999Q1–2006Q4, but the
house-price index begins in 2000 and the year-on-year change needs four prior
quarters, so the first usable observation is 2001Q1. The register, the config
comments and the module docstring all said 32 until this chapter was written
and the fit's own output disagreed (F-019). Nothing in the numbers changed; the
description of how much evidence sits behind them did, in the direction of
less.

### Scenarios

| Scenario | Weight | Unemployment | HPI y/y | PD multiplier | Extrapolates | ECL | 95% band |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Base | 0.50 | 6.0% | +2% | 1.31 | no | $44.2M | $37.4M – $52.3M |
| Mild recession | 0.35 | 8.0% | −5% | 1.91 | **yes** | $63.5M | $39.6M – $101.2M |
| Severe stress | 0.15 | 10.0% | −20% | 3.24 | **yes** | $105.2M | $45.6M – $236.9M |
| **Probability-weighted** | | | | | | **$60.1M** | |

The severe band spans a factor of five from end to end. That is not a weakness
of the presentation; it is the correct representation of what 24 benign quarters
can tell you about a crisis. Staging is held at the base assignment across
scenarios (R-008), matching the common practice of staging once and varying only
the loss measurement.

### The backtest

Feed the model fitted on 1999–2006 the **realised** 2008–2009 economy:

| Quarter | Unemployment | HPI y/y | Predicted | Observed | Ratio |
| --- | --- | --- | --- | --- | --- |
| 2008Q1 | 5.0% | −12.4% | 0.92% | 1.81% | 1.97 |
| 2008Q4 | 6.9% | −18.3% | 1.28% | 2.96% | 2.31 |
| 2009Q1 | 8.3% | −18.6% | 1.52% | 3.17% | 2.08 |
| 2009Q4 | 9.9% | −5.2% | 1.44% | 2.55% | 1.77 |

Mean ratio of observed to predicted: **2.04**. The overlay reduces the Phase 3
shortfall from 3.2× to 2.0× and does not close it.

**The reason is structural (F-009).** Year-on-year HPI in the estimation sample
runs from +1.7% to +17.0%. It contains **no house-price decline at all**. The
fitted HPI coefficient is −0.019; on the full 1999–2019 history it is −0.042, a
factor of 2.19, which accounts almost exactly for the 2.04 shortfall. No
estimation technique recovers a sensitivity the sample does not contain. A
stress overlay calibrated on benign data under-predicts stress, and the honest
output is the range, the comparison fit, and this sentence in the executive
summary.

---

## 7. Two critical defects, and how they were found

**R-012 — every loan was provisioned on another loan's PD.** The per-loan PD
function passed the portfolio through DuckDB to derive time-varying covariates,
and DuckDB does not preserve row order across a join. The PD vectors came back
permuted against the frame the caller still held. The mean PD never moved, which
is exactly why nothing flagged it; two consecutive runs on unchanged code
reporting $36.16M and $36.19M did. Corrected, the provision fell 5.9% to $34.0M
and stage-2 mean PD rose from 0.024 to 0.062 — the high-PD loans finally in
stage 2 instead of scattered across stage 1. Row order is now carried explicitly
through every query, and three regression tests fail if it is removed.

**R-013 — the modelling package was never under version control.** The pattern
`models/` in `.gitignore`, unanchored, matched `src/riskos/models/` as well as
the artefact directory at the root. Fourteen modules — the scorecard, the
challenger, calibration, both hazard modules — existed only in the working tree.
One fresh clone and the core deliverable of three phases was gone. The pattern is
now anchored, and a test runs `git check-ignore` over every source tree.

Neither is a modelling error. Both would have invalidated every claim the
repository makes about reproducibility, and both were found by the repository's
own checks rather than by a reader.

---

## Artefacts

| File | What it holds |
| --- | --- |
| `reports/figures/hazard_empirical_seasoning.csv` | Observed hazards by age band, both causes |
| `reports/figures/hazard_baseline.csv` | Fitted age-band hazard ratios |
| `reports/figures/hazard_coefficients.csv` | Covariate coefficients with clustered intervals |
| `reports/figures/hazard_term_structure.csv` | Cumulative default and survival to 300 months, against chaining |
| `reports/figures/hazard_specification_comparison.csv` | Primary vs delinquency-banded, with reconciliation |
| `reports/figures/lgd_raw_distribution.csv`, `lgd_tail_investigation.csv` | The unclipped distribution and its tails |
| `reports/figures/lgd_denominator_sensitivity.csv`, `lgd_bounding_sensitivity.csv` | LGD_001 and LGD_002 |
| `reports/figures/lgd_segments.csv` | 304 shrunk segments |
| `reports/figures/ecl_by_stage.csv`, `ecl_staging_sensitivity.csv` | The provision and the SICR sweep |
| `reports/figures/macro_coefficients.csv`, `macro_scenario_shifts.csv`, `macro_backtest.csv` | The overlay, the scenarios, the crisis test |
| `reports/figures/ecl_scenarios.csv` | Scenario ECL with bands and the weighted figure |
| `models/hazard_fit.json` | Both fitted hazards |

## Findings raised in this phase

| ID | Severity | Status | Title |
| --- | --- | --- | --- |
| F-008 | high | open | Lifetime PD beyond 96 months is extrapolated, not fitted |
| F-009 | high | open | The macro overlay under-predicts the crisis by half; the reason is structural |
| F-010 | medium | remediated | A portfolio-constant PD degenerated the SICR sensitivity |
| F-011 | high | partial | The hazard cannot detect SICR; staging collapses toward the backstop |
| F-012 | medium | remediated | Delinquency state cannot enter as a linear term (target contamination) |
| F-013 | medium | remediated | The prepayment hazard did not converge (separation, not conditioning) |
| F-014 | high | open | A state-conditional hazard cannot be projected without a transition model |
| R-005 | high | remediated | Imputation used the scoring batch's median |
| R-006 | medium | remediated | Competing-risk convention unstated |
| R-008 | high | remediated | Scenario ECL bypassed staging and discounting |
| R-009 | high | remediated | The month-of-termination balance revealed the termination |
| R-010 | high | remediated | The acceptance criterion had been failing unnoticed |
| R-012 | **critical** | remediated | Per-loan PDs permuted against the portfolio |
| R-013 | **critical** | remediated | The modelling package was gitignored |
