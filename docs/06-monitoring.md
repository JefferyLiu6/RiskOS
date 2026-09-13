# Phase 6 — Monitoring, registry, and serving

**Status: complete.** 64 quarterly windows, 2 models, 6 committed rules, an
inventory-gated scoring service, **366 tests**

> The headline finding is a negative one. **Score PSI does not move at all
> during the 2008 collapse** — it stays below 0.012 for both PD models while
> they under-predict default by factors of four and five. The drift control that
> exists to give early warning gives none, and it gives none for two different
> model families, so the failure is a property of the metric rather than of
> either model.

**Run it:** `uv run riskos monitor`, `uv run riskos registry`, `uv run riskos serve` (~2 minutes)

---

## 1. The asymmetry the whole design rests on

A monitoring function measures two kinds of thing, and they arrive at different
times.

**Drift** compares the population being scored with the population the model was
fitted on. Both sides are inputs, so the number exists the moment a window is
scored.

**Performance** compares predictions with outcomes. The label here is a
12-month forward default flag, so the outcome for an observation dated `T` is
unknown until `T+12m`. At any real reporting date, the most recent quarter whose
AUC and observed-over-expected can be computed **closed twelve months ago**.

That gap is the entire justification for watching input distributions. If
outcomes arrived instantly nobody would bother; they would watch the default
rate. They do not, so for a year the only available evidence that a model has
stopped working is that the population it is scoring has stopped looking like
the population it was fitted on.

Every performance row therefore carries `outcome_matured_at`, and every alert
carries `detectable_at` — the earliest date it could have been raised. Collapsing
those into the window date would let a monitoring pack claim, in hindsight, to
have caught something twelve months before it could possibly have known.

---

## 2. Commit the rulebook first

`conf/monitoring.yaml` fixes six rules — metric, comparator, warn and breach
thresholds, severity, owner, and required action — **before the run**. Same
discipline as the Phase 4 selection rubric, and for the same reason: a threshold
chosen after seeing the timeline is not a threshold, it is a description of what
happened dressed as a control.

| Rule | Metric | Kind | Warn | Breach |
| --- | --- | --- | --- | --- |
| MON-01 | Score PSI | leading | 0.10 | 0.25 |
| MON-02 | Worst feature CSI | leading | 0.10 | 0.25 |
| MON-03 | Count of features above the band | leading | >0 | >2 |
| MON-04 | Observed / expected | lagging | outside 0.80–1.25 | outside 0.50–2.00 |
| MON-05 | Gini, proportional fall | lagging | 10% | 25% |
| MON-06 | Brier reliability, relative rise | lagging | 100% | 400% |

Every rule names an **owner** and an **action**, and config refuses a severity
with no escalation route. A monitoring report that produces a number but names
no threshold, no owner and no action is a dashboard, not a control.

Thresholds live in `conf/assumptions.yaml` as MON_001 through MON_007, and a
test binds the register to the rulebook so the two files cannot drift apart.

---

## 3. What actually happened

Both PD models, fitted on 1999–2006, were scored forward across all 64 quarters
to 2019Q4 without refitting, on the same windows and under the same rulebook.
That is what a deployed model experiences, and running both means the comparison
is between models rather than between monitoring setups.

The tables below are the scorecard (RISKOS_PD_001) unless stated. The challenger
is compared directly in section 3.3.

### Calibration caught it. Twelve months late.

| Window | Observed | Expected | O/E | Gini | Computable from |
| --- | --- | --- | --- | --- | --- |
| in-sample 1999–2006 | — | — | 0.56–1.17 | 0.781 avg | — |
| 2007Q1 | 0.76% | 0.54% | **1.43** warn | 0.702 | 2008-03-31 |
| 2007Q4 | 1.36% | 0.65% | **2.08** breach | 0.677 | 2008-12-31 |
| 2008Q1 | 1.81% | 0.63% | 2.90 | 0.643 | 2009-03-31 |
| 2009Q1 | 3.17% | 0.67% | **4.72** | 0.614 | 2010-03-31 |

The expected rate barely moves — 0.54% to 0.67% across the whole crisis — while
the observed rate rises sixfold. The model is not confused about *who* is risky;
Gini only falls from 0.78 to 0.61. It is wrong about *how much*, and by 2009 it
is wrong by a factor of nearly five.

This is the Phase 3 finding F-005 restated on a quarterly axis, and it is the
same shape as the Phase 4 result that both candidates fail the calibration
dimension. **Discrimination degrades slowly under regime change; calibration
collapses.** A monitoring pack that led with AUC would have looked mildly
concerning throughout.

### Drift caught nothing

| Window | Score PSI | Band |
| --- | --- | --- |
| 2006Q4 (last in-sample) | 0.0057 | stable |
| 2007Q1 | 0.0064 | stable |
| 2008Q1 | 0.0049 | stable |
| 2009Q1 | 0.0021 | stable |

The highest score PSI anywhere in the 21-year run is **0.1545, at 2000Q4, inside
the training window**. Every crisis quarter is an order of magnitude below the
"stable" band. MON-01 never warns and never breaches during 2007, 2008 or 2009.

**Why, stated mechanically.** PSI compares two distributions of the model's own
output. What changed in 2008 was not the population of borrowers but the mapping
from borrower characteristics to default: the same credit score, LTV and DTI
carried several times the risk they had carried in 2004. No comparison of input
or score distributions can see a change that lives entirely in the relationship
between inputs and outcome. PSI is exactly such a comparison, so it cannot.

This is recorded as **F-015**.

### 3.3 The same thing happens to the challenger

The obvious objection is that this is a scorecard problem — a linear model on
binned inputs might well produce a stable score distribution while being wrong.
So the LightGBM challenger was put through the identical rulebook on the
identical windows.

| Quarter | Scorecard PSI | Challenger PSI | Scorecard O/E | Challenger O/E |
| --- | --- | --- | --- | --- |
| 2007Q1 | 0.0064 | 0.0037 | 1.43 | 1.37 |
| 2007Q4 | 0.0048 | 0.0012 | 2.08 | 1.93 |
| 2008Q1 | 0.0049 | 0.0013 | 2.90 | 2.63 |
| 2009Q1 | 0.0021 | 0.0006 | 4.72 | 3.91 |

A different family, a different feature treatment, a different functional form —
and the same flat PSI against the same collapsing calibration. The challenger is
marginally the better-calibrated of the two and is still wrong by a factor of
four at the peak. **The failure is a property of what PSI measures.**

This directly qualifies the Phase 4 selection rubric, which weights score PSI at
0.25 and awarded LightGBM 0.437 on that dimension against the scorecard's 0.149.
That difference is real, but on this evidence it measures how far each model's
output distribution moved between two populations, not anything about
reliability under regime change. The dimension still detects population change
and data-supply failure. The weight it carries is not supported by what it
turned out to detect here.

---

## 4. Measuring the rules instead of trusting them

Every window inside the development sample is one on which the model is, by
construction, working: it is the data the coefficients were fitted to. A rule
that fires there is producing a **false positive**, and the rate is measurable.

| Rule | Metric | Fires on in-sample windows |
| --- | --- | --- |
| MON-05 | Gini | **0%** |
| MON-01 | Score PSI | 15.6% |
| MON-04 | Observed / expected | 15.6% |
| MON-06 | Brier reliability | **76.7%** |
| MON-03 | Count above band | **87.5%** |
| MON-02 | Worst feature CSI | **90.6%** |

Three of the six rules fire on most windows where nothing is wrong. That is not
a detail — a rule that is always on is not a control, and the predictable human
response is to stop reading it, which is how real monitoring functions go blind
to the alerts that do matter.

It also poisons the headline. MON-02 is the rule that "first breached", at
2007Q1, twelve months before any lagging indicator could. Crediting it with
twelve months of warning would be false: it also breached in 1999, 2000, 2001
and almost every quarter since. **It did not detect 2007; it was already on when
2007 arrived.** The detection gap therefore reports the winning rule's
false-positive rate beside it, and prints a caveat when that rate is high. The
number and its refutation travel together.

### Why MON-02 fires constantly

`loan_age` at 1999Q1 has a CSI of **12.40**, with 10 of its 11 reference bins
empty on the window side. Loans observed in 1999Q1 are 0 to 2 months old; the
pooled 1999–2006 reference has a mean age of 19.4 months. A single early quarter
cannot overlap a pooled multi-year age distribution, so almost every bin is empty
and the reported CSI is carried by the **1e-6 epsilon floor** (PSI_002) rather
than by data.

The instrumentation built in Phase 3 for exactly this reason does the work: the
`empty_bins` count that R-002 added is what makes the artefact diagnosable
rather than merely suspicious, and `monitoring_csi_epsilon_dominated.csv` lists
every window whose headline CSI comes from a feature with more than half its bins
empty.

Recorded as **F-016**. The fix is to split the features by what CSI can say about
them — origination characteristics belong in a drift rule, time-structural
features like age and vintage belong in a portfolio-composition exhibit — not to
raise the threshold until the rate looks acceptable. Tuning a control until its
in-sample rate is comfortable is fitting the control to the data it is supposed
to police, which is the same error as choosing selection weights after seeing the
winner. **The committed thresholds are left exactly as they were and the failure
rate is published.**

**F-017** records the same lesson in miniature: MON-06 puts a 400% relative
threshold on a quantity whose development-sample baseline is 4.6e-6, and
quarterly windows range from 3.1e-7 to 1.2e-5 through sampling variation alone. A
relative threshold on a near-zero baseline has no stable meaning, because the
denominator is noise.

---

## 5. What this leaves

The honest position at the end of Phase 6 is that this model has **no working
leading indicator** of the failure mode that matters most to it. The only control
that caught the 2008 deterioration was calibration monitoring, and that is
structurally twelve months late — the 2007Q1 warn was not computable until
2008-03-31.

The candidate fix is already in the repository. The Phase 5 macro overlay fits
the quarterly default rate on unemployment and house prices, both observable in
near real time, so a divergence between the macro-implied rate and the model's
expected rate is computable the quarter it happens. F-009 records that the
overlay under-predicts the crisis by half, so it would not have called the
magnitude — but it would have called the direction in 2007 rather than 2009.

---

## 6. The model registry

A monitoring pack answers "is this model still working". An inventory answers a
prior question: **which models are there, and does the record describe what is
actually running.** The second question is the one that is easy to answer
falsely, because a list of models is just a list.

`governance/model_inventory.yaml` holds four models and nothing but judgements —
purpose, tier and why, owner, intended use, prohibited use, approval and its
conditions, and the findings each model carries. Everything derivable from an
artefact (feature list, training window, row counts, config fingerprints) is
deliberately **not** copied there, because a copied fact is a fact that can go
stale silently. It is read from the artefact at reconciliation time instead.

`uv run riskos registry` reads three independent sources — the inventory, the
artefacts on disk, and the findings register — and reports every disagreement.
The checks are the unflattering ones:

- a model recorded as in use with no loadable artefact;
- an artefact nobody inventoried, which is the classic shadow model;
- a bundle fitted against configuration that has since changed;
- a limitation citing a finding that does not exist;
- an open high-severity finding that names a model and is absent from its
  limitations;
- a tier-1 model with no monitoring rule pointed at it;
- an approval past its review date.

Config refuses several records outright rather than reporting them: a model in
use that names no artefact, an approval with no review date (which is a
permanent exemption, not an approval), and a conditional approval that states no
conditions.

### It found something immediately

The first reconciliation returned a high-severity discrepancy:

> the Phase 4 rubric selected `lightgbm` (RISKOS_PD_002) but no scoring-ready
> artefact exists for it, so every downstream figure is produced by a model the
> rubric did not select

The challenger run persisted the comparison metrics, the reliability curves, the
explanation comparison and a selection record naming the winner — and never
called `save_bundle`. The scorecard path did. So the selected model could not be
loaded, and the Phase 6 monitoring run had been executed against the runner-up
because that was the only model that could be scored.

Recorded as **F-018** and now remediated: the challenger persists a bundle, and
monitoring runs both models rather than swapping one timeline for the other. The
comparison in section 3.3 is what the extra work bought, and it is the evidence
that F-015 is about PSI rather than about the scorecard.

The two remaining discrepancies are genuine gaps, left open and reported:
RISKOS_HAZ_001 and RISKOS_LGD_001 are both tier 1 and in use with no monitoring
rule pointed at either.

### A hypothesis that was investigated and rejected

While adding the challenger bundle, the categorical encoding looked wrong. The
in-time validation split really is missing a level that training has — 65 Virgin
Islands rows, shifting the pandas category code of 5.6% of that split's rows by
one — and that is the same shape as R-003 and R-005, where a value depended on
the batch rather than on the model.

It is not that defect. LightGBM stores the fitted level **names** on the booster
and re-aligns an incoming frame by name before predicting, and SHAP's
TreeExplainer inherits the same path. Both were tested directly on a frame with
a level deliberately removed: identical predictions, identical SHAP values. Every
Phase 4 metric is bit-identical before and after the encoding was made explicit.

The mapping is carried in the bundle anyway, for a serving reason rather than a
correctness one — a scoring request arriving as JSON has no categorical dtype at
all, and a model should not depend on a dependency's internals to supply one. The
rejected hypothesis is recorded in F-018's remediation note, because a hypothesis
that was checked and did not hold is evidence too.

---

## 7. Serving: the inventory decides what runs

Serving is the point at which a model stops being an analysis and starts
producing numbers other systems act on, so it is the point at which the
governance record has to bind. The rule is enforced in code rather than in a
runbook: **the service serves whatever the inventory says is in use, and refuses
anything else.**

`riskos serve` does not take a bundle path. At startup it asks the inventory for
the PD model with status `in_use`, checks that the approval is current, loads the
artefact that entry names, and confirms the artefact's own manifest agrees about
which model it is. Each of these refuses, at startup and with the reason:

| Situation | Refused because |
| --- | --- |
| Model is `candidate`, or approval is `not_approved` | A bundle on disk is not a reason to serve. An approval is. |
| Approval past its review date | A lapsed approval is not an approval. |
| Two models in use | The service will not guess. |
| Inventory names an artefact whose manifest declares a different model id | The record and the artefact disagree about what this is. |

The challenger has a bundle and is refused, correctly, because the inventory
records it as a candidate that nobody has approved.

**Every response carries its provenance.** Alongside the PD, the points score,
and the principal drivers, a consumer receives the model id and version, the
inventory status, the approval status with its conditions and review date, the
risk tier, the findings the model is recorded as carrying, and whether the bundle
was fitted against configuration that has since changed. The number cannot be
used without seeing what produced it and what its stated limitations are.

Two smaller properties are asserted at the serving boundary rather than assumed:

- **Null is not absent.** The scorecard bins a missing value as its own category,
  so a `null` DTI is a legitimate input. A field absent from the request
  altogether is malformed and is refused by name. Pydantic quietly converts the
  second into the first by filling defaults, which the first smoke test caught
  when an absent field returned 200; the endpoints now dump only the fields the
  client actually sent.
- **A loan scored alone reports exactly the drivers it reports in a batch.** This
  is defect R-003 tested at the last place it could resurface. A borrower's
  stated reasons for a decision must not depend on who else was scored that
  second.

Scoring over HTTP is bit-identical to calling the scorecard directly on the same
rows.

---

## Artefacts

| File | What it holds |
| --- | --- |
| `reports/figures/monitoring_<model>_drift_timeline.csv` | PSI and CSI per window |
| `reports/figures/monitoring_<model>_csi_detail.csv` | Every feature's CSI, every window |
| `reports/figures/monitoring_<model>_csi_epsilon_dominated.csv` | Windows where CSI is floor-driven |
| `reports/figures/monitoring_<model>_performance_timeline.csv` | Discrimination and calibration, with maturity dates |
| `reports/figures/monitoring_<model>_performance_visible.csv` | The timeline as of a chosen reporting date |
| `reports/figures/monitoring_<model>_rule_false_positives.csv` | Per-rule in-sample firing rate |
| `reports/figures/monitoring_summary.json` | Run summary and the detection gap, per model |
| `governance/alert_register.csv` | Every rule firing, with model, owner, route and action |
| `governance/model_inventory.yaml` | The inventory itself |
| `governance/inventory_reconciliation.csv` | Every disagreement between record and reality |

`<model>` is `scorecard` or `challenger`.

## Findings raised

| ID | Severity | Status | Title |
| --- | --- | --- | --- |
| F-015 | high | open | Score PSI is blind to the 2008 regime change, for both model families |
| F-016 | medium | open | CSI against a pooled reference measures seasoning, not drift |
| F-017 | low | open | A relative threshold on a near-zero baseline is unusable |
| F-018 | high | remediated | The selected model had no scoring-ready artefact |
