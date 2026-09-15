# Model selection memo — 12-month PD

**Date:** 2026-08-30 · **Phase:** 4 · **Author:** project author
**Candidates:** WOE scorecard (champion candidate) vs LightGBM (challenger)
**Decision: select LightGBM — with a condition that materially qualifies it.**

> **Scope.** U.S. residential mortgage data (Freddie Mac SFLLD), applying IFRS 9
> concepts and OSFI Guideline E-23 as a methodological framework. Not a
> regulatory implementation and not a compliance claim.

---

## 1. Decision

The recorded rubric selects **LightGBM**, 0.4035 against 0.3648.

The margin is 0.0387 — outside the 0.02 tie-breaker band, so the tie-breaker
(prefer the more explainable model) does not engage. The decision follows the
arithmetic.

**The condition that goes with it is not a formality.** Both candidates score
approximately **zero** on calibration, the most heavily weighted dimension. The
rubric has selected the less-bad of two models that both fail the criterion that
matters most for expected credit loss. Neither is fit for provisioning under
stress in its current form. See §5.

---

## 2. The hypothesis that did not survive

The build plan anticipated (§7.6):

> Expect the uncalibrated GBM to look strong on AUC and weak on reliability
> under regime change. If that happens, it is the project's headline finding.

**It did not happen.** LightGBM is better on *both* dimensions in the crisis:

| OOT-stress (2007–09), uncalibrated | Gini | O/E ratio | Brier | Reliability |
| --- | --- | --- | --- | --- |
| LightGBM | **0.6648** | **2.80** | **0.018654** | **0.000434** |
| Scorecard | 0.6382 | 3.21 | 0.019725 | 0.000535 |

The expected trade-off is absent. The GBM ranks better *and* its predicted level
is less wrong. Recording this rather than forcing the anticipated narrative is
the point — a hypothesis stated in advance and then contradicted by the evidence
is a result, not a failure.

**Why the expectation was reasonable but wrong here.** The usual mechanism —
an unconstrained GBM exploiting interactions that do not survive a regime change
— is largely blocked in this setup. The challenger carries the same monotone
constraints as the scorecard on the five features where direction is
economically known, `min_child_samples` is 200, and the grid tops out at 63
leaves. The GBM was given very little room to fit crisis-fragile structure.

---

## 3. Rubric detail

Weights are recorded in `conf/models.yaml`, with calibration and stability
weighted above discrimination. The results below apply the recorded rules
and include a sensitivity analysis of explainability scoring.

| Dimension | Weight | LightGBM | Scorecard |
| --- | --- | --- | --- |
| Calibration | 0.35 | **0.0617** | **0.0000** |
| Stability | 0.25 | **0.4371** | 0.1486 |
| Discrimination | 0.20 | **0.6648** | 0.6382 |
| Explainability | 0.15 | 0.6000 | **1.0000** |
| Latency | 0.05 | 0.9941 | **0.9998** |
| **Weighted total** | | **0.4035** | 0.3648 |

**Stability drove the gap more than discrimination did.** The rubric takes the
worse of the two out-of-time PSI values. On OOT-stress both are essentially
zero (0.0004 and 0.0019). The separation comes from **OOT-benign**: LightGBM
0.1407 against the scorecard's 0.2128, both in the "moderate shift" band. The
scorecard's score distribution moves more between 1999–2006 and 2015–2019.

---

## 4. Where the scorecard is genuinely better

Stated plainly, because a memo that only lists the winner's merits is an
advocacy document.

**It does not overfit.** Train-to-validation AUC decay, same period, disjoint
loans:

| | Train | In-time validation | Decay |
| --- | --- | --- | --- |
| LightGBM | 0.9162 | 0.8937 | **0.0225** |
| Scorecard | 0.8936 | 0.8918 | **0.0018** |

The scorecard loses essentially nothing. LightGBM's in-sample advantage
(+0.0226 AUC) almost entirely evaporates out of sample (+0.0019). Its real
advantage appears only out-of-time, which is a more interesting result than the
in-sample number suggests.

**It is inspectable in full, ahead of time.** The entire scorecard is 15
features and their bins — it can be printed, tabled, and handed to an
adjudicator before any borrower is scored. A 15-leaf, depth-6 boosted ensemble
cannot be.

**Its explanation is on a scale humans use.** "This borrower lost 34 points for
LTV" against "this feature contributed −0.21 to the log-odds margin relative to
the population expectation."

**Its coefficients are directly reviewable for economic sense.** Every one is
signed and testable against prior expectation — the check that caught two
wrong-sign features in Phase 3. There is no equivalent single-number check on a
boosted ensemble.

---

## 5. The finding that qualifies the decision

**Both candidates fail the calibration dimension.**

| | O/E on OOT-stress | Calibration score |
| --- | --- | --- |
| LightGBM | 2.80 | 0.0617 |
| Scorecard | 3.21 | 0.0000 |

LightGBM predicts 0.78% where 2.17% occurred. The scorecard predicts 0.68%. On
an ECL basis both understate the provision roughly **threefold**. LightGBM is
better; neither is adequate.

**Calibration barely helped, and the reason is structural.**

| OOT-stress O/E | Uncalibrated | Platt | Isotonic |
| --- | --- | --- | --- |
| LightGBM | 2.80 | 2.75 | 2.76 |
| Scorecard | 3.21 | 3.11 | 3.03 |

Isotonic and Platt are both monotone, so neither reverses any ordering — Gini
moves by at most 0.0016 across all six cells. (Platt preserves AUC exactly,
being strictly monotone; isotonic's flat segments create ties, which perturbs it
very slightly. Not a defect, and worth knowing before someone reports it as
one.) They act on the level, not the ranking.
And they were fitted on **in-time validation (1999–2006)**, a period whose
default rate was 0.68%. A mapping estimated where the base rate is 0.68% cannot
know that the level will triple in a period it never saw.

**This is not a defect in the calibration step — it is the correct result.**
Recalibration fixes a level that is *systematically* wrong on data you have. It
cannot fix a level that is wrong because the economy changed. That is what the
Phase 5 macroeconomic overlay is for, and this table is the evidence that the
overlay is necessary rather than decorative.

Recorded as **finding F-006, severity high**.

---

## 6. A correction to the rubric's own explainability scoring

The recorded scoring rule awarded 1.0 for an "exact decomposition" and 0.6
for "approximate attribution", and I assigned the scorecard 1.0 and SHAP 0.6
a priori.

**Measurement contradicts that assignment.** Both mechanisms reconstruct their
model's output essentially exactly:

| | Max reconstruction error |
| --- | --- |
| Scorecard points → score | 1.14e-13 |
| SHAP values + base → log-odds margin | 1.15e-14 |

TreeSHAP is exact for tree ensembles, not approximate. The 0.6 was based on an
assumption the evidence does not support.

**Handling.** The rule as written was applied unchanged — rewriting a
recorded rubric after seeing results is precisely what committing it in
advance is meant to prevent. Instead the sensitivity is reported:

| Explainability treatment | LightGBM | Scorecard | Winner |
| --- | --- | --- | --- |
| As committed (1.0 / 0.6) | 0.4035 | 0.3648 | LightGBM by 0.0387 |
| Equalised at 1.0 | **0.4635** | 0.3648 | LightGBM by **0.0987** |

**The selection is unchanged and the margin widens**, so the decision is robust
to the error. The rubric's defect is recorded as **finding F-007**.

The genuine distinction the rubric *meant* to capture is not exactness. It is
that scorecard points are **absolute, on a human scale, and enumerable in
advance**, while SHAP values are **relative to a population baseline, in
log-odds, and computed per borrower**. Both are additive and both are exact.
That is a real difference and the rubric measured the wrong proxy for it.

---

## 7. Explanation comparison (build plan §7.7)

> Which explanation mechanism is suitable for communicating the principal drivers
> of a credit decision to a borrower or an adjudicator, and why?

Ten OOT-stress borrowers, both mechanisms:

- **Top-driver agreement: 10/10.** The two model families identify the *same*
  single most important driver for every borrower tested.
- **Top-3 overlap: 0.73.**

High agreement is the reassuring outcome. It means the choice between these
models is about the *mechanism* of explanation, not about substance — the two
would not tell a borrower materially different stories about why they were
declined. Low agreement would have been a serious finding in its own right.

**Answer to the question.** For communicating to a borrower or an adjudicator,
the scorecard's points remain preferable — not because SHAP is inexact, but
because points are absolute, denominated in a unit the reader already
understands, and the full table can be published in advance. SHAP requires the
reader to accept a counterfactual baseline and reason in log-odds.

That said, the honest conclusion is narrower than the project initially assumed:
**this is a communication-design advantage, not a mathematical one.**

---

## 8. Conditions of selection

LightGBM is selected **subject to**:

1. **Not approved for provisioning.** Crisis defaults are 2.80 times the
   challenger’s predictions (F-006). The Phase 5 macro extension also
   under-predicts crisis defaults (F-009); it does not resolve clearance.
2. **The scorecard is retained as the reference model**, not discarded. It is
   the interpretability benchmark and the fallback if the overlay cannot be
   made to work.
3. **Monitoring must track calibration drift, not just discrimination.** Both
   candidates' PSI on OOT-stress is ~0.0004–0.0019 — a score-distribution
   monitor would have raised **nothing** while the model was under-predicting
   defaults threefold. This is the most operationally important line in the
   memo.
4. **Delinquency-status dependence is quantified for both candidates** in the
   [ablation study](delinquency_ablation.md). F-004 remains open because other
   behavioural covariates remain and stress calibration is unresolved.
5. **Re-review on any material change** to the candidate pool, the monotone
   constraints, or the grid.

---

## 9. Reproducibility

| Item | Location |
| --- | --- |
| Rubric weights and scoring rules | `conf/models.yaml` |
| Full metric grid (2 models × 3 calibrations × 4 splits) | `reports/figures/champion_challenger_metrics.csv` |
| Rubric scores | `reports/figures/selection_rubric.csv` |
| LightGBM grid results | `reports/figures/lgbm_grid.csv` |
| Explanation comparison | `reports/figures/explanation_comparison.csv` |
| Selection record | `models/selection_record.json` |
| Runs | MLflow experiment `riskos-pd-12m` |

Selected hyperparameters: `max_depth=6, num_leaves=15` — the *smallest* leaf
count in the grid, at the deepest setting. No monotone constraint was silently
dropped (`constraints_ignored: []`).
