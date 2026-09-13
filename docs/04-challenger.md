# Phase 4 — Challenger, calibration, and selection

**Status: complete.** LightGBM selected, 0.4035 vs 0.3648 · **189 tests**

> Two findings here matter more than the winner. First, the build plan's stated
> hypothesis was **contradicted by the evidence** — and reporting that is the
> whole value of stating it in advance. Second, **both candidates fail** the
> dimension that matters most, so "LightGBM wins" is a much weaker sentence than
> it looks.

**Run it:** `uv run riskos challenger` (~4 minutes)

---

## 1. The governance move: commit the rubric first

`conf/models.yaml` fixes the weights, the scoring transforms, and the
tie-breaker **before the challenger is fitted**. That ordering is the entire
point. Weights chosen after seeing which model wins are not a rubric; they are a
justification written backwards.

| Dimension | Weight | Why |
| --- | --- | --- |
| **Calibration** | **0.35** | ECL is a money number. Wrong level = wrong provision |
| **Stability** | 0.25 | IFRS 9 needs an estimate every period, not once |
| Discrimination | 0.20 | Necessary, not sufficient — and the metric most likely to flatter |
| Explainability | 0.15 | E-23 expects explicable behaviour |
| Latency | 0.05 | Neither candidate is remotely slow |

The weights come from build plan §1's ordering, fixed before any modelling.
Config **rejects** a weighted dimension with no written rationale or no scoring
rule.

**Honest caveat, stated rather than glossed:** by Phase 4 I had already seen the
scorecard's own out-of-time results from Phase 3. What I had not seen was any
challenger number or any head-to-head. That's a weaker claim than a true blind
commitment, and it's the reason the ordering is worth making verifiable rather
than merely asserted.

### Fairness of the comparison

Both candidates get the **same 27-feature candidate pool**. Each then applies
its own family-appropriate selection: the scorecard does IV screening,
collinearity pruning, and wrong-sign elimination; LightGBM handles weak and
correlated inputs natively.

Handing the challenger only the scorecard's surviving 15 would have rigged the
comparison. Both also carry the **same monotone constraints** on the five
features where direction is economically known — so the comparison isolates
model form, not economic assumptions.

---

## 2. The hypothesis that didn't survive

The build plan anticipated:

> Expect the uncalibrated GBM to look strong on AUC and weak on reliability
> under regime change. **If that happens, it is the project's headline finding.**

**It didn't happen.**

| OOT-stress (2007–09) | Gini | O/E | Brier | Reliability |
| --- | --- | --- | --- | --- |
| **LightGBM** | **0.6648** | **2.80** | **0.018654** | **0.000434** |
| Scorecard | 0.6382 | 3.21 | 0.019725 | 0.000535 |

LightGBM is better on **both**. The expected trade-off is simply absent.

**Why the expectation was reasonable but wrong here.** The usual mechanism — an
unconstrained GBM exploiting interactions that don't survive a regime change —
was largely designed out. The challenger carries the same monotone constraints,
`min_child_samples=200`, and a grid topping out at 63 leaves. It had very little
room to fit crisis-fragile structure. The selected config is `max_depth=6,
num_leaves=15` — the *smallest* leaf count in the grid.

**This is the more valuable outcome.** A prediction stated in advance and then
falsified by the evidence is a result. Quietly reshaping the narrative to match
the anticipated finding would have been the actual failure.

---

## 3. The finding that qualifies the win (F-006)

Look at the rubric scores, not just the total:

| Dimension | Weight | LightGBM | Scorecard |
| --- | --- | --- | --- |
| Calibration | 0.35 | **0.0617** | **0.0000** |
| Stability | 0.25 | 0.4371 | 0.1486 |
| Discrimination | 0.20 | 0.6648 | 0.6382 |
| Explainability | 0.15 | 0.6000 | 1.0000 |
| Latency | 0.05 | 0.9941 | 0.9998 |
| **Total** | | **0.4035** | 0.3648 |

**Both score ≈ 0 on the highest-weighted dimension.** The rubric selected the
less-bad of two models that both fail the criterion that matters most. LightGBM
predicts 0.78% where 2.17% occurred; the scorecard predicts 0.68%. Both
understate the provision roughly threefold.

### Calibration barely helped — and that's the correct result

| OOT-stress O/E | Uncalibrated | Platt | Isotonic |
| --- | --- | --- | --- |
| LightGBM | 2.80 | 2.75 | 2.76 |
| Scorecard | 3.21 | 3.11 | 3.03 |

Isotonic and Platt were fitted on in-time validation — **1999–2006, where the
default rate was 0.68%.** A mapping estimated at that base rate cannot know the
level will triple in a period it never saw.

**Recalibration corrects a level that is systematically wrong on data you have.
It cannot correct a level that is wrong because the economy changed.** That is
precisely what the Phase 5 macro overlay is for, and this table is the evidence
it is necessary rather than decorative.

### The most operationally important line in the phase

```
OOT-stress, uncalibrated:
  LightGBM   score PSI 0.0004  ("stable")   O/E 2.80
  Scorecard  score PSI 0.0019  ("stable")   O/E 3.21
```

**A monitoring programme watching only score distribution would have raised
nothing** — while the model under-predicted defaults threefold. The score
distribution barely moved; the *relationship between score and outcome* broke
completely.

That has a direct consequence for Phase 6: PSI alone is not a sufficient
monitoring control. Calibration drift and observed-vs-predicted default rate
must be tracked independently. There's a test asserting this exact
juxtaposition.

---

## 4. Where the scorecard is genuinely better

A memo listing only the winner's merits is advocacy. Four real advantages:

**It doesn't overfit.**

| | Train | In-time validation | Decay |
| --- | --- | --- | --- |
| LightGBM | 0.9162 | 0.8937 | **0.0225** |
| Scorecard | 0.8936 | 0.8918 | **0.0018** |

LightGBM's in-sample advantage (+0.0226 AUC) almost entirely **evaporates** out
of sample (+0.0019). Its real edge appears only out-of-time — a more interesting
result than the in-sample number suggests, and one you'd miss without an in-time
validation split.

**It's fully inspectable ahead of time.** 15 features and their bins can be
printed and handed to an adjudicator before any borrower is scored. A boosted
ensemble cannot.

**Its explanation is on a human scale.** "Lost 34 points for LTV" vs
"contributed −0.21 to the log-odds margin relative to population expectation."

**Its coefficients are reviewable for economic sense** — the check that caught
two wrong-sign features in Phase 3. There's no equivalent single-number check on
an ensemble.

---

## 5. I got the explainability scoring wrong (F-007)

The pre-committed rule awarded 1.0 for "exact decomposition" and 0.6 for
"approximate attribution", and I assigned SHAP 0.6 **a priori**.

Then I measured it:

| | Max reconstruction error |
| --- | --- |
| Scorecard points → score | 1.14e-13 |
| SHAP + base value → log-odds margin | **1.15e-14** |

**TreeSHAP is exact for tree ensembles.** Not approximate. My 0.6 was based on
an assumption the evidence contradicts.

### What I did about it

**Applied the rule as written, unchanged.** Rewriting a pre-committed rubric
after seeing results is exactly what committing it in advance is meant to
prevent. Instead, reported the sensitivity:

| Explainability treatment | LightGBM | Scorecard | Margin |
| --- | --- | --- | --- |
| As committed (1.0 / 0.6) | 0.4035 | 0.3648 | 0.0387 |
| Equalised at 1.0 | **0.4635** | 0.3648 | **0.0987** |

Selection unchanged, margin **widens**. The decision is robust to my error.

The distinction the rubric *meant* to capture is real, just mis-measured:
scorecard points are **absolute, human-scaled, enumerable in advance**; SHAP
values are **relative to a population baseline, in log-odds, computed per
borrower**. Both are additive and exact. The rubric measured the wrong proxy.

> This is a better governance story than a clean rubric would have been. A
> pre-committed rule can be committed *and still be wrong*. The discipline is to
> apply it as written and report the sensitivity — not to quietly amend it.

---

## 6. Explanation comparison (§7.7)

> Which explanation mechanism is suitable for communicating the principal drivers
> of a credit decision to a borrower or an adjudicator, and why?

Ten OOT-stress borrowers, both mechanisms:

- **Top-driver agreement: 10/10**
- **Top-3 overlap: 0.73**

Perfect agreement on the single most important driver is the reassuring outcome:
the choice between these models is about the *mechanism* of explanation, not the
substance. The two would not tell a borrower materially different stories about
why they were declined. **Low agreement would have been a serious finding** —
two models of similar accuracy giving contradictory reasons.

**Answer:** the scorecard's points remain preferable for communication — not
because SHAP is inexact, but because points are absolute, in a unit the reader
already understands, and the full table can be published in advance. SHAP asks
the reader to accept a counterfactual baseline and reason in log-odds.

The honest conclusion is narrower than the project assumed: **a
communication-design advantage, not a mathematical one.**

---

## 7. Likely interview questions

**"How did you choose between the models?"**
> A weighted rubric committed to config before the challenger was fitted —
> calibration 0.35, stability 0.25, discrimination 0.20, explainability 0.15,
> latency 0.05, taken from the project's stated priorities. LightGBM won
> 0.40 to 0.36. But the headline isn't that it won: both candidates scored
> essentially zero on calibration, so the rubric picked the less-bad of two
> models that both fail the thing that matters most for provisioning.

**"Did the GBM overfit?"**
> In-sample, yes — train AUC 0.916 against the scorecard's 0.894. But that
> advantage almost entirely disappears in in-time validation: 0.8937 vs 0.8918,
> a gap of 0.002. Its genuine edge only shows up out-of-time. That's exactly why
> the in-time validation split exists — without it I'd have concluded the GBM
> was 2 points better, which it isn't.

**"Why didn't calibration fix the under-prediction?"**
> Because isotonic and Platt were fitted on 1999–2006, where the default rate
> was 0.68%. They're monotone maps of the score — they can correct a level
> that's systematically wrong on data you have, but they can't anticipate a
> regime they've never seen. O/E only moved from 3.21 to 3.03. That's not a
> defect in the calibration step, it's the correct result, and it's the
> justification for building a macro overlay in Phase 5.

**"Your rubric was wrong. What did you do?"**
> I'd scored SHAP 0.6 for being an approximate attribution. Then I measured the
> reconstruction error and TreeSHAP came back exact to 1e-14. I applied the rule
> as written anyway — amending a pre-committed rubric after seeing results
> defeats the purpose — and reported the sensitivity instead. Equalising
> explainability, LightGBM wins by more, so the decision was robust to my error.
> It's logged as a finding.

**"What would your monitoring have caught?"**
> Not enough, and that's the point. Score PSI on the crisis split was 0.0004 —
> firmly "stable" — while the model was under-predicting defaults threefold. The
> score distribution barely moved; the relationship between score and outcome
> broke. Any monitoring programme that only watches PSI would have been silent
> through 2008.

---

## 8. Conditions on the selection

1. **Not approved for provisioning without the Phase 5 macro overlay** (F-006)
2. **The scorecard is retained as the reference model**, not discarded — the
   interpretability benchmark and the fallback
3. **Monitoring must track calibration drift, not just PSI**
4. **F-004 (delinquency dominance) must be quantified** for both candidates
5. **Re-review on any change** to the candidate pool, constraints, or grid

Full memo: [`reports/model_selection_memo.md`](../reports/model_selection_memo.md)

---

**Next:** Phase 5 — discrete-time hazard with competing risks, empirical LGD
with tail investigation, the ECL engine, and the macro overlay that F-006 says
is mandatory rather than optional.
