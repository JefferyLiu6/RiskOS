# Business context — why this system exists

*Read this first. Everything else in the repo is a technical answer to a problem
stated here.*

Building the models is the smaller part of this problem. The larger part is
understanding **what the number is for, who is accountable for it, and what
happens when it is wrong.** This document is that background; the technical
chapters that follow are answers to it.

---

## 1. The commercial problem

A lender's core economics are simple to state: it earns interest on money it
lends, and loses money when borrowers do not repay. The hard part is that the
interest arrives **now** and the losses arrive **later**.

Accounting has to bridge that gap. If a bank booked interest income in full
today and recognised losses only when a borrower actually stopped paying, its
profits would look excellent right up until the moment they didn't. That is
approximately what happened in 2008: the "incurred loss" model then in force
only let banks provision once a loss event had already occurred, so provisions
lagged reality badly and arrived exactly when capital was scarcest.

The regulatory response was **IFRS 9** (and its U.S. cousin, CECL). It requires
banks to recognise **expected** credit losses forward-looking, from the day a
loan is written. That single change is what created the entire modelling
discipline this project sits in.

### What this costs

Provisioning is not an accounting footnote — it is one of the largest and most
volatile lines in a bank's income statement.

| Bank | Illustrative scale |
| --- | --- |
| A Canadian Big Five bank | residential mortgage book on the order of $300–400B |
| Annual provision for credit losses | typically hundreds of millions to several billion, swinging hard with the cycle |

The arithmetic that makes this matter: on a **$300B** mortgage book, an error of
just **10 basis points** in the expected loss rate is **$300 million** of
misstated provision. That flows straight to pre-tax profit. It moves reported
earnings per share. It is disclosed publicly, reviewed by external auditors, and
scrutinised by the regulator.

**This is why credit risk models are governed the way they are.** It is not
bureaucratic caution. A model error here is a material misstatement of a public
company's financial statements.

---

## 2. What the system actually has to produce

The deliverable is one number, decomposed many ways:

```
ECL  =  Σ over loans of   PD × LGD × EAD × discount factor
```

| Term | Meaning | Where it comes from |
| --- | --- | --- |
| **PD** | Probability of default | A model — the part you'd build |
| **LGD** | Loss given default (what fraction is lost once default happens) | Estimated from actual recoveries |
| **EAD** | Exposure at default (balance owed at that moment) | Largely mechanical for mortgages |

IFRS 9 then adds **staging**, which decides *which* PD to use:

| Stage | Condition | Loss measured over |
| --- | --- | --- |
| 1 | Performing, no significant deterioration | next **12 months** |
| 2 | Credit risk has increased significantly since origination | **remaining lifetime** |
| 3 | Credit-impaired (in default) | **remaining lifetime** |

The Stage 1 → Stage 2 boundary is commercially enormous. Moving a loan from
Stage 1 to Stage 2 switches it from a 12-month loss estimate to a lifetime one,
which for a 25-year mortgage can multiply its provision several times over. In
March 2020, banks moved large populations into Stage 2 at once, and provisions
spiked accordingly.

**That threshold is a judgement call set by humans.** Which is precisely why it
must be documented, justified, and sensitivity-tested — and why this repo
insists every such number lives in `conf/assumptions.yaml` with a stated source.

And because the estimate is forward-looking, it has to be conditioned on the
**economy** — unemployment, house prices — under multiple weighted scenarios.
A single point estimate isn't acceptable; you owe a probability-weighted answer.

---

## 3. Who is accountable — the three lines of defence

This is the organisational question, and it is the one a pure ML background
leaves least prepared for.

| Line | Who | Role |
| --- | --- | --- |
| **First** | Model development, the business | Builds the model, owns the risk |
| **Second** | Model Risk Management / Model Validation | **Independently** challenges and approves it |
| **Third** | Internal Audit | Checks that lines one and two did their jobs |

A model does not go into production because it has good AUC. It goes into
production because **second line reviewed it and signed an approval**, usually
with conditions and a re-review date. Second line can and does reject models.

In Canada this is governed by **OSFI Guideline E-23, Model Risk Management** —
published 11 September 2025, effective **1 May 2027**. The 2027 version is
notably broader than the 2017 one: it covers *all* models carrying risk at
federally regulated financial institutions, not just capital models, with
explicit attention to AI/ML. It sets expectations across the model lifecycle:
rationale, data, development, **independent review**, approval, deployment,
ongoing monitoring, and decommissioning — plus a **model inventory** and a
**risk-rating** of each model.

> **Honest limitation, stated up front.** This project is built by one person, so
> it cannot claim genuine organisational independence. The review artifact is
> called a *developer validation with simulated second-line review*, and that
> limitation is recorded in the report rather than glossed.

---

## 4. What this project is built to demonstrate

Four claims, each of which the later chapters have to earn with evidence:

1. **A model with better AUC can be the worse risk model.** Discrimination
   (ranking borrowers correctly) and calibration (getting the probability level
   right) are different things. ECL is a **money** number, so a model that ranks
   perfectly but predicts 2% when the truth is 4% halves the provision.
2. **Behaviour under regime change is the test that matters.** Train on
   1999–2006, then watch the model meet 2008. Any model looks fine in-sample;
   the question is what it does when the world changes.
3. **A validation with no findings is not credible.** This one carries a
   findings register, populated at discovery.
4. **Approval with conditions, not a pretence of no limitations.** Every
   approval names its conditions and its review date.

### Scope statement

> The underlying portfolio is U.S. residential mortgage data, because comparable
> public Canadian loan-level default and loss data is not available. The project
> applies IFRS 9 concepts and OSFI Guideline E-23 as a methodological and
> governance framework relevant to Canadian financial institutions. It does not
> represent a regulatory implementation, does not reproduce any institution's ECL
> system, and makes no claim of OSFI compliance.

---

## 5. Where engineering discipline adds value here

Risk teams know the mathematics. Where a modelling function is most often weak
is in making that mathematics **reproducible, testable, and governable**, and
that is the gap this repository is built around:

- Config-driven pipelines with **no magic numbers in code** (`conf/`)
- **Schema-as-code** validation that fails loudly rather than silently coercing
- Assumption registers **enforced by tests** — change a constant without
  updating its register entry and the build fails
- Leakage prevention as an **executable test**, not a code-review convention
- Governance artifacts as **structured data** (`governance/*.yaml`), reconciled
  against what is actually on disk, not prose in a document

---

## 6. Vocabulary

| Term | One-line meaning |
| --- | --- |
| **PD / LGD / EAD** | Probability of default / loss fraction given default / balance at default |
| **ECL** | Expected credit loss = PD × LGD × EAD, discounted |
| **IFRS 9 staging** | 1 = performing, 2 = significantly deteriorated, 3 = credit-impaired |
| **SICR** | Significant Increase in Credit Risk — the Stage 1 → 2 trigger |
| **Through-the-cycle vs point-in-time** | Long-run average PD vs PD conditioned on today's economy. IFRS 9 needs point-in-time |
| **Scorecard** | Logistic regression on binned features, expressed as points. The industry default because it is explainable |
| **WOE / IV** | Weight of Evidence (log-odds transform of a bin) / Information Value (a feature's predictive strength) |
| **KS / Gini** | Discrimination measures — can the model separate good from bad |
| **PSI / CSI** | Population / Characteristic Stability Index — has the scored population drifted from training |
| **Calibration** | Do predicted probabilities match observed frequencies |
| **Champion / challenger** | Incumbent model vs candidate replacement, compared on a pre-committed rubric |
| **OOT** | Out-of-time validation — test on a *later* period, never a random split |
| **Vintage** | The cohort of loans originated in a given period |
| **Basis point (bp)** | 0.01%. 10bp on $300B is $300M |

---

## 7. The one-paragraph version

> Banks must estimate expected credit losses forward-looking under IFRS 9, and
> those estimates move reported earnings by hundreds of millions of dollars.
> The estimate is PD × LGD × EAD, conditioned on macroeconomic scenarios, with
> IFRS 9 staging deciding whether the horizon is twelve months or lifetime.
> Because the number is material and model-derived, OSFI Guideline E-23 requires
> it to be developed, independently reviewed, approved with conditions,
> monitored, and inventoried. RiskOS builds that end to end on public U.S.
> mortgage data — and the point of the exercise is not the AUC, it is the
> calibration, the stability under regime change, and the governance around it.

---

**Next:** [`01-ingest.md`](01-ingest.md) — Phase 1, getting the data in without
corrupting it.
