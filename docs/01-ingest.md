# Phase 1 — Ingest and validate

> Detailed reference. Start with the [demo](../README.md); use the
> [reference index](README.md) to find a specific topic.

**Implemented:** 19 vintages, 950,000 loans, 57,422,329 loan-months.

> Phases 1 and 2 are roughly 40% of the work. That is not a warning about
> tedium — it is where the errors that silently invalidate everything
> downstream get made. A modelling mistake shows up as bad metrics. A data
> mistake shows up as *good* metrics that mean nothing.

---

## 1. What this phase had to achieve

Turn 6.0 GB of raw pipe-delimited text into typed, validated, queryable Parquet
— without silently altering a single value.

| Criterion | Result |
| --- | --- |
| All 19 vintages parse without error | ✅ 1999–2012, 2015–2019 |
| Row counts logged | ✅ per vintage, in a manifest |
| Schema validation passes | ✅ Pandera, strict and ordered |
| Zero-balance mapping complete, no unmapped values | ✅ all 7 codes |
| Nothing under `data/` tracked by git | ✅ verified with a probe commit |

**Run it:** `make ingest` · **Check state:** `uv run riskos status`

---

## 2. The domain you need to understand first

### Two files per vintage, and why

Freddie Mac ships each vintage year as a pair:

| File | Grain | Rows (1999) | Contents |
| --- | --- | --- | --- |
| `sample_orig_YYYY.txt` | **one row per loan** | 50,000 | What was true at origination: FICO, LTV, DTI, term, purpose, state |
| `sample_perf_YYYY.txt` | **one row per loan-month** | 2,503,335 | What happened since: balance, delinquency status, and eventually how it ended |

A "vintage" is the cohort of loans originated in that year. The performance file
tracks each of them monthly until it terminates — so a 1999 loan can have 20+
years of rows.

This shape is why the project uses DuckDB over Parquet rather than pandas: 57
million loan-months is comfortable for columnar scans on a laptop and painful
for anything row-oriented.

### Zero balance codes — the most important field in the dataset

When a loan leaves the book, one code says why. **This field defines your target
variable**, so getting it wrong invalidates the entire project:

| Code | Guide label | Is it a default? |
| --- | --- | --- |
| `01` | Prepaid or Matured (Voluntary Payoff) | **No** — borrower paid in full |
| `02` | Third Party Sale | **Yes** — credit event |
| `03` | Short Sale or Charge Off | **Yes** — credit event |
| `09` | REO Disposition | **Yes** — bank took and sold the property |
| `15` | Whole Loan sales | Judgement — see below |
| `16` | Reperforming loan securitizations | **No** — a funding event |
| `96` | Confirmed Underwriting/Servicing Defect | **Excluded entirely** |

Three things here need to be stated precisely:

**Prepayment is not default, and it is not "good" either.** A borrower who
refinances away is *censored* — you stop observing them. Treating prepayment as
a non-event biases your model, which is why Phase 5 models it as a **competing
risk** rather than ignoring it.

**Code 96 is excluded, not counted as good.** These are loans repurchased
because of a defect in how they were underwritten or serviced. They didn't
default and they didn't survive — the outcome is simply unobservable. Counting
them as non-defaults would understate risk.

**Code 15 is a judgement, and it is flagged as one.** The build plan lists "note
sale" as a credit event; Release 47 of the Guide labels code 15 "Whole Loan
sales". Those are the same underlying event under a renamed label, so it's
mapped as a credit event — but `conf/panel.yaml` records that this is an
inference rather than a transcription, and Phase 2 must quantify its impact.
Expected materiality is low: code 15 first appears in 2014, totals ~1,200
terminations across all vintages, and those loans are predominantly already
seriously delinquent, so the 90+ DPD rule catches them anyway.

### Sentinel values — the classic silent corruption

The Guide encodes "not available" as **in-range-looking numbers**:

| Field | Sentinel | What happens if you miss it |
| --- | --- | --- |
| Credit score | `9999` | Mean FICO becomes ~2,400. Every model breaks visibly |
| LTV / CLTV / DTI | `999` | A 999% LTV silently becomes your riskiest segment |
| Number of borrowers | `99` | 99 co-borrowers on a mortgage |
| Delinquency status | `XX` | **Reads as "current" if you coerce to numeric** |

The credit-score one is obvious enough that you'd catch it. **`999` for LTV is
the dangerous one** — it's numerically plausible, it survives a sanity check,
and it lands in your worst risk bucket where it does maximum damage.

Every sentinel is declared per-column in `conf/data.yaml`, sourced from the
Guide, and nulled at parse time. None was inferred from the data.

---

## 3. Design decisions, and the reasoning

### The parser knows no column names

Nothing in `src/riskos/ingest/` contains a Freddie Mac field name or position.
The layout is data, living in `conf/data.yaml`:

```yaml
- {name: credit_score, dtype: int, null_values: ["9999"],
   description: "Classic FICO, 300-850; 9999 = Not Available"}
```

Parsing, casting, and the Pandera schema are all generated from that one list,
so they cannot drift apart. The layout is version-stamped to
*"General User Guide, Release 47, July 2026"*, and ingest refuses to run without
that stamp.

**Why this matters:** the layout changes between releases. A
hardcoded parser silently mis-assigns columns when the file gains a field — and
mis-assigned columns produce a model that trains fine and is completely wrong.
Every field was verified position-by-position against a real record before being
committed.

### Casting is loud, with no tolerance threshold

Fields are read as strings, then documented sentinels are nulled, then each
column is cast to its declared type. **Any value that survives and still fails
to cast aborts the entire run**, naming the column, the count, and examples.

There is deliberately no "allow up to 0.1% bad rows" setting. Two reasons:

1. A tolerance is an unsourced magic number, which the whole project exists to
   avoid.
2. A cast failure means either the layout is wrong or there is an undocumented
   sentinel. Both need a human. Nulling silently converts a **structural** error
   into missing data that looks ordinary.

### Never silently clip, impute, or drop

This runs through everything. A gap in a FRED macro series is converted to null
and **counted**, not forward-filled. It's information about the series.

The same principle returns with much higher stakes in Phase 5, where LGD values
below 0 and above 1 are economically real (MI recoveries can exceed the balance;
accrued interest plus foreclosure expenses can too). `np.clip(lgd, 0, 1)` with
no analysis is treated as a project-level failure.

---

## 4. The bug worth recording

The end-to-end pipeline test passed. Then the first run on real data failed
Pandera validation with:

```
column 'credit_score' not unique — 49,982 distinct values in 50,000 rows
```

**The cause.** `run.py` asserted uniqueness on the loan key by taking
`columns[0]` — the first column. In the test fixture the loan ID *was* first. In
the real origination file, field 1 is `credit_score`; `loan_sequence_number` is
field **20**. In the performance file it *is* field 1. So any positional
assumption is wrong on exactly one of the two files.

It was asserting that credit scores are unique across borrowers.

**The fix.** Name the key in config (`layout.key_column`), validate at load time
that it exists in both layouts, and — the part that matters — **rewrite the test
fixture so the key sits in the last position**, mirroring reality. A fixture
with the key first would have hidden this class of bug forever.

**Why it matters.** It's a concrete case of a test passing while being
structurally unrepresentative of production data. The lesson isn't "write more
tests", it's *"make fixtures adversarial to your assumptions."* And note the
failure mode: had the key been a genuinely unique column, this would never have
errored — it would just have validated the wrong thing silently.

---

## 5. Does the data look real? The sanity check

Before trusting anything, check the credit content shows the crisis:

| Year | Prepaid | **Credit events** |
| --- | --- | --- |
| 2005 | 21,946 | 295 |
| 2006 | 16,815 | 251 |
| 2007 | 18,430 | 259 |
| 2008 | 22,198 | **490** |
| 2009 | 49,180 | **1,215** |
| 2010 | 47,189 | **2,258** |
| 2011 | 40,041 | **2,858** |
| 2012 | 59,741 | **3,065** |
| 2015 | 25,345 | 1,138 |

Two things to be able to explain:

**Why the peak is 2010–2012, not 2008.** The crash is 2007–08, but these are
*dispositions* — the date the property was finally sold. Foreclosure takes
1–3 years, longer in judicial-foreclosure states. **The credit event lags the
credit deterioration.** This is exactly why Phase 2 labels defaults from
delinquency status at the observation date rather than from termination date,
and why a naive analysis on termination dates would mis-date the crisis by
three years.

**The prepayment spikes are real too.** 2003 and 2020–21 are refinancing booms
driven by low rates. If prepayment didn't spike there, something would be wrong.

---

## 6. Questions a reviewer would ask

**"Walk me through your data pipeline."**
> 6 GB of pipe-delimited text, two files per vintage year — one row per loan at
> origination, one row per loan-month afterwards. Parsed to Parquet partitioned
> by vintage, queried with DuckDB. The parser is layout-driven: column
> definitions live in version-stamped config transcribed from the User Guide, so
> parsing and schema validation can't drift apart. 950,000 loans, 57 million
> loan-months, ingests in 25 seconds.

**"How do you know your data is right?"**
> Three layers. Structural: declared field count must match the file, or it
> aborts. Type: every value must cast to its declared type — no tolerance
> threshold, because a cast failure means the layout is wrong or there's an
> undocumented sentinel, and both need a human. Semantic: the zero-balance
> mapping must be exhaustive, and credit events by year have to show the
> crisis with the right 2–3 year disposition lag. That last one is the check
> that would catch a mis-assigned column.

**"What's a sentinel value and why do you care?"**
> The Guide encodes "not available" as in-range-looking numbers — 9999 for
> credit score, 999 for LTV. The credit score one is obvious enough you'd catch
> it. The 999 LTV is the dangerous one: numerically plausible, survives a sanity
> check, and lands in your riskiest bucket where it does maximum damage. All of
> them are declared per-column from the Guide and nulled at parse.

**"Why not just use pandas?"**
> 57 million loan-months. Columnar Parquet plus DuckDB gives real windowed SQL
> over that on a laptop with no server. Polars for the panel reshaping;
> pandas only at the scikit-learn boundary.

**"You didn't have the data at first. What did you do?"**
> Built the layout-driven parser and proved the whole path end-to-end against a
> toy fixture — discovery, parse, cast, schema, code mapping, Parquet, manifest
> — including the failure modes. So when the files landed, the only untested
> variable was transcribing the layout. That's also how the positional-key bug
> got caught within a minute of first contact with real data.

---

## 7. What could still bite

Recorded honestly rather than discovered later:

- **Code 15 is a judgement**, not a transcription. Phase 2 must quantify it.
- **`estimated_ltv` is only populated from April 2017.** A tempting feature that
  doesn't exist for the crisis period — using it would leak availability
  information.
- **Delinquency status is a string**, not a number: `00`–`99` months, plus `RA`
  (REO acquisition) and `XX` (not available). 90+ DPD means numeric ≥ 3 **or**
  `RA`, and `XX` must not be read as current.
- **`sample_perf` vs `sample_svcg`.** The naming guess was wrong; discovery tries
  multiple conventions and reports unclassified files rather than skipping them.

---

**Next:** Phase 2 — building the observation panel and the labels. The
acceptance criterion is a default-rate-by-quarter chart showing the 2007–09
spike. Nothing downstream means anything until that plot looks right.
