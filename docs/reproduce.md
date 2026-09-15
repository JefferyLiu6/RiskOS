# Setup and reproduction

[Demo](../README.md) · [Reference index](README.md)

Run commands from the repository root. Python 3.12 and `uv` are required.

## Inspect saved results (no source data)

```bash
make setup
make evaluate
make test
make lint
```

Setup may download dependencies. Evaluation reads committed aggregate CSVs; it
does not fit models. Data-dependent tests skip with a reason when their local
inputs are missing. You can also read the [comparison CSV](../reports/figures/champion_challenger_metrics.csv)
and [monitoring chart](../reports/figures/monitoring_blind_spot.png) without Python.

## Rebuild the core experiment

### Download the source data

1. Open Freddie Mac's official [Single-Family Loan-Level Dataset page](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset).
2. Follow **Access Historical Data**, register/sign in to [Clarity Data Intelligence](https://capitalmarkets.freddiemac.com/clarity), and accept the applicable data terms.
3. In **SFLLD Data Download**, select the **sample files** for vintages
   **1999–2012** and **2015–2019** (19 years). Download both origination and
   monthly performance files for each year.
4. Extract `sample_orig_YYYY.txt` and `sample_perf_YYYY.txt` into `data/raw/`.
   See [ingest](01-ingest.md) for file layout and validation.

The parser is configured for **Release 47 (July 2026)** in `conf/data.yaml`.
Check the download's layout against that configuration; changing only the version
label does not adapt the parser. Freddie Mac updates historical data, so a later
release may not reproduce the saved metrics exactly. Record the release used.

Downloads require registration; there is no anonymous ZIP bundled here. Loan-level
data is licensed and is not committed. `data/` can be a local directory or a
symlink to external storage.

FRED series require a key in `.env` (see `.env.example`). `make ingest` first runs
`make macro`, which fetches those series or uses the local cache.

```bash
make ingest
make panel
make train
make challenger
make evaluate
make monitor
```

The panel and fitting stages require the source data and local storage; this is
not the five-minute saved-results walkthrough. Phase notes describe their inputs,
outputs, and original run details.

## Optional extensions

After the core pipeline:

```bash
make hazard
make ecl
make registry
make report
```

`make registry` exits non-zero for high-severity discrepancies between inventory
and artifacts. Inspect its output before continuing. `make report` generates the
validation report and model card from available artifacts, names missing inputs,
and attempts a PDF render when Quarto is installed.

For the optional scoring API, run `uv run riskos serve` (localhost, port 8000).
It needs a fitted model and the inventory's designated model to pass loading
checks. Inventory clearance is illustrative developer clearance only.

`make all` runs the full pipeline, including extensions. Individual targets are
listed in the [Makefile](../Makefile).

## Public integration checks and retrospective ablation

```bash
uv run pytest tests/test_ecl_integration.py -v  # synthetic inputs; no licensed data
make ablate                                  # requires the existing panel and baseline bundles
make report                                  # refresh reports and the ablation chart
```

`make ablate` writes a separate CSV and run manifest. It refits without current
delinquency and reuses the existing evaluation splits. The manifest contains input
and source hashes, row counts, fitted features, and the LightGBM grid. It preserves
the main models. [Validation coverage](validation.md) describes what each check establishes.
