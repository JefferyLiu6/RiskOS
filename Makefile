.PHONY: ablate setup macro ingest panel features train challenger evaluate hazard ecl monitor registry serve report test lint all

PY := uv run

setup:
	uv sync --frozen

macro:             ## Phase 1 — pull and cache FRED series (offline after first run)
	$(PY) riskos macro

ingest: macro      ## Phase 1 — parse raw SFLLD files to validated Parquet
	$(PY) riskos ingest

panel:             ## Phase 2 — observation panel + monthly risk set + labels
	$(PY) riskos panel

features:          ## Phase 3 — binning, WOE, IV screening
	$(PY) riskos features

train:             ## Phase 3 — WOE scorecard champion candidate
	$(PY) riskos train

challenger:        ## Phase 4 — LightGBM challenger, calibration, selection rubric
	$(PY) riskos challenger

ablate:            ## Retrospective delinquency sensitivity on the existing panel
	$(PY) riskos ablate

evaluate:          ## Phase 3/4 — metrics on all four splits
	$(PY) riskos evaluate

hazard:            ## Phase 5 — discrete-time hazard, competing risks
	$(PY) riskos hazard

ecl:               ## Phase 5 — LGD, EAD, staging, ECL, scenarios, backtest
	$(PY) riskos ecl

monitor:           ## Phase 6 — PSI/CSI, calibration decay, alert records
	$(PY) riskos monitor

registry:          ## Phase 6 — reconcile the model inventory against the artefacts
	$(PY) riskos registry

serve:             ## Phase 6 — serve the inventoried, approved PD model (refuses anything else)
	$(PY) riskos serve --host 0.0.0.0 --port 8000

report:            ## Phase 7 — validation report and model card, generated from the artefacts
	$(PY) riskos report
	@if command -v quarto >/dev/null 2>&1; then \
		quarto render reports/validation_report.md --to pdf; \
	else \
		echo "quarto not installed; Markdown written to reports/ (install quarto to render PDF)"; \
	fi

test:
	$(PY) pytest

lint:
	$(PY) ruff check src tests
	$(PY) ruff format --check src tests
	$(PY) mypy

all: ingest panel features train challenger evaluate hazard ecl monitor registry report
