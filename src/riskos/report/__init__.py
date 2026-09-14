"""Phase 7 — the validation report and model card, generated from the artefacts.

Figures in the report are read from CSV, JSON, or YAML artefacts at generation
time, so the report cannot silently disagree with the pipeline. F-019 is why
that matters: a hand-edited register once described a fit that had never used
those observations. A report that reads artefacts surfaces that class of error.

A missing artefact is rendered as missing, naming the command that produces it,
rather than as a blank or a guess.
"""

from riskos.report.build import build_model_card, build_report, run
from riskos.report.sources import Sources, load_sources

__all__ = ["Sources", "build_model_card", "build_report", "load_sources", "run"]
