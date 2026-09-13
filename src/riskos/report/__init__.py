"""Phase 7 — the validation report and model card, generated from the artefacts.

Nothing in the report is typed by hand. Every figure is read from the CSV, JSON
or YAML artefact that produced it, at generation time, so the report cannot
disagree with the pipeline without the disagreement being visible as a stale
artefact. F-019 is the reason this matters: a register entry hand-typed as 32
observations described a fit that had only ever used 24, and nothing that read
the register could see it. A report that reads the artefacts can.

A missing artefact is rendered as a missing artefact, naming the command that
produces it, rather than as a blank or a guess.
"""

from riskos.report.build import build_model_card, build_report, run
from riskos.report.sources import Sources, load_sources

__all__ = ["Sources", "build_model_card", "build_report", "load_sources", "run"]
