"""Phase 7 — the report is generated from artefacts, and says so when they are absent.

The rendering helpers are tested against hand-written expectations. The
end-to-end build runs against whatever artefacts are present: on a clone with no
data it must still produce a document that names every missing artefact, and on
the full estate every finding id and every inventoried model must appear.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl
import pytest

from riskos.report import build_model_card, build_report, load_sources, render
from riskos.report.sources import Sources

FULL_ESTATE = (
    Path("reports/figures/ecl_by_stage.csv").exists()
    and Path("reports/figures/champion_challenger_metrics.csv").exists()
)


# --- rendering -------------------------------------------------------------------


def test_money_picks_a_magnitude_suffix() -> None:
    assert render.money(34_008_220.7) == "$34.0M"
    assert render.money(15_908_209_546.6) == "$15.91B"
    assert render.money(1234.5) == "$1,234"
    assert render.money(None) == "—"


def test_pct_renders_a_fraction() -> None:
    assert render.pct(0.002137778, 3) == "0.214%"
    assert render.pct(1.0, 0) == "100%"


def test_num_uses_thousands_separators() -> None:
    assert render.num(2046874) == "2,046,874"
    assert render.num(0.5, 2) == "0.50"


def test_md_table_is_github_flavoured() -> None:
    table = render.md_table(["a", "b"], [[1, "x"], [2, "y"]])

    assert table.splitlines()[1] == "| --- | --- |"
    assert table.splitlines()[-1] == "| 2 | y |"


def test_df_table_applies_formatters_per_column() -> None:
    frame = pl.DataFrame({"k": ["s1"], "v": [0.25], "n": [1000]})

    table = render.df_table(
        frame, [("k", "Split", render.text), ("v", "Rate", render.pct), ("n", "Rows", render.num)]
    )

    assert "| s1 | 25.00% | 1,000 |" in table


def test_squash_collapses_folded_yaml() -> None:
    assert render.squash("a\n  b   c\n") == "a b c"


# --- sources ---------------------------------------------------------------------------


def test_a_missing_artefact_is_recorded_not_raised(tmp_path: Path) -> None:
    s = Sources(root=tmp_path, figures=tmp_path, models=tmp_path, governance=tmp_path)

    assert s.csv("ecl_by_stage") is None
    assert "ecl_by_stage" in s.missing
    assert s.produced_by("ecl_by_stage") == "riskos ecl"


def test_governance_sources_read_the_committed_files() -> None:
    s = load_sources()

    assert any(f["id"] == "F-001" for f in s.findings)
    assert s.inventory["models"]
    assert s.assumption("PSI_001") is not None


# --- the build --------------------------------------------------------------------------


def test_the_report_builds_on_an_empty_estate_and_names_what_is_missing(tmp_path: Path) -> None:
    """A clone with no data must still produce a document, not a traceback."""
    s = Sources(
        root=tmp_path,
        figures=tmp_path / "figures",
        models=tmp_path / "models",
        governance=Path("governance"),
    )

    report = build_report(s)

    assert "# RiskOS — Model Validation Report" in report
    assert "Artefact missing" in report
    assert "riskos ecl" in report
    assert "Artefacts missing at generation" in report


def test_every_finding_and_every_model_appears_in_the_report() -> None:
    s = load_sources()

    report = build_report(s)

    for f in s.findings:
        assert f["id"] in report, f"{f['id']} not in the report"
    for m in s.inventory["models"]:
        assert m["id"] in report
    assert "developer validation with simulated second-line review" in report


def test_the_report_carries_the_scope_statement_verbatim() -> None:
    report = build_report(load_sources())

    assert "makes no claim of OSFI compliance" in report


@pytest.mark.skipif(not FULL_ESTATE, reason="needs the Phase 4 and Phase 5 artefacts")
def test_the_headline_ecl_is_the_artefact_total() -> None:
    """The one number a reader will quote must be the CSV's, not a transcription."""
    s = load_sources()
    stage = pl.read_csv("reports/figures/ecl_by_stage.csv")
    total = float(stage["ecl"].sum())

    report = build_report(s)

    assert f"Portfolio ECL as at the last training date is {render.money(total)}" in report
    assert not s.missing, f"artefacts missing on a full estate: {s.missing}"


@pytest.mark.skipif(not FULL_ESTATE, reason="needs the Phase 4 artefacts")
def test_the_stress_calibration_ratios_come_from_the_comparison_table() -> None:
    cc = pl.read_csv("reports/figures/champion_challenger_metrics.csv")
    row = cc.filter(
        (pl.col("model") == "scorecard")
        & (pl.col("calibration") == "uncalibrated")
        & (pl.col("split") == "oot_stress")
    ).row(0, named=True)

    report = build_report(load_sources())

    assert (
        f"observed-over-expected ratio of {render.ratio(row['observed_over_expected'])}" in report
    )


def test_the_model_card_has_one_section_per_inventoried_model() -> None:
    s = load_sources()

    card = build_model_card(s)

    headings = re.findall(r"^## (RISKOS_[A-Z]+_\d+)", card, flags=re.MULTILINE)
    assert headings == [m["id"] for m in s.inventory["models"]]
    assert "Prohibited use" in card
