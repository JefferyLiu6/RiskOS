"""Discovery reports what is on disk; it never fetches and never invents."""

from __future__ import annotations

from pathlib import Path

import pytest

from riskos.config import DataConfig, data_config
from riskos.ingest import discover as disco


def _cfg_rooted_at(tmp_path: Path) -> DataConfig:
    """Real conf/data.yaml, but pointed at a temporary raw directory."""
    cfg = data_config()
    raw = cfg.model_dump()
    raw["paths"]["raw"] = tmp_path
    return DataConfig.model_validate(raw)


def test_reports_missing_vintages_without_raising(tmp_path: Path) -> None:
    cfg = _cfg_rooted_at(tmp_path)

    report = disco.discover(cfg)

    assert report.complete_years == ()
    assert report.missing_years(cfg.vintages.all) == cfg.vintages.all


def test_require_complete_names_the_gaps(tmp_path: Path) -> None:
    cfg = _cfg_rooted_at(tmp_path)
    report = disco.discover(cfg)

    with pytest.raises(FileNotFoundError, match="Clarity Data Intelligence"):
        disco.require_complete(report, cfg.vintages.all)


def test_half_a_vintage_is_incomplete(tmp_path: Path) -> None:
    (tmp_path / "sample_orig_1999.txt").write_text("x\n", encoding="utf-8")
    cfg = _cfg_rooted_at(tmp_path)

    report = disco.discover(cfg)

    assert report.found[1999].origination
    assert report.found[1999].missing == ("performance",)
    assert 1999 not in report.complete_years


def test_matched_pair_is_complete_and_records_the_convention(tmp_path: Path) -> None:
    (tmp_path / "sample_orig_1999.txt").write_text("x\n", encoding="utf-8")
    (tmp_path / "sample_perf_1999.txt").write_text("x\n", encoding="utf-8")
    cfg = _cfg_rooted_at(tmp_path)

    report = disco.discover(cfg)

    assert report.complete_years == (1999,)
    assert report.found[1999].convention == "sample_annual"
    assert report.unclassified == ()


def test_stray_txt_files_are_surfaced_not_ignored(tmp_path: Path) -> None:
    (tmp_path / "readme_notes.txt").write_text("x\n", encoding="utf-8")
    cfg = _cfg_rooted_at(tmp_path)

    report = disco.discover(cfg)

    assert [p.name for p in report.unclassified] == ["readme_notes.txt"]
