"""End-to-end Phase 1: raw pipe-delimited files to validated, partitioned Parquet.

The layout here is a four-column toy, not the Freddie Mac schema. Its purpose is
to prove the machinery — discovery, parsing, schema validation, zero-balance
mapping, Parquet landing, manifest — works as a unit, so that when the real
files and the User Guide arrive the only remaining variable is the layout
transcription in conf/data.yaml.

No plausible-looking mortgage records are invented (build plan rule 4): the
fields are named for their role and the values are obviously synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from riskos.config import DataConfig
from riskos.ingest import parse, zero_balance
from riskos.ingest.run import MANIFEST_NAME, run

FIXTURE_LAYOUT: dict[str, Any] = {
    "version": "fixture layout v0 (test only — not the SFLLD User Guide)",
    "guide_path": None,
    "key_column": "loan_seq",
    # The key deliberately sits in the LAST position here, mirroring the real
    # SFLLD layout where loan_sequence_number is field 20 of the origination
    # file but field 1 of the performance file. A fixture with the key first
    # hides any code that infers it positionally.
    "origination_columns": [
        {"name": "score_at_origination", "dtype": "int", "null_values": ["9999"]},
        {"name": "first_payment_date", "dtype": "date", "date_format": "%Y%m"},
        {"name": "loan_seq", "dtype": "str"},
    ],
    "performance_columns": [
        {"name": "loan_seq", "dtype": "str"},
        {"name": "reporting_period", "dtype": "date", "date_format": "%Y%m"},
        {"name": "current_upb", "dtype": "float"},
        {"name": "zb_code", "dtype": "str"},
    ],
}


def _config(tmp_path: Path) -> DataConfig:
    return DataConfig.model_validate(
        {
            "source": {
                "name": "fixture",
                "variant": "sample",
                "access": "manual",
                "licence_note": "test fixture",
            },
            "paths": {
                "raw": tmp_path / "raw",
                "interim": tmp_path / "interim",
                "panel": tmp_path / "panel",
                "macro": tmp_path / "macro",
            },
            "vintages": {"core": [1999], "benign": []},
            "discovery": {
                "candidate_patterns": [
                    {
                        "label": "sample_annual",
                        "origination": "**/sample_orig_{year}.txt",
                        "performance": "**/sample_svcg_{year}.txt",
                    }
                ]
            },
            "format": {},
            "layout": FIXTURE_LAYOUT,
            "zero_balance_codes": {
                "column": "zb_code",
                "enum": ["prepaid", "reo_disposition", "other"],
                "mapping": {"01": "prepaid", "09": "reo_disposition"},
            },
            "macro": {"provider": "FRED", "cache": tmp_path / "macro", "series": {}},
        }
    )


@pytest.fixture
def raw_vintage(tmp_path: Path) -> Path:
    """One complete 1999 vintage: 3 loans, 5 loan-months, 2 terminations."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "sample_orig_1999.txt").write_text(
        "720|199903|L0000000001\n"  # ordinary
        "9999|199903|L0000000002\n"  # sentinel score -> null
        "  680 |199906|L0000000003\n",  # padded, stripped then cast
        encoding="utf-8",
    )
    (raw / "sample_svcg_1999.txt").write_text(
        "L0000000001|199904|100000.00|\n"  # still on book
        "L0000000001|199905|99000.00|01\n"  # prepaid
        "L0000000002|199904|200000.00|\n"  # still on book
        "L0000000002|199905|199000.00|09\n"  # REO disposition
        "L0000000003|199907|300000.00|\n",  # still on book
        encoding="utf-8",
    )
    return raw


@pytest.mark.usefixtures("raw_vintage")
def test_full_pipeline_lands_partitioned_parquet(tmp_path: Path) -> None:
    cfg = _config(tmp_path)

    report = run(cfg)

    assert [v.year for v in report.vintages] == [1999]
    assert report.vintages[0].origination_rows == 3
    assert report.vintages[0].performance_rows == 5
    assert report.total_rows == 8
    assert (tmp_path / "interim" / "origination" / "vintage=1999" / "part-0.parquet").exists()
    assert (tmp_path / "interim" / "performance" / "vintage=1999" / "part-0.parquet").exists()


@pytest.mark.usefixtures("raw_vintage")
def test_declared_types_survive_the_round_trip(tmp_path: Path) -> None:
    run(_config(tmp_path))

    orig = pl.read_parquet(tmp_path / "interim" / "origination" / "vintage=1999" / "part-0.parquet")

    assert orig.schema["score_at_origination"] == pl.Int64
    assert orig.schema["first_payment_date"] == pl.Date
    # The 9999 sentinel from the layout became null, not the integer 9999.
    assert orig["score_at_origination"].to_list() == [720, None, 680]


@pytest.mark.usefixtures("raw_vintage")
def test_termination_type_is_attached_and_live_loans_stay_null(tmp_path: Path) -> None:
    run(_config(tmp_path))

    perf = pl.read_parquet(tmp_path / "interim" / "performance" / "vintage=1999" / "part-0.parquet")

    assert perf[zero_balance.TERMINATION_TYPE].to_list() == [
        None,
        "prepaid",
        None,
        "reo_disposition",
        None,
    ]


@pytest.mark.usefixtures("raw_vintage")
def test_manifest_records_lineage(tmp_path: Path) -> None:
    run(_config(tmp_path))

    manifest = json.loads((tmp_path / "interim" / MANIFEST_NAME).read_text(encoding="utf-8"))

    assert manifest["layout_version"].startswith("fixture layout v0")
    assert manifest["vintages"][0]["convention"] == "sample_annual"
    assert manifest["vintages"][0]["performance_rows"] == 5


def test_a_duplicate_loan_in_origination_fails_schema_validation(
    tmp_path: Path, raw_vintage: Path
) -> None:
    # The loan sequence number is the origination key. A duplicate silently
    # doubles a loan's weight in every downstream cohort.
    (raw_vintage / "sample_orig_1999.txt").write_text(
        "720|199903|L0000000001\n700|199904|L0000000001\n", encoding="utf-8"
    )

    with pytest.raises(Exception, match=r"(?i)unique|duplicate"):
        run(_config(tmp_path))


def test_an_unmapped_zero_balance_code_aborts_the_whole_run(
    tmp_path: Path, raw_vintage: Path
) -> None:
    (raw_vintage / "sample_svcg_1999.txt").write_text(
        "L0000000001|199904|100000.00|96\n", encoding="utf-8"
    )

    with pytest.raises(zero_balance.UnmappedCodeError, match=r"\['96'\]"):
        run(_config(tmp_path))


def test_a_malformed_number_aborts_rather_than_becoming_null(
    tmp_path: Path, raw_vintage: Path
) -> None:
    (raw_vintage / "sample_svcg_1999.txt").write_text("L0000000001|199904|N/A|\n", encoding="utf-8")

    with pytest.raises(parse.CoercionError, match="current_upb"):
        run(_config(tmp_path))


def test_an_incomplete_vintage_blocks_the_run_before_anything_is_written(
    tmp_path: Path, raw_vintage: Path
) -> None:
    (raw_vintage / "sample_svcg_1999.txt").unlink()

    with pytest.raises(FileNotFoundError, match="1999"):
        run(_config(tmp_path))
    assert not (tmp_path / "interim").exists()
