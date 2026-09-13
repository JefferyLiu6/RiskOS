"""Parsing must be loud: wrong width and bad values abort, sentinels become null."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from riskos.config import ColumnSpec, FileFormat
from riskos.ingest import parse


def test_reads_and_casts_declared_types(
    fixture_clean_file: Path, columns: tuple[ColumnSpec, ...], fmt: FileFormat
) -> None:
    df = parse.read_table(fixture_clean_file, columns, fmt)

    assert df.height == 3
    assert df.columns == ["seq", "amount", "period"]
    assert df.schema["amount"] == pl.Float64
    assert df.schema["period"] == pl.Date
    assert df["amount"][0] == pytest.approx(100.50)
    assert df["period"][0] == date(1999, 3, 1)


def test_whitespace_and_sentinels_become_null(
    fixture_clean_file: Path, columns: tuple[ColumnSpec, ...], fmt: FileFormat
) -> None:
    df = parse.read_table(fixture_clean_file, columns, fmt)

    # Row 2 is whitespace, row 3 is the per-column sentinel 9999 from the guide.
    assert df["amount"].to_list() == [pytest.approx(100.50), None, None]
    assert df["amount"].null_count() == 2


def test_unparseable_value_aborts_rather_than_nulling(
    fixture_bad_numeric_file: Path, columns: tuple[ColumnSpec, ...], fmt: FileFormat
) -> None:
    with pytest.raises(parse.CoercionError, match="NOT_A_NUMBER"):
        parse.read_table(fixture_bad_numeric_file, columns, fmt)


def test_coercion_report_counts_and_names_the_column(
    fixture_bad_numeric_file: Path, columns: tuple[ColumnSpec, ...], fmt: FileFormat
) -> None:
    raw = parse.read_raw(fixture_bad_numeric_file, columns, fmt)
    _, failures = parse.cast(parse.clean(raw, columns, fmt), columns)

    assert len(failures) == 1
    assert failures[0].column == "amount"
    assert failures[0].n_failed == 1
    assert failures[0].examples == ("NOT_A_NUMBER",)


def test_width_mismatch_names_both_counts(
    fixture_wrong_width_file: Path, columns: tuple[ColumnSpec, ...], fmt: FileFormat
) -> None:
    with pytest.raises(parse.ColumnCountError, match="4 fields but the layout declares 3"):
        parse.read_table(fixture_wrong_width_file, columns, fmt)


def test_date_column_requires_a_format() -> None:
    with pytest.raises(ValueError, match="no date_format"):
        ColumnSpec(name="period", dtype="date")
