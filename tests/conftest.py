"""Shared fixtures.

Fixture data here is deliberately NOT loan data. It exercises the layout-driven
machinery with a three-column toy layout, so no test depends on the real
Freddie Mac schema or invents plausible-looking mortgage records (build plan
rule 4).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from riskos.config import ColumnSpec, FileFormat, ZeroBalanceCodes

FIXTURE_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec(name="seq", dtype="str", description="toy key"),
    ColumnSpec(name="amount", dtype="float", null_values=("9999",)),
    ColumnSpec(name="period", dtype="date", date_format="%Y%m"),
)


@pytest.fixture
def columns() -> tuple[ColumnSpec, ...]:
    return FIXTURE_COLUMNS


@pytest.fixture
def fmt() -> FileFormat:
    return FileFormat()


@pytest.fixture
def fixture_clean_file(tmp_path: Path) -> Path:
    """Three well-formed rows, including a blank and a sentinel for `amount`."""
    path = tmp_path / "fixture_clean.txt"
    path.write_text(
        "F001|100.50|199903\n"  # ordinary
        "F002|   |199906\n"  # whitespace -> null
        "F003|9999|199909\n",  # per-column sentinel -> null
        encoding="utf-8",
    )
    return path


@pytest.fixture
def fixture_bad_numeric_file(tmp_path: Path) -> Path:
    """`amount` holds a value that is neither numeric nor a declared sentinel."""
    path = tmp_path / "fixture_bad_numeric.txt"
    path.write_text("F001|100.50|199903\nF002|NOT_A_NUMBER|199906\n", encoding="utf-8")
    return path


@pytest.fixture
def fixture_wrong_width_file(tmp_path: Path) -> Path:
    """Four fields where the layout declares three."""
    path = tmp_path / "fixture_wrong_width.txt"
    path.write_text("F001|100.50|199903|EXTRA\n", encoding="utf-8")
    return path


@pytest.fixture
def zbc() -> ZeroBalanceCodes:
    return ZeroBalanceCodes(
        column="zb_code",
        enum=("prepaid", "reo_disposition", "other"),
        mapping={"01": "prepaid", "09": "reo_disposition"},
    )
