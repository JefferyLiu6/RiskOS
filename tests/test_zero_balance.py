"""An unmapped zero-balance code must abort, never fall back to 'other'."""

from __future__ import annotations

import polars as pl
import pytest

from riskos.config import LayoutNotLoadedError, ZeroBalanceCodes
from riskos.ingest import zero_balance


def _frame(codes: list[str | None]) -> pl.DataFrame:
    return pl.DataFrame({"zb_code": codes}, schema={"zb_code": pl.String})


def test_maps_known_codes_and_leaves_live_loans_null(zbc: ZeroBalanceCodes) -> None:
    out = zero_balance.apply(_frame(["01", "09", None]), zbc)

    assert out[zero_balance.TERMINATION_TYPE].to_list() == ["prepaid", "reo_disposition", None]


def test_unmapped_code_aborts(zbc: ZeroBalanceCodes) -> None:
    with pytest.raises(zero_balance.UnmappedCodeError, match=r"\['96'\]"):
        zero_balance.apply(_frame(["01", "96"]), zbc)


def test_empty_mapping_is_refused_before_any_data_is_read() -> None:
    unpopulated = ZeroBalanceCodes(column=None, enum=("prepaid",), mapping={})

    with pytest.raises(LayoutNotLoadedError, match="User Guide"):
        unpopulated.require_ready()


def test_mapping_target_outside_enum_is_a_config_error() -> None:
    with pytest.raises(ValueError, match="not in enum"):
        ZeroBalanceCodes(column="zb_code", enum=("prepaid",), mapping={"09": "reo_disposition"})


def test_observed_codes_are_sorted_and_deduplicated() -> None:
    assert zero_balance.observed_codes(_frame(["09", "01", "09", None]), "zb_code") == ("01", "09")
