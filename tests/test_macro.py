"""FRED ingestion. No test touches the network: the client is injected."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from riskos.config import DataConfig
from riskos.ingest import macro


class FakeFred:
    """Stands in for ``fredapi.Fred``, recording what was asked for."""

    def __init__(self, series: dict[str, pd.Series] | None = None) -> None:
        self.series = series or {}
        self.calls: list[str] = []

    def get_series(self, series_id: str) -> Any:
        self.calls.append(series_id)
        if series_id not in self.series:
            raise KeyError(f"no such series {series_id}")
        return self.series[series_id]


def _unrate() -> pd.Series:
    return pd.Series(
        [4.0, 4.1, np.nan, 4.3],
        index=pd.to_datetime(["1999-01-01", "1999-02-01", "1999-03-01", "1999-04-01"]),
    )


def _config(tmp_path: Path, states: list[str] | None = None) -> DataConfig:
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
                    {"label": "x", "origination": "o_{year}.txt", "performance": "p_{year}.txt"}
                ]
            },
            "format": {},
            "layout": {},
            "zero_balance_codes": {"enum": ["other"]},
            "macro": {
                "provider": "FRED",
                "cache": tmp_path / "macro",
                "series": {"unemployment": "UNRATE"},
                "state_house_prices": {"pattern": "{state}STHPI", "states": states or []},
            },
        }
    )


def test_fetches_and_caches_as_parquet(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    fake = FakeFred({"UNRATE": _unrate()})

    out = macro.pull(cfg, api_key="test-key", client_factory=lambda _: fake)

    assert fake.calls == ["UNRATE"]
    assert out["UNRATE"].height == 4
    assert (tmp_path / "macro" / "UNRATE.parquet").exists()


def test_nan_becomes_null_and_is_counted_not_dropped(tmp_path: Path) -> None:
    cfg = _config(tmp_path)

    out = macro.pull(cfg, api_key="k", client_factory=lambda _: FakeFred({"UNRATE": _unrate()}))

    assert out["UNRATE"].height == 4  # the gap row survives
    assert out["UNRATE"]["value"].to_list() == [4.0, 4.1, None, 4.3]


def test_types_and_ordering(tmp_path: Path) -> None:
    cfg = _config(tmp_path)

    frame = macro.pull(cfg, api_key="k", client_factory=lambda _: FakeFred({"UNRATE": _unrate()}))[
        "UNRATE"
    ]

    assert frame.schema["date"] == pl.Date
    assert frame.schema["value"] == pl.Float64
    assert frame["date"].is_sorted()
    assert frame["series_id"].unique().to_list() == ["UNRATE"]


def test_second_pull_uses_the_cache_and_never_calls_fred(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    first = FakeFred({"UNRATE": _unrate()})
    macro.pull(cfg, api_key="k", client_factory=lambda _: first)

    second = FakeFred({"UNRATE": _unrate()})
    out = macro.pull(cfg, client_factory=lambda _: second)

    assert second.calls == []  # offline build
    assert out["UNRATE"].height == 4


def test_refresh_forces_a_refetch(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    macro.pull(cfg, api_key="k", client_factory=lambda _: FakeFred({"UNRATE": _unrate()}))

    again = FakeFred({"UNRATE": _unrate()})
    macro.pull(cfg, refresh=True, api_key="k", client_factory=lambda _: again)

    assert again.calls == ["UNRATE"]


def test_missing_key_with_nothing_cached_says_how_to_fix_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RISKOS_FRED_API_KEY", raising=False)
    monkeypatch.setattr(macro, "env", lambda: type("E", (), {"fred_api_key": None})())

    with pytest.raises(macro.MissingApiKeyError, match=r"fredaccount\.stlouisfed\.org"):
        macro.pull(_config(tmp_path))


def test_state_series_are_expanded_from_the_pattern(tmp_path: Path) -> None:
    cfg = _config(tmp_path, states=["ca", "NY"])

    assert macro.series_ids(cfg.macro) == ["UNRATE", "CASTHPI", "NYSTHPI"]


def test_a_failed_series_names_itself(tmp_path: Path) -> None:
    cfg = _config(tmp_path, states=["ZZ"])

    with pytest.raises(macro.MacroFetchError, match="ZZSTHPI"):
        macro.pull(cfg, api_key="k", client_factory=lambda _: FakeFred({"UNRATE": _unrate()}))


def test_manifest_records_span_and_gaps(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    macro.pull(cfg, api_key="k", client_factory=lambda _: FakeFred({"UNRATE": _unrate()}))

    import json

    manifest = json.loads((tmp_path / "macro" / macro.MANIFEST_NAME).read_text(encoding="utf-8"))

    assert manifest["series"]["UNRATE"]["rows"] == 4
    assert manifest["series"]["UNRATE"]["missing_values"] == 1
    assert manifest["series"]["UNRATE"]["start"] == "1999-01-01"
    assert manifest["fetched_this_run"] == ["UNRATE"]
