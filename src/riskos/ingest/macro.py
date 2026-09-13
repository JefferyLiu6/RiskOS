"""FRED macroeconomic series: fetch once, cache as Parquet, build offline after.

Build plan §4.2. The API key is read from the environment via pydantic-settings
and never committed. Every pull is cached to ``data/macro/`` so the build is
reproducible without network access, and a cached series is preferred over a
refetch unless refresh is requested — a macro series that silently changes
underneath a fitted model is a reproducibility failure.

Series are stored at their native frequency. Aggregating weekly mortgage rates
or interpolating quarterly GDP into months is a modelling choice with its own
assumptions, and belongs in feature engineering, not here.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import polars as pl

from riskos.config import DataConfig, MacroConfig, env
from riskos.log import get_logger

log = get_logger(__name__)

MANIFEST_NAME = "macro_manifest.json"


class FredClient(Protocol):
    """The one method used from ``fredapi.Fred``, so tests need no network."""

    def get_series(self, series_id: str) -> Any: ...


class MissingApiKeyError(RuntimeError):
    """No FRED API key available and no cached copy to fall back on."""


class MacroFetchError(RuntimeError):
    """A series could not be retrieved."""


def cache_path(cfg: MacroConfig, series_id: str) -> Path:
    return cfg.cache / f"{series_id}.parquet"


def series_ids(cfg: MacroConfig) -> list[str]:
    """Every series to pull: the named set plus any per-state HPI."""
    ids = list(cfg.series.values())
    if cfg.state_house_prices:
        ids += [
            cfg.state_house_prices.pattern.format(state=s.upper())
            for s in cfg.state_house_prices.states
        ]
    return list(dict.fromkeys(ids))


def to_frame(series_id: str, raw: Any) -> pl.DataFrame:
    """Normalise a FRED response into (series_id, date, value).

    FRED encodes a missing observation as NaN. It is converted to null and
    counted, never dropped or filled — a gap in a macro series is information
    about the series (build plan rule 6).
    """
    frame = (
        pl.DataFrame(
            {
                "series_id": [series_id] * len(raw),
                "date": pl.Series(list(raw.index)).cast(pl.Date),
                "value": pl.Series(raw.to_numpy(), dtype=pl.Float64).fill_nan(None),
            }
        )
        .sort("date")
        .unique(subset=["date"], keep="first", maintain_order=True)
    )
    missing = int(frame["value"].null_count())
    if missing:
        log.warning("macro_series_has_gaps", series_id=series_id, missing=missing)
    return frame


def _default_client(api_key: str) -> FredClient:
    from fredapi import Fred  # imported lazily so offline builds need no network stack

    return Fred(api_key=api_key)  # type: ignore[no-any-return]


def _fetch(client: FredClient, series_id: str) -> pl.DataFrame:
    try:
        raw = client.get_series(series_id)
    except Exception as exc:
        raise MacroFetchError(f"could not fetch {series_id} from FRED: {exc}") from exc
    if raw is None or len(raw) == 0:
        raise MacroFetchError(f"FRED returned no observations for {series_id}")
    return to_frame(series_id, raw)


def _resolve_client(
    api_key: str | None, factory: Callable[[str], FredClient] | None, needed: list[str]
) -> FredClient:
    key = api_key or env().fred_api_key
    if not key:
        raise MissingApiKeyError(
            f"{len(needed)} series are not cached ({', '.join(needed[:5])}...) and "
            "RISKOS_FRED_API_KEY is not set. Get a free key at "
            "https://fredaccount.stlouisfed.org/apikeys and put it in .env "
            "(see .env.example). The key is never committed."
        )
    return (factory or _default_client)(key)


def pull(
    cfg: DataConfig,
    *,
    refresh: bool = False,
    api_key: str | None = None,
    client_factory: Callable[[str], FredClient] | None = None,
) -> dict[str, pl.DataFrame]:
    """Return every configured series, fetching only what is not already cached."""
    macro = cfg.macro
    macro.cache.mkdir(parents=True, exist_ok=True)
    wanted = series_ids(macro)
    if not wanted:
        raise ValueError("conf/data.yaml macro.series is empty; nothing to pull")

    cached = {s: cache_path(macro, s) for s in wanted}
    stale = [s for s, p in cached.items() if refresh or not p.exists()]
    out: dict[str, pl.DataFrame] = {
        s: pl.read_parquet(p) for s, p in cached.items() if s not in stale
    }
    for series_id in out:
        log.info("macro_cache_hit", series_id=series_id)

    if stale:
        client = _resolve_client(api_key, client_factory, stale)
        for series_id in stale:
            frame = _fetch(client, series_id)
            frame.write_parquet(cached[series_id], compression="zstd")
            out[series_id] = frame
            log.info(
                "macro_fetched",
                series_id=series_id,
                rows=frame.height,
                start=str(frame["date"].min()),
                end=str(frame["date"].max()),
            )

    _write_manifest(macro, out, fetched=stale)
    return out


def _write_manifest(cfg: MacroConfig, frames: dict[str, pl.DataFrame], fetched: list[str]) -> None:
    """Record what was pulled, when, and over what span."""
    manifest = {
        "written_at": datetime.now(UTC).isoformat(),
        "fetched_this_run": sorted(fetched),
        "series": {
            series_id: {
                "rows": frame.height,
                "start": str(frame["date"].min()),
                "end": str(frame["date"].max()),
                "missing_values": int(frame["value"].null_count()),
            }
            for series_id, frame in sorted(frames.items())
        },
    }
    (cfg.cache / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
