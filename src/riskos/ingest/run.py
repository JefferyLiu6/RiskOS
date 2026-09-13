"""Phase 1 orchestration: raw pipe-delimited files to validated Parquet.

Acceptance criteria (build plan Phase 1): every vintage parses without error,
row counts are logged, schema validation passes, the zero-balance mapping is
complete with no unmapped values, and nothing under ``data/`` is tracked by git.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from riskos.config import DataConfig, data_config
from riskos.ingest import discover as disco
from riskos.ingest import parse, schemas, zero_balance
from riskos.log import get_logger

log = get_logger(__name__)

MANIFEST_NAME = "ingest_manifest.json"


@dataclass
class VintageResult:
    year: int
    convention: str | None
    origination_rows: int
    performance_rows: int
    origination_path: str
    performance_path: str


@dataclass
class IngestReport:
    layout_version: str
    vintages: list[VintageResult]

    @property
    def total_rows(self) -> int:
        return sum(v.origination_rows + v.performance_rows for v in self.vintages)


def _write(df: pl.DataFrame, root: Path, kind: str, year: int) -> Path:
    """Land one table as Parquet, partitioned by vintage."""
    out_dir = root / kind / f"vintage={year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "part-0.parquet"
    df.write_parquet(path, compression="zstd")
    log.info("wrote_parquet", kind=kind, year=year, rows=df.height, path=str(path))
    return path


def _ingest_origination(cfg: DataConfig, files: disco.VintageFiles) -> tuple[pl.DataFrame, int]:
    columns = cfg.layout.columns_for("origination")
    key = cfg.layout.key_column
    schema = schemas.build_schema(columns, unique=(key,), non_null=(key,))
    frames = [parse.read_table(p, columns, cfg.format) for p in files.origination]
    df = pl.concat(frames, how="vertical")
    schemas.validate(df, schema, label=f"origination:{files.year}")
    return df, df.height


def _ingest_performance(cfg: DataConfig, files: disco.VintageFiles) -> tuple[pl.DataFrame, int]:
    columns = cfg.layout.columns_for("performance")
    schema = schemas.build_schema(columns, non_null=(cfg.layout.key_column,))
    frames = [parse.read_table(p, columns, cfg.format) for p in files.performance]
    df = pl.concat(frames, how="vertical")
    schemas.validate(df, schema, label=f"performance:{files.year}")
    return zero_balance.apply(
        df, cfg.zero_balance_codes, label=f"performance:{files.year}"
    ), df.height


def _ingest_vintage(cfg: DataConfig, files: disco.VintageFiles) -> VintageResult:
    orig, n_orig = _ingest_origination(cfg, files)
    perf, n_perf = _ingest_performance(cfg, files)
    return VintageResult(
        year=files.year,
        convention=files.convention,
        origination_rows=n_orig,
        performance_rows=n_perf,
        origination_path=str(_write(orig, cfg.paths.interim, "origination", files.year)),
        performance_path=str(_write(perf, cfg.paths.interim, "performance", files.year)),
    )


def preflight(cfg: DataConfig) -> disco.DiscoveryReport:
    """Fail fast on the three things that block ingest, in order of actionability."""
    report = disco.discover(cfg)
    disco.require_complete(report, cfg.vintages.all)
    cfg.layout.require_ready()
    cfg.zero_balance_codes.require_ready()
    return report


def run(cfg: DataConfig | None = None) -> IngestReport:
    """Ingest every configured vintage and write a lineage manifest."""
    cfg = cfg or data_config()
    found = preflight(cfg)
    assert cfg.layout.version is not None  # narrowed by preflight

    results = [_ingest_vintage(cfg, found.found[year]) for year in found.complete_years]
    report = IngestReport(layout_version=cfg.layout.version, vintages=results)

    cfg.paths.interim.mkdir(parents=True, exist_ok=True)
    manifest = cfg.paths.interim / MANIFEST_NAME
    manifest.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    log.info(
        "ingest_complete",
        vintages=len(results),
        total_rows=report.total_rows,
        manifest=str(manifest),
    )
    return report
