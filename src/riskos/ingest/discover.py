"""Locate raw SFLLD files under ``data/raw/``.

Access to the dataset is manual (build plan §4.1): the human downloads from
Clarity Data Intelligence. This module never fetches anything. It reports what
is present, what is missing, and what it could not classify.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from riskos.config import DataConfig, NamingConvention
from riskos.log import get_logger

log = get_logger(__name__)

FileKind = str  # "origination" | "performance"


@dataclass(frozen=True)
class VintageFiles:
    """Files found for one vintage year."""

    year: int
    convention: str | None
    origination: tuple[Path, ...] = ()
    performance: tuple[Path, ...] = ()

    @property
    def is_complete(self) -> bool:
        return bool(self.origination and self.performance)

    @property
    def missing(self) -> tuple[FileKind, ...]:
        gaps: list[FileKind] = []
        if not self.origination:
            gaps.append("origination")
        if not self.performance:
            gaps.append("performance")
        return tuple(gaps)


@dataclass
class DiscoveryReport:
    """What ingest found on disk, and what it did not."""

    root: Path
    found: dict[int, VintageFiles] = field(default_factory=dict)
    unclassified: tuple[Path, ...] = ()

    @property
    def complete_years(self) -> tuple[int, ...]:
        return tuple(sorted(y for y, v in self.found.items() if v.is_complete))

    def missing_years(self, expected: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(y for y in expected if y not in self.complete_years)


def _match(root: Path, pattern: str, year: int) -> tuple[Path, ...]:
    """Glob one pattern for one vintage year, sorted for determinism."""
    return tuple(sorted(root.glob(pattern.format(year=year))))


def _discover_year(
    root: Path, year: int, conventions: tuple[NamingConvention, ...]
) -> VintageFiles:
    """Try each naming convention in order; first with any hit wins."""
    for conv in conventions:
        orig = _match(root, conv.origination, year)
        perf = _match(root, conv.performance, year)
        if orig or perf:
            return VintageFiles(
                year=year, convention=conv.label, origination=orig, performance=perf
            )
    return VintageFiles(year=year, convention=None)


def discover(cfg: DataConfig) -> DiscoveryReport:
    """Scan ``paths.raw`` for every expected vintage.

    Returns a report rather than raising: the caller decides whether an
    incomplete download is fatal.
    """
    root = cfg.paths.raw
    report = DiscoveryReport(root=root)
    if not root.exists():
        log.error("raw_dir_missing", path=str(root))
        return report

    claimed: set[Path] = set()
    for year in cfg.vintages.all:
        vf = _discover_year(root, year, cfg.discovery.candidate_patterns)
        report.found[year] = vf
        claimed.update(vf.origination)
        claimed.update(vf.performance)
        if vf.is_complete:
            log.info(
                "vintage_found",
                year=year,
                convention=vf.convention,
                origination=len(vf.origination),
                performance=len(vf.performance),
            )
        else:
            log.warning("vintage_incomplete", year=year, missing=vf.missing)

    report.unclassified = tuple(sorted(set(root.rglob("*.txt")) - claimed))
    for path in report.unclassified:
        log.warning("unclassified_file", path=str(path.relative_to(root)))
    return report


def require_complete(report: DiscoveryReport, expected: tuple[int, ...]) -> None:
    """Raise if any expected vintage is absent or half-present."""
    missing = report.missing_years(expected)
    if not missing:
        return
    # A year with neither file present reports both as missing; a half-present
    # vintage names only the side that is absent.
    detail = ", ".join(
        f"{y} (missing {'+'.join(report.found[y].missing) if y in report.found else 'both files'})"
        for y in missing
    )
    raise FileNotFoundError(
        f"{len(missing)} of {len(expected)} vintages unavailable under {report.root}: {detail}. "
        "Download the SFLLD sample dataset manually from Clarity Data Intelligence. "
        "No synthetic substitute will be generated (build plan rule 4)."
    )
