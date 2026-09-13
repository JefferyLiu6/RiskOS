"""Typed configuration.

Every numeric assumption and every path lives in ``conf/*.yaml`` and is loaded
through a pydantic model here. Modules must not hardcode them (build plan §5,
rule 5). Unknown YAML keys are a hard error, so a typo in config fails loudly
rather than silently falling back to a default.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = PROJECT_ROOT / "conf"

DType = Literal["str", "int", "float", "date"]


class Strict(BaseModel):
    """Base model: reject unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Env(BaseSettings):
    """Secrets, from the environment or a gitignored .env. Never committed."""

    model_config = SettingsConfigDict(
        env_prefix="RISKOS_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    fred_api_key: str | None = None


@lru_cache(maxsize=1)
def env() -> Env:
    return Env()


class LayoutNotLoadedError(RuntimeError):
    """Raised when ingest is attempted before the User Guide layout is recorded."""


class ColumnSpec(Strict):
    """One column in a pipe-delimited SFLLD file, in file order.

    Populated from the *Single Family Loan-Level Dataset General User Guide*
    shipped with the download — never from memory (build plan §4.1).
    """

    name: str
    dtype: DType
    description: str = ""
    # Strings treated as null before casting, in addition to the file-level set.
    # Per-column sentinels (e.g. a credit score of 9999 meaning "not available")
    # come from the User Guide and belong here, not in code.
    null_values: tuple[str, ...] = ()
    # For dtype == "date": a chrono format string, e.g. "%Y%m" or "%Y%m%d".
    date_format: str | None = None

    @model_validator(mode="after")
    def _date_needs_format(self) -> ColumnSpec:
        if self.dtype == "date" and not self.date_format:
            raise ValueError(f"column {self.name!r} has dtype 'date' but no date_format")
        return self


class Layout(Strict):
    """Column layouts, sourced from the User Guide and version-stamped."""

    version: str | None = None
    guide_path: str | None = None
    # The loan key, named rather than inferred from position. In the real SFLLD
    # layout it is field 20 of the origination file and field 1 of the
    # performance file, so any positional assumption is wrong on one of them.
    key_column: str = "loan_sequence_number"
    origination_columns: tuple[ColumnSpec, ...] = ()
    performance_columns: tuple[ColumnSpec, ...] = ()

    @model_validator(mode="after")
    def _key_is_present_in_both_files(self) -> Layout:
        for kind, cols in (
            ("origination", self.origination_columns),
            ("performance", self.performance_columns),
        ):
            if cols and self.key_column not in {c.name for c in cols}:
                raise ValueError(f"key_column {self.key_column!r} is not in the {kind} layout")
        return self

    @property
    def is_ready(self) -> bool:
        return bool(self.version and self.origination_columns and self.performance_columns)

    def require_ready(self) -> None:
        """Refuse to proceed without a version-stamped layout."""
        if self.is_ready:
            return
        raise LayoutNotLoadedError(
            "conf/data.yaml layout is not populated. Record the Single Family "
            "Loan-Level Dataset General User Guide version and its origination "
            "and performance column layouts before ingest. Column positions are "
            "taken from the guide, never from memory (build plan §4.1)."
        )

    def columns_for(self, kind: Literal["origination", "performance"]) -> tuple[ColumnSpec, ...]:
        self.require_ready()
        return self.origination_columns if kind == "origination" else self.performance_columns


class Paths(Strict):
    raw: Path
    interim: Path
    panel: Path
    macro: Path

    def resolve(self, root: Path = PROJECT_ROOT) -> Paths:
        """Resolve any relative path against the project root."""
        return Paths(**{k: (root / v) if not v.is_absolute() else v for k, v in self})


class Vintages(Strict):
    core: tuple[int, ...]
    benign: tuple[int, ...]

    @property
    def all(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.core) | set(self.benign)))


class NamingConvention(Strict):
    """One candidate filename convention. Freddie Mac has used more than one."""

    label: str
    origination: str
    performance: str


class Discovery(Strict):
    candidate_patterns: tuple[NamingConvention, ...]
    search_recursive: bool = True


class FileFormat(Strict):
    delimiter: str = "|"
    header: bool = False
    encoding: str = "utf-8"
    # Whitespace-only fields are Freddie's usual "missing"; casting them is not
    # a coercion failure.
    null_values: tuple[str, ...] = ("",)
    treat_whitespace_as_null: bool = True


class ZeroBalanceCodes(Strict):
    """Termination-type mapping, sourced from the User Guide."""

    # Performance-file column holding the code; its name comes from the guide.
    column: str | None = None
    enum: tuple[str, ...]
    mapping: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _values_in_enum(self) -> ZeroBalanceCodes:
        unknown = sorted(set(self.mapping.values()) - set(self.enum))
        if unknown:
            raise ValueError(f"zero_balance_codes.mapping targets not in enum: {unknown}")
        return self

    @property
    def is_ready(self) -> bool:
        return bool(self.mapping and self.column)

    def require_ready(self) -> None:
        if self.is_ready:
            return
        raise LayoutNotLoadedError(
            "conf/data.yaml zero_balance_codes is not populated. Map every code "
            "in the User Guide to a termination_type and name the column holding "
            "it. Numeric codes are not to be guessed (build plan §4.1)."
        )


class StateHousePrices(Strict):
    """State-level HPI series, preferred over a national index (build plan §4.2)."""

    pattern: str
    states: tuple[str, ...] = ()


class MacroConfig(Strict):
    provider: str
    cache: Path
    series: dict[str, str]
    state_house_prices: StateHousePrices | None = None


class SourceInfo(Strict):
    name: str
    variant: str
    access: str
    licence_note: str


class DataConfig(Strict):
    source: SourceInfo
    paths: Paths
    vintages: Vintages
    discovery: Discovery
    format: FileFormat
    layout: Layout
    zero_balance_codes: ZeroBalanceCodes
    macro: MacroConfig

    @field_validator("paths")
    @classmethod
    def _resolve_paths(cls, v: Paths) -> Paths:
        return v.resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping, failing loudly on a missing or non-mapping file."""
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"expected a YAML mapping at {path}, got {type(loaded).__name__}")
    return loaded


@lru_cache(maxsize=1)
def data_config(conf_dir: Path = CONF_DIR) -> DataConfig:
    """Load and validate ``conf/data.yaml``."""
    raw = load_yaml(conf_dir / "data.yaml")
    # Documentation-only keys are stripped rather than tolerated, so that
    # `extra="forbid"` still catches real typos in the modelled sections.
    raw.pop("status", None)
    return DataConfig.model_validate(raw)
