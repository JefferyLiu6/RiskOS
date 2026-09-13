"""Zero-balance code to ``termination_type`` mapping.

Codes encode how a loan left the book — prepayment, third-party sale, short
sale, REO disposition, note sale, repurchase, and others. They are mapped from
the User Guide, never guessed (build plan §4.1). An unmapped code aborts ingest:
a Phase 1 acceptance criterion is that the mapping is complete with no unmapped
values, and defaulting an unknown code to ``other`` would quietly corrupt the
default definition in §7.1.
"""

from __future__ import annotations

import polars as pl

from riskos.config import ZeroBalanceCodes
from riskos.log import get_logger

log = get_logger(__name__)

TERMINATION_TYPE = "termination_type"


class UnmappedCodeError(ValueError):
    """A zero-balance code was observed that the User Guide mapping does not cover."""


def observed_codes(df: pl.DataFrame, column: str) -> tuple[str, ...]:
    """Distinct non-null codes present in the frame, as strings."""
    if column not in df.columns:
        raise KeyError(f"zero-balance column {column!r} not in frame")
    values = df[column].drop_nulls().cast(pl.String).unique().sort()
    return tuple(values.to_list())


def check_exhaustive(df: pl.DataFrame, codes: ZeroBalanceCodes, *, label: str = "") -> None:
    """Raise if the frame contains a code the mapping does not cover."""
    codes.require_ready()
    assert codes.column is not None  # narrowed by require_ready
    present = observed_codes(df, codes.column)
    unmapped = tuple(c for c in present if c not in codes.mapping)
    if unmapped:
        raise UnmappedCodeError(
            f"{label or 'frame'}: zero-balance codes {list(unmapped)} are not in "
            f"conf/data.yaml zero_balance_codes.mapping. Map them from the User "
            f"Guide. Do not fold unknown codes into 'other' — termination type "
            f"drives the default definition."
        )
    log.info("zero_balance_codes_checked", label=label, distinct=len(present))


def apply(df: pl.DataFrame, codes: ZeroBalanceCodes, *, label: str = "") -> pl.DataFrame:
    """Attach ``termination_type``. Null code means the loan is still on book."""
    check_exhaustive(df, codes, label=label)
    assert codes.column is not None
    return df.with_columns(
        pl.col(codes.column)
        .cast(pl.String)
        .replace_strict(codes.mapping, default=None, return_dtype=pl.String)
        .cast(pl.Enum(list(codes.enum)))
        .alias(TERMINATION_TYPE)
    )
