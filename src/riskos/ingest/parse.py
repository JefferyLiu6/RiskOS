"""Layout-driven parsing of pipe-delimited SFLLD files.

Nothing here knows any column name or position. The layout comes from
``conf/data.yaml``, which is transcribed from the User Guide (build plan §4.1).

Casting is deliberately loud. Fields are first read as strings, then blanks and
User-Guide sentinels are nulled, then each column is cast to its declared type.
Any value that survives null-normalisation but fails to cast is a
``CoercionFailure`` and aborts ingest — silently nulling it would be exactly the
kind of quiet data loss this project exists to avoid. There is no tolerance
threshold, because a tolerance would be an unsourced magic number.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from riskos.config import ColumnSpec, FileFormat
from riskos.log import get_logger

log = get_logger(__name__)

_EXAMPLES_PER_FAILURE = 5


class ColumnCountError(ValueError):
    """File width does not match the declared layout."""


class CoercionError(ValueError):
    """One or more values failed to cast to their declared type."""


@dataclass(frozen=True)
class CoercionFailure:
    column: str
    dtype: str
    n_failed: int
    examples: tuple[str, ...]


def read_raw(path: Path, columns: tuple[ColumnSpec, ...], fmt: FileFormat) -> pl.DataFrame:
    """Read every field as a string, then apply the declared column names.

    Reading as strings first means a malformed numeric field is visible rather
    than being turned into a null by the CSV reader.
    """
    df = pl.read_csv(
        path,
        separator=fmt.delimiter,
        has_header=fmt.header,
        infer_schema_length=0,  # everything as Utf8
        truncate_ragged_lines=False,
        encoding="utf8" if fmt.encoding.replace("-", "") == "utf8" else fmt.encoding,
    )
    if df.width != len(columns):
        raise ColumnCountError(
            f"{path.name}: file has {df.width} fields but the layout declares "
            f"{len(columns)}. Check conf/data.yaml layout against the User Guide "
            f"version recorded there."
        )
    return df.rename(dict(zip(df.columns, [c.name for c in columns], strict=True)))


def _null_expr(spec: ColumnSpec, fmt: FileFormat) -> pl.Expr:
    """Blank out file-level and per-column sentinel values."""
    col = pl.col(spec.name)
    if fmt.treat_whitespace_as_null:
        col = col.str.strip_chars()
    sentinels = list(dict.fromkeys([*fmt.null_values, *spec.null_values]))
    return (pl.when(col.is_in(sentinels)).then(pl.lit(None, dtype=pl.String)).otherwise(col)).alias(
        spec.name
    )


def clean(df: pl.DataFrame, columns: tuple[ColumnSpec, ...], fmt: FileFormat) -> pl.DataFrame:
    """Normalise missing-value sentinels to null, still as strings."""
    return df.with_columns([_null_expr(spec, fmt) for spec in columns])


def _cast_expr(spec: ColumnSpec) -> pl.Expr:
    col = pl.col(spec.name)
    match spec.dtype:
        case "str":
            return col
        case "int":
            return col.cast(pl.Int64, strict=False)
        case "float":
            return col.cast(pl.Float64, strict=False)
        case "date":
            return col.str.to_date(format=spec.date_format, strict=False)
    raise AssertionError(f"unhandled dtype {spec.dtype!r}")  # pragma: no cover


def cast(
    cleaned: pl.DataFrame, columns: tuple[ColumnSpec, ...]
) -> tuple[pl.DataFrame, tuple[CoercionFailure, ...]]:
    """Cast to declared types, reporting every value that failed to convert."""
    casted = cleaned.with_columns([_cast_expr(spec) for spec in columns])
    failures: list[CoercionFailure] = []
    for spec in columns:
        if spec.dtype == "str":
            continue
        mask = cleaned[spec.name].is_not_null() & casted[spec.name].is_null()
        n_failed = int(mask.sum())
        if n_failed:
            offenders = cleaned.filter(mask)[spec.name].unique().head(_EXAMPLES_PER_FAILURE)
            failures.append(
                CoercionFailure(spec.name, spec.dtype, n_failed, tuple(offenders.to_list()))
            )
    return casted, tuple(failures)


def read_table(path: Path, columns: tuple[ColumnSpec, ...], fmt: FileFormat) -> pl.DataFrame:
    """Read, null-normalise, and cast one file. Raises on any coercion failure."""
    frame = clean(read_raw(path, columns, fmt), columns, fmt)
    casted, failures = cast(frame, columns)
    if failures:
        detail = "; ".join(
            f"{f.column} ({f.dtype}): {f.n_failed} values, e.g. {list(f.examples)}"
            for f in failures
        )
        raise CoercionError(
            f"{path.name}: {len(failures)} column(s) contain values that do not match "
            f"their declared type: {detail}. Either the layout is wrong or these are "
            f"undocumented sentinels — resolve against the User Guide and record them "
            f"in the column's null_values. Do not widen the type to make this pass."
        )
    log.info("parsed", file=path.name, rows=casted.height, columns=casted.width)
    return casted
