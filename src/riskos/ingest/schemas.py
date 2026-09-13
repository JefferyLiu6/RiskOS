"""Pandera schemas, built from the declared layout.

Schema-as-code that fails loudly in-pipeline (build plan §5). The schema is
generated from the same ``ColumnSpec`` list that drives parsing, so the two
cannot drift apart.
"""

from __future__ import annotations

import polars as pl
from pandera.polars import Column, DataFrameSchema

from riskos.config import ColumnSpec
from riskos.log import get_logger

log = get_logger(__name__)

_PANDERA_DTYPE: dict[str, pl.DataType] = {
    "str": pl.String(),
    "int": pl.Int64(),
    "float": pl.Float64(),
    "date": pl.Date(),
}


def build_schema(
    columns: tuple[ColumnSpec, ...],
    *,
    unique: tuple[str, ...] = (),
    non_null: tuple[str, ...] = (),
) -> DataFrameSchema:
    """Build a strict schema: declared columns, declared types, nothing extra.

    ``unique`` and ``non_null`` name columns whose key/completeness properties
    are asserted — e.g. loan sequence number on the origination file.
    """
    declared = {c.name for c in columns}
    for name in (*unique, *non_null):
        if name not in declared:
            raise KeyError(f"{name!r} is not in the declared layout")

    return DataFrameSchema(
        columns={
            spec.name: Column(
                _PANDERA_DTYPE[spec.dtype],
                nullable=spec.name not in non_null,
                unique=spec.name in unique,
                description=spec.description or None,
                required=True,
            )
            for spec in columns
        },
        strict=True,  # reject undeclared columns
        ordered=True,  # positional layout — order is part of the contract
        coerce=False,  # parse.py already cast; a mismatch here is a real defect
        name="sflld",
    )


def validate(df: pl.DataFrame, schema: DataFrameSchema, *, label: str) -> pl.DataFrame:
    """Validate, logging the outcome. Pandera raises on failure."""
    validated = schema.validate(df, lazy=True)
    log.info("schema_validated", label=label, rows=df.height, columns=df.width)
    return validated
