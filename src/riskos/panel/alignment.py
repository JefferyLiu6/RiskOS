"""Fail before positional model outputs can be paired with the wrong observations."""

from __future__ import annotations

import polars as pl

KEYS = ["loan_sequence_number", "observation_date"]


def require_same_observations(before: pl.DataFrame, after: pl.DataFrame) -> None:
    """Joins may enrich a frame, but must not duplicate, drop, or reorder its keys."""
    original = before.select(KEYS)
    if original.null_count().sum_horizontal().sum() or original.is_duplicated().any():
        raise ValueError("observation keys must be non-null and unique")
    if not original.equals(after.select(KEYS)):
        raise ValueError(
            "observation alignment changed: rows were reordered, dropped, or duplicated"
        )
