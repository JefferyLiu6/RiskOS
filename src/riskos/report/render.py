"""Markdown rendering helpers. Formatting only; no numbers originate here."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import polars as pl

Formatter = Callable[[Any], str]


def money(value: float | None, dp: int | None = None) -> str:
    """Dollar amount with a magnitude suffix: $34.0M, $15.91B."""
    if value is None:
        return "—"
    v = float(value)
    if abs(v) >= 1e9:
        return f"${v / 1e9:,.{2 if dp is None else dp}f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:,.{1 if dp is None else dp}f}M"
    return f"${v:,.0f}"


def pct(value: float | None, dp: int = 2) -> str:
    """A fraction rendered as a percentage."""
    return "—" if value is None else f"{100.0 * float(value):.{dp}f}%"


def num(value: float | int | None, dp: int = 0) -> str:
    if value is None:
        return "—"
    return f"{float(value):,.{dp}f}" if dp else f"{round(float(value)):,}"


def ratio(value: float | None, dp: int = 2) -> str:
    return "—" if value is None else f"{float(value):.{dp}f}"


def sci(value: float | None) -> str:
    return "—" if value is None else f"{float(value):.2e}"


def text(value: Any) -> str:
    return "—" if value is None else str(value)


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """A GitHub-flavoured Markdown table. Cells are rendered with str()."""
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def df_table(
    frame: pl.DataFrame,
    columns: Sequence[tuple[str, str, Formatter]],
) -> str:
    """Render selected columns of a frame: (column, header, formatter) triples."""
    headers = [header for _, header, _ in columns]
    rows = [[fmt(row.get(col)) for col, _, fmt in columns] for row in frame.to_dicts()]
    return md_table(headers, rows)


def missing(artefact: str, command: str) -> str:
    return f"> **Artefact missing:** `{artefact}` was not found. Run `{command}` to produce it."


def squash(value: Any) -> str:
    """Collapse a folded YAML block into one line for a table cell."""
    return " ".join(str(value).split())


def section(title: str, *blocks: str) -> str:
    body = "\n\n".join(b for b in blocks if b)
    return f"## {title}\n\n{body}\n"
