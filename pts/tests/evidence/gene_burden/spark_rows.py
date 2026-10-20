"""Rows of the removed Spark parsers, captured as literals, and the new polars parsers in one comparable form."""

from __future__ import annotations

from typing import Any


def _canonical(value: Any) -> Any:
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        return tuple(_canonical(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _canonical(v)) for k, v in value.items()))
    return value


def multiset(rows: list[dict[str, Any]], columns: list[str]) -> list[tuple[Any, ...]]:
    """The rows' `columns` as sorted tuples: floats by repr, structs by field name."""
    return sorted((tuple(_canonical(row[c]) for c in columns) for row in rows), key=repr)


def expand(constant: dict[str, Any], columns: list[str], rows: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
    """Each row of `columns` values as a dict, with the `constant` columns added."""
    return [constant | dict(zip(columns, row, strict=True)) for row in rows]
