"""Compare a spark frame from `pts.pyspark.target` with its polars port, as python rows."""

from __future__ import annotations

import json
from typing import Any

import polars as pl
from pyspark.sql import DataFrame


def _plain(value: Any) -> Any:
    if hasattr(value, 'asDict'):
        return {k: _plain(v) for k, v in value.asDict().items()}
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def _key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def rows(
    df: DataFrame | pl.DataFrame,
    key: str,
    unordered: tuple[str, ...] = (),
    empty_as_null: tuple[str, ...] = (),
) -> list[dict]:
    """Rows sorted by `key`; `unordered` lists sorted; `empty_as_null` lists `[]` read as null.

    A dotted name in `unordered` sorts a list inside a struct column, e.g. `hallmarks.attributes`.
    """
    plain = [_plain(r.asDict()) for r in df.collect()] if isinstance(df, DataFrame) else df.to_dicts()
    for row in plain:
        for column in empty_as_null:
            if row[column] == []:
                row[column] = None
        for column in unordered:
            outer, _, inner = column.partition('.')
            holder, name = (row[outer], inner) if inner else (row, outer)
            if holder is not None and holder[name] is not None:
                holder[name] = sorted(holder[name], key=_key)
    return sorted(plain, key=lambda r: _key(r[key]))
