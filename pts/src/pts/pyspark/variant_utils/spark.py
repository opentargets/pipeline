"""Column helpers for the variant index, ported from Gentropy's `gentropy.common.spark`."""

from __future__ import annotations

from itertools import chain
from typing import Any

from pyspark.sql import Column
from pyspark.sql import functions as f


def map_column_by_dictionary(column: Column, mapping: dict[str, Any]) -> Column:
    """Look a column's values up in a literal map; values missing from it become null."""
    return f.create_map(*[f.lit(item) for item in chain(*mapping.items())])[column]


def order_structs_by_field(array: str, field: str) -> Column:
    """Sort an array of structs by one field, descending with nulls last; ties keep their order."""
    return f.expr(
        f"""
        array_sort({array}, (left, right) -> case
            when left.{field} is null and right.{field} is null then 0
            when left.{field} is null then 1
            when right.{field} is null then -1
            when left.{field} < right.{field} then 1
            when left.{field} > right.{field} then -1
            else 0
        end)
        """
    )


def order_structs_by_two_fields(array: str, descending: str, ascending: str) -> Column:
    """Sort an array of structs by one field descending, then another ascending; nulls last."""
    return f.expr(
        f"""
        array_sort({array}, (left, right) -> case
            when left.{descending} is null and right.{descending} is null then 0
            when left.{ascending} is null and right.{ascending} is null then 0
            when left.{descending} is null then 1
            when right.{descending} is null then -1
            when left.{ascending} is null then 1
            when right.{ascending} is null then -1
            when left.{descending} < right.{descending} then 1
            when left.{descending} > right.{descending} then -1
            when left.{ascending} > right.{ascending} then 1
            when left.{ascending} < right.{ascending} then -1
            else 0
        end)
        """
    )


def safe_array_union(left: str, right: str, fields: list[str] | None = None) -> Column:
    """Union two array columns, keeping one side when the other is null.

    For arrays of structs, `fields` gives the struct field order; both sides are rebuilt in
    that order and sorted so equal structs deduplicate regardless of their source layout.
    """
    a, b = f.col(left), f.col(right)
    if fields:
        expression = ', '.join(f'x.{field}' for field in fields)
        a = f.expr(f'sort_array(transform(`{left}`, x -> struct({expression})), false)')
        b = f.expr(f'sort_array(transform(`{right}`, x -> struct({expression})), false)')
    return f.when(a.isNotNull() & b.isNotNull(), f.array_union(a, b)).otherwise(f.coalesce(a, b))
