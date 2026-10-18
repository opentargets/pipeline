"""List helpers shared by the target builders."""

from __future__ import annotations

import polars as pl


def union(*exprs: pl.Expr) -> pl.Expr:
    """Concatenate lists, null as empty, keeping the first of each repeated element.

    A single list is only null-filled, not deduplicated.
    """
    filled = [e.fill_null([]) for e in exprs]
    if len(filled) == 1:
        return filled[0]
    return pl.concat_list(filled).list.unique(maintain_order=True)


def tag(expr: pl.Expr, field: str, source: str) -> pl.Expr:
    """Wrap each element of a string list as a `{field, source}` struct; null stays null."""
    return expr.list.eval(pl.struct(pl.element().alias(field), pl.lit(source).alias('source')))


def flatten_distinct(frame: pl.DataFrame, key: str, column: str) -> pl.DataFrame:
    """Per key, concatenate the distinct non-null lists of `column`, first occurrence first.

    Every key of `frame` is kept; a key whose lists are all null gets `[]`.
    """
    keys = frame.select(key).unique(maintain_order=True)
    flat = (
        frame
        .select(key, column)
        .drop_nulls(column)
        .unique(subset=[key, column], keep='first', maintain_order=True)
        .group_by(key, maintain_order=True)
        .agg(pl.col(column).explode(empty_as_null=False, keep_nulls=True))
    )
    return keys.join(flat, on=key, how='left', maintain_order='left', nulls_equal=True).with_columns(
        pl.col(column).fill_null([])
    )
