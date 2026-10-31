"""Frames shared by the variant step tests."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from pts.transformers.variant.records import CONSEQUENCES_SCHEMA

EFFECTS_SCHEMA = pl.Schema({
    'conservation': pl.Float64,
    'am_class': pl.String,
    'am_pathogenicity': pl.Float64,
    'lof': pl.String,
    'lof_filter': pl.String,
    'sift_score': pl.Float64,
    'sift_prediction': pl.String,
})
"""The columns the effects add to the consequences frame."""


def consequences(**columns: Sequence[object]) -> pl.DataFrame:
    """A consequences frame of the given columns, one list of values per column.

    Columns of `CONSEQUENCES_SCHEMA` that are not given are null; the given ones are cast to
    their dtype. The effect columns of `EFFECTS_SCHEMA` are added after them, in the order given,
    only when given.

    Raises:
        ValueError: on a column that is in neither schema.
    """
    schema = CONSEQUENCES_SCHEMA | EFFECTS_SCHEMA
    unknown = sorted(set(columns) - set(schema))
    if unknown:
        raise ValueError(f'not a consequences column: {unknown}')
    height = max((len(values) for values in columns.values()), default=0)
    given = pl.DataFrame({name: pl.Series(name, values, dtype=schema[name]) for name, values in columns.items()})
    return given.select(
        *[
            pl.col(name) if name in columns else pl.lit(None, dtype).alias(name)
            for name, dtype in CONSEQUENCES_SCHEMA.items()
        ],
        *[pl.col(name) for name in EFFECTS_SCHEMA if name in columns],
    ).head(height)
