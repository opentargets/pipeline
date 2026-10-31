"""Frames shared by the variant step tests."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from pts.transformers.variant.colocated import ID_LISTS
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

VARIANTS_SCHEMA = pl.Schema({
    'row': pl.Int64,
    'chromosome': pl.String,
    'position': pl.Int32,
    'referenceAllele': pl.String,
    'alternateAllele': pl.String,
    'vepStart': pl.Int64,
    'vepEnd': pl.Int64,
    'vepReference': pl.String,
    'vepAlternate': pl.String,
    'hgvsg': pl.String,
    'most_severe_consequence': pl.String,
    'reported_transcripts': pl.Boolean,
    **ID_LISTS,
})
"""The columns of the variants frame, with the colocated id lists `add_colocated` adds."""


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


def with_effects(frame: pl.DataFrame) -> pl.DataFrame:
    """The consequences with every effect column of `EFFECTS_SCHEMA` they lack added as null."""
    return frame.with_columns(
        pl.lit(None, dtype).alias(name) for name, dtype in EFFECTS_SCHEMA.items() if name not in frame.columns
    )


def variants(**columns: Sequence[object]) -> pl.DataFrame:
    """A variants frame of the given columns, one list of values per column.

    Columns of `VARIANTS_SCHEMA` that are not given are null; the given ones are cast to their dtype.

    Raises:
        ValueError: on a column that is not in the schema.
    """
    unknown = sorted(set(columns) - set(VARIANTS_SCHEMA))
    if unknown:
        raise ValueError(f'not a variants column: {unknown}')
    height = max((len(values) for values in columns.values()), default=0)
    given = pl.DataFrame({
        name: pl.Series(name, values, dtype=VARIANTS_SCHEMA[name]) for name, values in columns.items()
    })
    return given.select(
        pl.col(name) if name in columns else pl.lit(None, dtype).alias(name) for name, dtype in VARIANTS_SCHEMA.items()
    ).head(height)
