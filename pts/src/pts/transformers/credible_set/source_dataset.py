"""Read the credible sets or credible set variants of one source, as the credible set stages expect them."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from pts.transformers.utils.dataset import scan_dataset


def read_source_dataset(path: str, schema: pl.Schema, required: Sequence[str]) -> pl.LazyFrame:
    """Read one source's dataset in `schema`, with no null in the `required` columns and `qualityControls` a list.

    A column of `schema` absent from the dataset raises in `match_to_schema`. A null in a `required`
    column means the source breaks its contract, so the stage stops before any check runs; a null
    `qualityControls` becomes an empty list, so every check can rely on it.

    Raises:
        ValueError: if a `required` column has nulls, naming each such column and its count.
    """
    frame = (
        scan_dataset(path)
        .match_to_schema(schema)
        .with_columns(pl.col('qualityControls').fill_null(pl.lit([], dtype=schema['qualityControls'])))
    )
    null_count = frame.select(pl.col(*required).null_count()).collect().row(0, named=True)
    if missing := {column: count for column, count in null_count.items() if count}:
        msg = f'{path}: required columns have nulls: {missing}'
        raise ValueError(msg)
    return frame
