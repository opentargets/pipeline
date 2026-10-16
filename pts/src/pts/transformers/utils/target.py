"""Target lookups shared by transformers."""

from __future__ import annotations

import polars as pl


def ensg_lookup(target: pl.LazyFrame) -> pl.LazyFrame:
    """Map each protein accession and approved symbol of a target to its ENSG id.

    Args:
        target: the target dataset.

    Returns:
        One row per distinct (`ensgId`, `name`) pair.
    """
    proteins = target.select(
        pl.col('id').alias('ensgId'),
        pl.col('proteinIds').list.eval(pl.element().struct.field('id')).alias('name'),
    ).explode('name')
    symbols = target.select(pl.col('id').alias('ensgId'), pl.col('approvedSymbol').alias('name'))
    return pl.concat([proteins, symbols]).drop_nulls('name').unique(maintain_order=True)
