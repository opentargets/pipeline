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


def tss_lut(target: pl.LazyFrame) -> pl.LazyFrame:
    """Resolve gene TSS from preferred, canonical transcript, then genomic coordinates."""
    genomic = pl.col('genomicLocation')
    genomic_tss = (
        pl
        .when(genomic.struct.field('strand') == 1)
        .then(genomic.struct.field('start'))
        .when(genomic.struct.field('strand') == -1)
        .then(genomic.struct.field('end'))
    )
    candidates = [pl.col('tss')]
    if 'canonicalTranscript' in target.collect_schema():
        canonical = pl.col('canonicalTranscript')
        canonical_tss = (
            pl
            .when(canonical.struct.field('strand') == '+')
            .then(canonical.struct.field('start'))
            .when(canonical.struct.field('strand') == '-')
            .then(canonical.struct.field('end'))
        )
        candidates.append(canonical_tss)
    return target.select(
        pl.col('id').alias('geneId'),
        pl.coalesce(*candidates, genomic_tss).cast(pl.Int64).alias('tss'),
    )
