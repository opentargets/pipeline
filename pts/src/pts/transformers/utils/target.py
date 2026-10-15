"""Target lookup transformations."""

import polars as pl


def tss_lut(target: pl.LazyFrame) -> pl.LazyFrame:
    """Resolve gene TSS from preferred, canonical transcript, then genomic coordinates."""
    canonical = pl.col('canonicalTranscript')
    genomic = pl.col('genomicLocation')
    canonical_tss = (
        pl
        .when(canonical.struct.field('strand') == '+')
        .then(canonical.struct.field('start'))
        .when(canonical.struct.field('strand') == '-')
        .then(canonical.struct.field('end'))
    )
    genomic_tss = (
        pl
        .when(genomic.struct.field('strand') == 1)
        .then(genomic.struct.field('start'))
        .when(genomic.struct.field('strand') == -1)
        .then(genomic.struct.field('end'))
    )
    return target.select(
        pl.col('id').alias('geneId'),
        pl.coalesce(pl.col('tss'), canonical_tss, genomic_tss).cast(pl.Int64).alias('tss'),
    )
