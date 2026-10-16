"""Target lookup transformations."""

import polars as pl


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
