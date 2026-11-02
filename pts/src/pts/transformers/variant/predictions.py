"""GERP conservation and AlphaMissense as columns of the consequences frame."""

from __future__ import annotations

import polars as pl

from pts.transformers.variant.alphamissense import alphamissense
from pts.transformers.variant.gerp import gerp_max

_BASE = r'^[ACGT]$'


def _conservation(spans: pl.DataFrame, bigwig: str, chromosome: str) -> pl.Series:
    """The GERP score of each span to the three decimals VEP's plugin writes."""
    scores = gerp_max(bigwig, chromosome, spans)
    rounded = {score: float(f'{score:.3f}') for score in scores.drop_nulls().unique().to_list()}
    return pl.Series('conservation', [rounded.get(score) for score in scores], dtype=pl.Float64)


def add_gerp_and_alphamissense(
    consequences: pl.DataFrame, bigwig: str, alphamissense_path: str, chromosome: str
) -> pl.DataFrame:
    """Add `conservation`, `am_class` and `am_pathogenicity` to the consequences.

    GERP is the numeric maximum over the variant's span, rounded to three decimals, read once per
    variant. AlphaMissense is matched on the single-base substitution of the original alleles and
    the transcript id without its version. A contig the bigWig lacks, or a chromosome the
    AlphaMissense file has no rows for, leaves the columns null. Without consequences neither file
    is read.

    Args:
        consequences: The consequences frame of one chromosome.
        bigwig: The GERP bigWig.
        alphamissense_path: The `AlphaMissense_hg38.tsv.gz` file.
        chromosome: The chromosome of the consequences.

    Returns:
        The consequences, row for row, with the three columns added.
    """
    if consequences.is_empty():
        return consequences.with_columns(
            pl.lit(None, pl.Float64).alias('conservation'),
            pl.lit(None, pl.String).alias('am_class'),
            pl.lit(None, pl.Float64).alias('am_pathogenicity'),
        )
    variants = consequences.unique('row', maintain_order=True)
    conservation = variants.select('row').with_columns(_conservation(variants, bigwig, chromosome))
    snv = pl.col('referenceAllele').str.contains(_BASE) & pl.col('alternateAllele').str.contains(_BASE)
    keys = consequences.select(
        pl.when(snv).then(pl.col('position').cast(pl.Int64)).alias('position'),
        'referenceAllele',
        'alternateAllele',
        'transcript_id',
    )
    predictions = alphamissense(alphamissense_path, chromosome, keys)
    attached = consequences.select('row').join(conservation, on='row', how='left', maintain_order='left')
    return consequences.with_columns(
        attached['conservation'], *predictions.select('am_class', 'am_pathogenicity').get_columns()
    )
