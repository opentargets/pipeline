"""GERP conservation and AlphaMissense filled into the VEP-shaped records."""

from __future__ import annotations

from typing import cast

import polars as pl

from pts.schemas.vep import vep_schema
from pts.transformers.variant.alleles import vep_allele_form
from pts.transformers.variant.alphamissense import alphamissense
from pts.transformers.variant.gerp import gerp_max

_TRANSCRIPT = cast(pl.Struct, cast(pl.List, vep_schema['transcript_consequences']).inner)
_ALPHAMISSENSE = _TRANSCRIPT.to_schema()['alphamissense']
_BASE = r'^[ACGT]$'


def _variants(records: pl.DataFrame) -> pl.DataFrame:
    """The VCF fields of each record's `input` line, with VEP's coordinates."""
    fields = pl.col('input').str.split('\t')
    return vep_allele_form(
        records.select(
            fields.list.get(1).cast(pl.Int64).alias('position'),
            fields.list.get(3).alias('referenceAllele'),
            fields.list.get(4).alias('alternateAllele'),
        ).lazy()
    ).collect()


def _conservation(variants: pl.DataFrame, bigwig: str, chromosome: str) -> pl.Series:
    """The GERP score of each variant to the three decimals VEP's plugin writes."""
    scores = gerp_max(bigwig, chromosome, variants)
    rounded = {score: float(f'{score:.3f}') for score in scores.drop_nulls().unique().to_list()}
    return pl.Series('conservation', [rounded.get(score) for score in scores], dtype=pl.Float64)


def add_gerp_and_alphamissense(
    records: pl.LazyFrame, bigwig: str, alphamissense_path: str, chromosome: str
) -> pl.LazyFrame:
    """Fill `conservation` and `alphamissense` of every transcript consequence of the records.

    GERP is the numeric maximum over the variant's span, rounded to three decimals. AlphaMissense
    is matched on the single-base substitution and the transcript id without its version.
    A contig the bigWig lacks, or a chromosome the AlphaMissense file has no rows for, leaves the
    field null.

    Args:
        records: The records of one chromosome, with VCF-form `input` lines.
        bigwig: The GERP bigWig.
        alphamissense_path: The `AlphaMissense_hg38.tsv.gz` file.
        chromosome: The chromosome of the records.

    Returns:
        The records, row for row, with the two fields filled.
    """
    frame = records.collect()
    variants = _variants(frame)
    transcripts = (
        frame
        .select(pl.col('transcript_consequences').alias('transcript'))
        .with_columns(
            pl.Series('row', range(frame.height), dtype=pl.UInt32),
            _conservation(variants, bigwig, chromosome),
            variants.get_column('position').alias('position'),
            variants.get_column('referenceAllele').alias('referenceAllele'),
            variants.get_column('alternateAllele').alias('alternateAllele'),
        )
        .filter(pl.col('transcript').list.len() > 0)
        .explode('transcript')
    )
    transcripts = transcripts.with_columns(pl.col('transcript').struct.field('transcript_id'))
    snv = pl.col('referenceAllele').str.contains(_BASE) & pl.col('alternateAllele').str.contains(_BASE)
    keys = transcripts.with_columns(
        pl.when(snv).then(pl.col('position')).alias('position'),
    )
    predictions = alphamissense(
        alphamissense_path, chromosome, keys.select('position', 'referenceAllele', 'alternateAllele', 'transcript_id')
    )
    filled = (
        transcripts
        .with_columns(predictions.select('am_class', 'am_pathogenicity'))
        .rechunk()
        .select(
            'row',
            pl.col('transcript').struct.with_fields(
                pl.col('conservation'),
                pl
                .when(pl.col('am_class').is_not_null())
                .then(pl.struct('am_class', 'am_pathogenicity'))
                .cast(_ALPHAMISSENSE)
                .alias('alphamissense'),
            ),
        )
        .group_by('row', maintain_order=True)
        .agg(pl.col('transcript').alias('filled'))
    )
    return (
        frame
        .with_row_index('row')
        .join(filled, on='row', how='left', maintain_order='left')
        .with_columns(pl.coalesce('filled', 'transcript_consequences').alias('transcript_consequences'))
        .drop('row', 'filled')
        .select(frame.columns)
        .lazy()
    )
