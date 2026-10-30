"""AlphaMissense predictions joined to transcript consequences."""

from __future__ import annotations

import polars as pl

KEY = ['position', 'referenceAllele', 'alternateAllele', 'transcript_id']
SCHEMA = pl.Schema({
    'chromosome': pl.String,
    'position': pl.Int64,
    'referenceAllele': pl.String,
    'alternateAllele': pl.String,
    'genome': pl.String,
    'uniprot_id': pl.String,
    'transcript_id': pl.String,
    'protein_variant': pl.String,
    'am_pathogenicity': pl.Float64,
    'am_class': pl.String,
})


def _predictions(path: str, chromosome: str) -> pl.DataFrame:
    """The AlphaMissense rows of one chromosome, with the transcript id stripped of its version."""
    return (
        pl
        .scan_csv(path, separator='\t', comment_prefix='#', has_header=False, schema=SCHEMA)
        .filter(pl.col('chromosome') == f'chr{chromosome}')
        .with_columns(pl.col('transcript_id').str.replace(r'\.\d+$', ''))
        .unique(KEY, keep='first', maintain_order=True)
        .select(*KEY, 'am_class', 'am_pathogenicity')
        .collect(engine='streaming')
    )


def alphamissense(path: str, chromosome: str, rows: pl.DataFrame) -> pl.DataFrame:
    """The AlphaMissense class and pathogenicity of each row.

    Predictions are matched on position, reference allele, alternate allele and the transcript
    id without its version, as the VEP plugin ran with `transcript_match=1`. The file's
    `#CHROM` carries a `chr` prefix, and its rows are single-base substitutions.

    Args:
        path: The `AlphaMissense_hg38.tsv.gz` file, local or remote.
        chromosome: The chromosome of `rows`, without the `chr` prefix.
        rows: One row per transcript consequence, with `position`, `referenceAllele`,
            `alternateAllele` and the unversioned `transcript_id`.

    Returns:
        `rows` in the same order with `am_class` and `am_pathogenicity`, null where the file has
        no prediction, including when it has no row for the chromosome.
    """
    return rows.join(_predictions(path, chromosome), on=KEY, how='left', maintain_order='left')
