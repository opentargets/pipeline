"""AlphaMissense predictions joined to transcript consequences."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

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


def _contig(chromosome: str) -> str:
    """The file's `#CHROM` for an index chromosome: `chr` and the name, `chrM` for MT."""
    return 'chrM' if chromosome == 'MT' else f'chr{chromosome}'


def _scan(path: str) -> pl.LazyFrame:
    """The AlphaMissense file, or one chromosome of it written by `partition_alphamissense`."""
    if path.endswith('.parquet'):
        return pl.scan_parquet(path)
    return pl.scan_csv(path, separator='\t', comment_prefix='#', has_header=False, schema=SCHEMA)


def partition_alphamissense(path: str, directory: Path, chromosomes: Iterable[str]) -> dict[str, str]:
    """The AlphaMissense rows of each chromosome, read in one pass, as `directory/<chromosome>.parquet`.

    Every chromosome gets a file, without rows when the source has none for it.

    Returns:
        Chromosome to the path of its file, which `alphamissense` reads in place of the source.
    """
    directory.mkdir(parents=True, exist_ok=True)
    names = {chromosome: _contig(chromosome) for chromosome in chromosomes}
    frame = (
        _scan(path)
        .filter(pl.col('chromosome').is_in(list(names.values())))
        .select('chromosome', *KEY, 'am_pathogenicity', 'am_class')
        .collect(engine='streaming')
    )
    parts = {}
    for chromosome, name in names.items():
        parts[chromosome] = str(directory / f'{chromosome}.parquet')
        frame.filter(pl.col('chromosome') == name).write_parquet(parts[chromosome])
    return parts


def _predictions(path: str, chromosome: str) -> pl.DataFrame:
    """The AlphaMissense rows of one chromosome, with the transcript id stripped of its version."""
    return (
        _scan(path)
        .filter(pl.col('chromosome') == _contig(chromosome))
        .with_columns(pl.col('transcript_id').str.replace(r'\.\d+$', ''))
        .unique(KEY, keep='first', maintain_order=True)
        .select(*KEY, 'am_class', 'am_pathogenicity')
        .collect(engine='streaming')
    )


def alphamissense(path: str, chromosome: str, rows: pl.DataFrame) -> pl.DataFrame:
    """The AlphaMissense class and pathogenicity of each row.

    Predictions are matched on position, reference allele, alternate allele and the transcript
    id without its version, as the VEP plugin ran with `transcript_match=1`. The file's
    `#CHROM` carries a `chr` prefix (`chrM` for MT), and its rows are single-base substitutions.

    Args:
        path: The `AlphaMissense_hg38.tsv.gz` file, local or remote, or its `partition_alphamissense` part.
        chromosome: The chromosome of `rows`, without the `chr` prefix.
        rows: One row per transcript consequence, with `position`, `referenceAllele`,
            `alternateAllele` and the unversioned `transcript_id`.

    Returns:
        `rows` in the same order with `am_class` and `am_pathogenicity`, null where the file has
        no prediction, including when it has no row for the chromosome.
    """
    return rows.join(_predictions(path, chromosome), on=KEY, how='left', maintain_order='left')
