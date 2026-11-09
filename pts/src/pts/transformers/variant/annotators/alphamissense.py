"""AlphaMissense predictions joined to transcript consequences."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import polars as pl

KEY = ['position', 'referenceAllele', 'alternateAllele', 'transcript_id']
ROW_GROUP_ROWS = 50_000
_BASE = r'^[ACGT]$'
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
ISOFORMS_SCHEMA = pl.Schema({name: dtype for name, dtype in SCHEMA.items() if name != 'uniprot_id'})


def _contig(chromosome: str) -> str:
    """The file's `#CHROM` for an index chromosome: `chr` and the name, `chrM` for MT."""
    return 'chrM' if chromosome == 'MT' else f'chr{chromosome}'


def _scan(path: str, schema: pl.Schema = SCHEMA) -> pl.LazyFrame:
    """An AlphaMissense file with `schema`, or one chromosome of it written by `partition_alphamissense`."""
    if path.endswith('.parquet'):
        return pl.scan_parquet(path)
    return pl.scan_csv(path, separator='\t', comment_prefix='#', has_header=False, schema=schema)


def partition_alphamissense(
    canonical: str,
    isoforms: str,
    directory: Path,
    chromosomes: Iterable[str],
    row_group_rows: int = ROW_GROUP_ROWS,
) -> dict[str, str]:
    """The AlphaMissense rows of each chromosome, from both files, as `directory/<chromosome>.parquet`.

    `AlphaMissense_hg38.tsv.gz` predicts the canonical isoform of each UniProt protein and
    `AlphaMissense_isoforms_hg38.tsv.gz` the other GENCODE V32 coding transcripts. Where both
    predict a key, the canonical file's row comes first, and `alphamissense` keeps the first.
    Both are read once into `directory/staged.parquet`, which is removed once every chromosome is
    written. Every chromosome gets a file, without rows when neither source has any for it, sorted
    by position in row groups of `row_group_rows`, so that a position range reads only its own.

    Returns:
        Chromosome to the path of its file, which `alphamissense` reads in place of the sources.
    """
    directory.mkdir(parents=True, exist_ok=True)
    names = {chromosome: _contig(chromosome) for chromosome in chromosomes}
    staged = directory / 'staged.parquet'
    columns = ['chromosome', *KEY, 'am_pathogenicity', 'am_class']
    pl.concat([_scan(canonical).select(columns), _scan(isoforms, ISOFORMS_SCHEMA).select(columns)]).filter(
        pl.col('chromosome').is_in(list(names.values()))
    ).sink_parquet(staged, row_group_size=row_group_rows)
    parts = {}
    for chromosome, name in names.items():
        parts[chromosome] = str(directory / f'{chromosome}.parquet')
        (
            pl
            .scan_parquet(staged)
            .filter(pl.col('chromosome') == name)
            .collect()
            .sort('position', maintain_order=True)
            .write_parquet(parts[chromosome], row_group_size=row_group_rows)
        )
    staged.unlink()
    return parts


def _predictions(path: str, chromosome: str, positions: pl.Series) -> pl.DataFrame:
    """The AlphaMissense rows of one chromosome between the least and greatest of `positions`.

    The transcript id is stripped of its version.
    """
    return (
        _scan(path)
        .filter(pl.col('chromosome') == _contig(chromosome))
        .filter(pl.col('position').is_between(positions.min(), positions.max()))
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
        path: The chromosome's `partition_alphamissense` part, or an `AlphaMissense_hg38.tsv.gz` file.
        chromosome: The chromosome of `rows`, without the `chr` prefix.
        rows: One row per transcript consequence, with `position`, `referenceAllele`,
            `alternateAllele` and the unversioned `transcript_id`.

    Returns:
        `rows` in the same order with `am_class` and `am_pathogenicity`, null where the file has
        no prediction, including when it has no row for the chromosome.
    """
    return rows.join(_predictions(path, chromosome, rows['position']), on=KEY, how='left', maintain_order='left')


def add_alphamissense(consequences: pl.DataFrame, path: str, chromosome: str) -> pl.DataFrame:
    """Add `am_class` and `am_pathogenicity` to the consequences.

    AlphaMissense is matched on the single-base substitution of the original alleles and the
    transcript id without its version. A chromosome the file has no rows for leaves the columns
    null. Without consequences the file is not read.

    Args:
        consequences: The consequences frame of one chromosome.
        path: The chromosome's `partition_alphamissense` part, or an `AlphaMissense_hg38.tsv.gz` file.
        chromosome: The chromosome of the consequences.

    Returns:
        The consequences, row for row, with the two columns added.
    """
    if consequences.is_empty():
        return consequences.with_columns(
            pl.lit(None, pl.String).alias('am_class'),
            pl.lit(None, pl.Float64).alias('am_pathogenicity'),
        )
    snv = pl.col('referenceAllele').str.contains(_BASE) & pl.col('alternateAllele').str.contains(_BASE)
    keys = consequences.select(
        pl.when(snv).then(pl.col('position').cast(pl.Int64)).alias('position'),
        'referenceAllele',
        'alternateAllele',
        'transcript_id',
    )
    predictions = alphamissense(path, chromosome, keys)
    return consequences.with_columns(*predictions.select('am_class', 'am_pathogenicity').get_columns())
