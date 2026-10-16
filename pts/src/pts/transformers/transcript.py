"""Transcript index dataset generation.

Produces a per-transcript index by joining GeneCode GFF3 coordinates with
per-transcript UniProt IDs from the Ensembl parquet produced by pts_pre_target.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.transformers.utils.dataset import scan_dataset, write_dataset

INCLUDE_CHROMOSOMES = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']

GFF3_COLUMNS = [f'_c{i}' for i in range(9)]

#: Characters spark trims before casting a string to a number.
_SPARK_CAST_TRIM = ''.join(chr(c) for c in range(0x21)) + '\x7f'

#: Tag, label and value of each static flag, in output order.
_STATIC_FLAGS = [
    ('MANE_Select', 'mane', 'select'),
    ('MANE_Plus_Clinical', 'mane', 'plus_clinical'),
    ('GENCODE_Primary', 'gencode_primary', 'true'),
    ('Ensembl_canonical', 'ensembl_canonical', 'true'),
]

OUTPUT_COLUMNS = [
    'targetId',
    'transcriptId',
    'biotype',
    'proteinId',
    'uniprotSwissprotIds',
    'uniprotTremblIds',
    'uniprotIsoformIds',
    'alphafoldIds',
    'isEnsemblCanonical',
    'chromosome',
    'start',
    'end',
    'strand',
    'transcriptionStartSite',
    'flags',
    'exons',
]


def regexp_extract(col: pl.Expr, pattern: str) -> pl.Expr:
    """Group 1 of `pattern` in `col`, `''` when it does not match, as spark's `regexp_extract`."""
    return pl.when(col.is_not_null()).then(col.str.extract(pattern, 1).fill_null(''))


def to_long(col: pl.Expr) -> pl.Expr:
    """Cast a string to Int64 as spark's non-ANSI cast does."""
    trimmed = col.str.strip_chars(_SPARK_CAST_TRIM)
    whole = trimmed.str.extract(r'^([+-]?[0-9]*)(?:\.[0-9]*)?$', 1)
    return (
        pl
        .when(whole.str.contains('[0-9]'))
        .then(whole)
        .when(whole.is_not_null() & trimmed.str.contains('.', literal=True))
        .then(pl.lit('0'))
        .cast(pl.Int64, strict=False)
    )


def gff3_strand(col: pl.Expr) -> pl.Expr:
    """GFF3 `+`/`-` as Ensembl's 1/-1; anything else is null."""
    return pl.when(col == '+').then(pl.lit(1)).when(col == '-').then(pl.lit(-1)).cast(pl.Int32)


def chromosome(col: pl.Expr) -> pl.Expr:
    """The chromosome name without the `chr` prefix, with `M` as `MT`."""
    raw = regexp_extract(col, r'([0-9]{1,2}|X|Y|M)$')
    return pl.when(raw == 'M').then(pl.lit('MT')).otherwise(raw)


def read_gff3(path: str) -> pl.DataFrame:
    """Read the transcript and exon rows of a GFF3 file as nine string columns `_c0`..`_c8`.

    Args:
        path: the GFF3 file, optionally gzipped.

    Returns:
        DataFrame of the transcript and exon rows, in file order.
    """
    return (
        scan_dataset(
            path,
            format='tsv',
            has_header=False,
            comment_prefix='#',
            quote_char=None,
            infer_schema=False,
            new_columns=GFF3_COLUMNS,
        )
        .filter(pl.col('_c2').is_in(['transcript', 'exon']))
        .collect()
    )


def parse_gff3(df: pl.DataFrame) -> pl.DataFrame:
    """Parse transcript rows from a GFF3 DataFrame.

    Filters to transcript features, extracts identifiers from the attributes
    column, normalises chromosomes, derives transcriptionStartSite from strand,
    and builds the flags list.

    Args:
        df: raw GFF3 DataFrame with string columns `_c0`..`_c8`.

    Returns:
        DataFrame with one row per transcript on canonical chromosomes,
        containing targetId, transcriptId, biotype, proteinId, chromosome,
        start, end, strand, transcriptionStartSite, isEnsemblCanonical,
        and flags columns.
    """
    attrs = pl.col('_c8')
    tags = regexp_extract(attrs, r'tag=([^;]+)').str.split(',')
    protein_id = regexp_extract(attrs, r'protein_id=([^;.]+)')
    return (
        df
        .filter(pl.col('_c2') == 'transcript')
        .select(
            regexp_extract(attrs, r'gene_id=([^;.]+)').alias('targetId'),
            regexp_extract(attrs, r'transcript_id=([^;.]+)').alias('transcriptId'),
            regexp_extract(attrs, r'transcript_type=([^;]+)').alias('biotype'),
            pl.when(protein_id != '').then(protein_id).alias('proteinId'),
            to_long(pl.col('_c3')).alias('start'),
            to_long(pl.col('_c4')).alias('end'),
            gff3_strand(pl.col('_c6')).alias('strand'),
            tags.list.contains('Ensembl_canonical').fill_null(False).alias('isEnsemblCanonical'),
            build_flags(tags).alias('flags'),
            chromosome(pl.col('_c0')).alias('chromosome'),
        )
        .filter(
            pl.col('chromosome').is_in(INCLUDE_CHROMOSOMES)
            & pl.col('targetId').str.starts_with('ENSG')
            & pl.col('transcriptId').str.starts_with('ENST')
        )
        .with_columns(
            pl
            .when(pl.col('strand') == 1)
            .then(pl.col('start'))
            .otherwise(pl.col('end'))
            .cast(pl.Int32, wrap_numerical=True)
            .alias('transcriptionStartSite')
        )
    )


def build_flags(tags: pl.Expr) -> pl.Expr:
    """Build a structured flags list from a GFF3 tag list.

    Static flags (MANE, GENCODE Primary, Ensembl canonical) have fixed values.
    APPRIS flags have dynamic values extracted from the tag string itself
    (e.g. `appris_principal_1` -> `{label: "appris", value: "principal_1"}`).

    Args:
        tags: list of tag strings parsed from the GFF3 attributes field.

    Returns:
        List of `{label: str, value: str}` structs, the static flags first; null when `tags` is null.
    """
    static = pl.concat_list([
        pl.when(tags.list.contains(tag)).then(pl.struct(label=pl.lit(label), value=pl.lit(value)))
        for tag, label, value in _STATIC_FLAGS
    ]).list.drop_nulls()
    appris = tags.list.eval(
        pl
        .struct(label=pl.lit('appris'), value=pl.element().str.extract(r'^appris_(.*)', 1))
        .filter(pl.element().str.starts_with('appris_'))
    )
    return pl.when(tags.is_not_null()).then(pl.concat_list(static, appris))


def parse_exons(df: pl.DataFrame) -> pl.DataFrame:
    """Parse exon rows from a GFF3 DataFrame and group them by transcript.

    Filters to exon features, extracts identifiers and coordinates from the
    attributes column, applies the same chromosome normalisation as
    `parse_gff3`, and aggregates into a per-transcript exons list in file order.

    Args:
        df: raw GFF3 DataFrame with string columns `_c0`..`_c8`.

    Returns:
        DataFrame with columns `transcriptId` and `exons`
        (`list[struct[exonId, chromosome, start, end, strand]]`).
    """
    attrs = pl.col('_c8')
    return (
        df
        .filter(pl.col('_c2') == 'exon')
        .select(
            regexp_extract(attrs, r'transcript_id=([^;.]+)').alias('transcriptId'),
            regexp_extract(attrs, r'exon_id=([^;.]+)').alias('exonId'),
            chromosome(pl.col('_c0')).alias('chromosome'),
            to_long(pl.col('_c3')).alias('start'),
            to_long(pl.col('_c4')).alias('end'),
            gff3_strand(pl.col('_c6')).alias('strand'),
        )
        .filter(
            pl.col('chromosome').is_in(INCLUDE_CHROMOSOMES)
            & pl.col('transcriptId').str.starts_with('ENST')
            & (pl.col('exonId') != '')
        )
        .group_by('transcriptId', maintain_order=True)
        .agg(pl.struct('exonId', 'chromosome', 'start', 'end', 'strand').alias('exons'))
    )


def build_uniprot_lut(ensembl: pl.DataFrame) -> pl.DataFrame:
    """Build a per-transcript UniProt/AlphaFold lookup from the Ensembl parquet.

    Swiss-Prot (reviewed) and TrEMBL (unreviewed) are kept separate so consumers
    can derive provenance.

    Args:
        ensembl: Ensembl parquet from `intermediate/target/ensembl/homo_sapiens.parquet`.

    Returns:
        DataFrame with columns `targetId`, `transcriptId`, `uniprotSwissprotIds`,
        `uniprotTremblIds`, `uniprotIsoformIds` and `alphafoldIds`.
    """
    tx = pl.col('tx').struct
    return (
        ensembl
        .select(pl.col('id').alias('targetId'), pl.col('transcripts').alias('tx'))
        .filter(pl.col('tx').list.len() > 0)
        .explode('tx')
        .rechunk()
        .select(
            'targetId',
            tx.field('id').alias('transcriptId'),
            tx.field('uniprot_swissprot').alias('uniprotSwissprotIds'),
            tx.field('uniprot_trembl').alias('uniprotTremblIds'),
            tx.field('uniprot_isoform').alias('uniprotIsoformIds'),
            tx.field('alphafold').alias('alphafoldIds'),
        )
    )


def join_and_finalise(gff: pl.DataFrame, exons: pl.DataFrame, uniprot: pl.DataFrame) -> pl.DataFrame:
    """Join GFF3 transcript data with exons and per-transcript UniProt IDs.

    Args:
        gff: parsed transcripts from `parse_gff3`.
        exons: per-transcript exon lists from `parse_exons`.
        uniprot: per-transcript UniProt lookup from `build_uniprot_lut`.

    Returns:
        DataFrame with the final transcript index schema, sorted by gene and transcript.
    """
    return (
        gff
        .join(exons, on='transcriptId', how='left', maintain_order='left')
        .join(uniprot, on=['targetId', 'transcriptId'], how='left', maintain_order='left')
        .select(OUTPUT_COLUMNS)
        .sort('targetId', 'transcriptId', 'chromosome', 'start', 'end', maintain_order=True)
    )


def transcript(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Generate the transcript index dataset.

    Combines GeneCode GFF3 and Ensembl parquet to produce a per-transcript
    index with genomic coordinates, biotype, protein/UniProt identifiers, and
    transcript flags (MANE, APPRIS, Ensembl canonical, GENCODE Primary).

    Args:
        source: paths keyed by `gene_code` (GFF3, optionally gzipped) and `ensembl` (Ensembl parquet).
        destination: path of the transcript dataset.
        settings: unused.
        config: unused.
    """
    raw = read_gff3(source['gene_code'])
    ensembl = scan_dataset(source['ensembl']).select('id', 'transcripts').collect()

    result = join_and_finalise(parse_gff3(raw), parse_exons(raw), build_uniprot_lut(ensembl))

    logger.info(f'writing {result.height} transcripts to {destination}')
    write_dataset(result, destination)
