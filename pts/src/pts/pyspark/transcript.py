"""Transcript index dataset generation.

Produces a per-transcript index by joining GeneCode GFF3 coordinates with
per-transcript UniProt IDs from the Ensembl parquet produced by pts_pre_target.
"""

from __future__ import annotations

from typing import Any

import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import Column, DataFrame
from pyspark.sql.types import IntegerType, LongType

from pts.pyspark.common.session import Session
from pts.pyspark.common.utils import maybe_coalesce, safe_array_union

INCLUDE_CHROMOSOMES = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']


def transcript(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Generate the transcript index dataset.

    Combines GeneCode GFF3 and Ensembl parquet to produce a per-transcript
    index with genomic coordinates, biotype, protein/UniProt identifiers, and
    transcript flags (MANE, APPRIS, Ensembl canonical, GENCODE Primary).

    Args:
        source: Mapping of logical input names to paths. Expected keys:
            ``gene_code`` (GFF3 gz) and ``ensembl`` (Ensembl parquet).
        destination: Output path for the transcript parquet dataset.
        settings: Step settings; supports ``partition_count`` (int, default 2).
        properties: Spark session properties passed to :class:`Session`.
    """
    spark = Session(app_name='transcript', properties=properties).spark

    gene_code_raw = spark.read.option('sep', '\t').option('comment', '#').csv(source['gene_code'])
    ensembl_raw = spark.read.parquet(source['ensembl'])

    gff = _parse_gff3(gene_code_raw)
    exon_lut = _parse_exons(gene_code_raw)
    uniprot_lut = _build_uniprot_lut(ensembl_raw)
    result = _join_and_finalise(gff, exon_lut, uniprot_lut)

    partition_count = (settings or {}).get('partition_count', 2)
    logger.info(f'writing transcript index to {destination} ({partition_count} partitions)')
    maybe_coalesce(result, partition_count).write.mode('overwrite').parquet(destination)


def _parse_gff3(df: DataFrame) -> DataFrame:
    """Parse transcript rows from a GFF3 DataFrame.

    Filters to transcript features, extracts identifiers from the attributes
    column, normalises chromosomes, derives transcriptionStartSite from strand,
    and builds the flags array.

    Args:
        df: Raw GFF3 DataFrame (tab-separated, comment lines skipped).

    Returns:
        DataFrame with one row per transcript on canonical chromosomes,
        containing targetId, transcriptId, biotype, proteinId, chromosome,
        start, end, strand, transcriptionStartSite, isEnsemblCanonical,
        and flags columns.
    """
    attrs = f.col('_c8')
    tags = f.split(f.regexp_extract(attrs, r'tag=([^;]+)', 1), ',')

    return (
        df
        .filter(f.col('_c2') == 'transcript')
        .select(
            # Strip version suffix by stopping at the first dot
            f.regexp_extract(attrs, r'gene_id=([^;.]+)', 1).alias('targetId'),
            f.regexp_extract(attrs, r'transcript_id=([^;.]+)', 1).alias('transcriptId'),
            f.regexp_extract(attrs, r'transcript_type=([^;]+)', 1).alias('biotype'),
            # protein_id absent for non-coding transcripts → empty string
            f.regexp_extract(attrs, r'protein_id=([^;.]+)', 1).alias('_proteinId_raw'),
            # Extract chromosome number/letter (strips chr-prefix; M handled below)
            f.regexp_extract(f.col('_c0'), r'([0-9]{1,2}|X|Y|M)$', 1).alias('_chrom_raw'),
            f.col('_c3').cast(LongType()).alias('start'),
            f.col('_c4').cast(LongType()).alias('end'),
            f.col('_c6').alias('strand'),
            f.coalesce(f.array_contains(tags, 'Ensembl_canonical'), f.lit(False)).alias('isEnsemblCanonical'),
            tags.alias('_tags'),
        )
        .withColumn('chromosome',
            f.when(f.col('_chrom_raw') == 'M', 'MT').otherwise(f.col('_chrom_raw'))
        )
        .filter(
            f.col('chromosome').isin(INCLUDE_CHROMOSOMES)
            & f.col('targetId').startswith('ENSG')
            & f.col('transcriptId').startswith('ENST')
        )
        .withColumn('proteinId',
            f.when(f.col('_proteinId_raw') != '', f.col('_proteinId_raw'))
        )
        .withColumn('transcriptionStartSite',
            f.when(f.col('strand') == '+', f.col('start'))
            .otherwise(f.col('end'))
            .cast(IntegerType())
        )
        .withColumn('flags', _build_flags(f.col('_tags')))
        .drop('_chrom_raw', '_proteinId_raw', '_tags')
    )


def _build_flags(tags_col: Column) -> Column:
    """Build a structured flags array from a GFF3 tag array column.

    Static flags (MANE, GENCODE Primary, Ensembl canonical) have fixed values.
    APPRIS flags have dynamic values extracted from the tag string itself
    (e.g. ``appris_principal_1`` → ``{label: "appris", value: "principal_1"}``).

    Args:
        tags_col: Array column of tag strings parsed from the GFF3 attributes field.

    Returns:
        Array column of ``{label: str, value: str}`` structs.
    """
    static = f.array_compact(f.array(
        f.when(
            f.array_contains(tags_col, 'MANE_Select'),
            f.struct(f.lit('mane').alias('label'), f.lit('select').alias('value')),
        ),
        f.when(
            f.array_contains(tags_col, 'MANE_Plus_Clinical'),
            f.struct(f.lit('mane').alias('label'), f.lit('plus_clinical').alias('value')),
        ),
        f.when(
            f.array_contains(tags_col, 'GENCODE_Primary'),
            f.struct(f.lit('gencode_primary').alias('label'), f.lit('true').alias('value')),
        ),
        f.when(
            f.array_contains(tags_col, 'Ensembl_canonical'),
            f.struct(f.lit('ensembl_canonical').alias('label'), f.lit('true').alias('value')),
        ),
    ))

    # APPRIS tags carry the tier in the tag name itself: appris_principal_1, appris_alternative_2, ...
    appris = f.transform(
        f.filter(tags_col, lambda t: t.startswith('appris_')),
        lambda t: f.struct(
            f.lit('appris').alias('label'),
            f.regexp_extract(t, r'^appris_(.*)', 1).alias('value'),
        ),
    )

    return f.concat(static, appris)


def _parse_exons(df: DataFrame) -> DataFrame:
    """Parse exon rows from a GFF3 DataFrame and group them by transcript.

    Filters to exon features, extracts identifiers and coordinates from the
    attributes column, applies the same chromosome normalisation as
    :func:`_parse_gff3`, and aggregates into a per-transcript exons array.

    Args:
        df: Raw GFF3 DataFrame (tab-separated, comment lines skipped).

    Returns:
        DataFrame with columns ``transcriptId`` and ``exons``
        (``array<struct<exonId, chromosome, start, end, strand>>``).
    """
    attrs = f.col('_c8')

    exons = (
        df
        .filter(f.col('_c2') == 'exon')
        .select(
            f.regexp_extract(attrs, r'transcript_id=([^;.]+)', 1).alias('transcriptId'),
            f.regexp_extract(attrs, r'exon_id=([^;.]+)', 1).alias('exonId'),
            f.regexp_extract(f.col('_c0'), r'([0-9]{1,2}|X|Y|M)$', 1).alias('_chrom_raw'),
            f.col('_c3').cast(LongType()).alias('start'),
            f.col('_c4').cast(LongType()).alias('end'),
            f.col('_c6').alias('strand'),
        )
        .withColumn('chromosome',
            f.when(f.col('_chrom_raw') == 'M', 'MT').otherwise(f.col('_chrom_raw'))
        )
        .filter(
            f.col('chromosome').isin(INCLUDE_CHROMOSOMES)
            & f.col('transcriptId').startswith('ENST')
            & (f.col('exonId') != '')
        )
        .drop('_chrom_raw')
    )

    return (
        exons
        .groupBy('transcriptId')
        .agg(
            f.collect_list(
                f.struct(
                    f.col('exonId'),
                    f.col('chromosome'),
                    f.col('start'),
                    f.col('end'),
                    f.col('strand'),
                )
            ).alias('exons')
        )
    )


def _build_uniprot_lut(ensembl_df: DataFrame) -> DataFrame:
    """Build a (targetId, transcriptId) → uniprotIds lookup from the Ensembl parquet.

    The Ensembl parquet (produced by pts_pre_target) stores per-transcript
    uniprot_swissprot and uniprot_trembl arrays. Explodes the transcripts array
    and merges both into a single deduplicated uniprotIds list.

    Args:
        ensembl_df: Ensembl parquet from ``intermediate/target/ensembl/homo_sapiens.parquet``.

    Returns:
        DataFrame with columns ``targetId``, ``transcriptId``, and ``uniprotIds``.
    """
    return (
        ensembl_df
        .select(f.col('id').alias('targetId'), f.explode('transcripts').alias('tx'))
        .select(
            f.col('targetId'),
            f.col('tx.id').alias('transcriptId'),
            f.col('tx.uniprot_swissprot').alias('_swissprot'),
            f.col('tx.uniprot_trembl').alias('_trembl'),
        )
        .withColumn('uniprotIds',
            f.array_distinct(safe_array_union(f.col('_swissprot'), f.col('_trembl')))
        )
        .select('targetId', 'transcriptId', 'uniprotIds')
    )


def _join_and_finalise(gff: DataFrame, exon_lut: DataFrame, uniprot_lut: DataFrame) -> DataFrame:
    """Join GFF3 transcript data with exons and per-transcript UniProt IDs.

    Args:
        gff: Parsed GFF3 DataFrame from :func:`_parse_gff3`.
        exon_lut: Per-transcript exon arrays from :func:`_parse_exons`.
        uniprot_lut: Per-transcript UniProt lookup from :func:`_build_uniprot_lut`.

    Returns:
        DataFrame with the final transcript index schema.
    """
    return (
        gff
        .join(exon_lut, on='transcriptId', how='left')
        .join(uniprot_lut, on=['targetId', 'transcriptId'], how='left')
        .select(
            'targetId',
            'transcriptId',
            'biotype',
            'proteinId',
            'uniprotIds',
            'isEnsemblCanonical',
            'chromosome',
            'start',
            'end',
            'strand',
            'transcriptionStartSite',
            'flags',
            'exons',
        )
    )
