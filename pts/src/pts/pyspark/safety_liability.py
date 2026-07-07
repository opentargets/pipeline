"""Safety liability dataset generation.

Produces a per-target safety liability index by joining the pre-processed safety
evidence parquet (from pts_target_safety) with output/target for ENSG ID
resolution and output/disease for obsolete EFO term remapping.
"""

from __future__ import annotations

from typing import Any

import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import DataFrame

from pts.pyspark.common.session import Session
from pts.pyspark.common.utils import maybe_coalesce


def safety_liability(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Generate the safety liability dataset.

    Resolves ENSG IDs for symbol-keyed entries (ToxCast), remaps obsolete EFO
    disease terms to their current equivalents, and groups safety evidence by
    target into a ``safetyLiabilities`` array.

    Args:
        source: Mapping of logical input names to paths. Expected keys:
            ``safety_evidence`` (intermediate safety parquet from pts_target_safety),
            ``target`` (output/target parquet), and ``diseases`` (output/disease parquet).
        destination: Output path for the safety liability parquet dataset.
        settings: Step settings; supports ``partition_count`` (int, default 2).
        properties: Spark session properties passed to :class:`Session`.
    """
    spark = Session(app_name='safety_liability', properties=properties).spark

    safety_raw = spark.read.parquet(source['safety_evidence'])
    target_raw = spark.read.parquet(source['target'])
    diseases_raw = spark.read.parquet(source['diseases'])

    ensg_lookup = _build_ensg_lookup(target_raw)
    result = _build_safety_liabilities(safety_raw, ensg_lookup, diseases_raw)

    partition_count = (settings or {}).get('partition_count', 2)
    logger.info(f'writing safety liability to {destination} ({partition_count} partitions)')
    maybe_coalesce(result, partition_count).write.mode('overwrite').parquet(destination)


def _build_ensg_lookup(target_df: DataFrame) -> DataFrame:
    """Build a symbol/proteinId → ENSG ID lookup from the target dataset.

    ToxCast safety entries carry only a gene symbol or protein accession in
    ``targetFromSourceId`` rather than an ENSG ID. This lookup enables resolving
    those entries to their canonical ENSG ID.

    Args:
        target_df: Target parquet from ``output/target``.

    Returns:
        DataFrame with columns ``ensgId`` and ``name``
        (``array<str>`` of protein accessions and approved symbol).
    """
    return (
        target_df
        .select(
            f.col('id').alias('ensgId'),
            f.flatten(f.array(
                f.col('proteinIds.id'),
                f.array(f.col('approvedSymbol')),
            )).alias('name'),
        )
    )


def _build_safety_liabilities(
    safety_df: DataFrame,
    ensg_lookup: DataFrame,
    diseases_df: DataFrame,
) -> DataFrame:
    """Build the safety liability output from pre-processed safety evidence.

    Resolves missing ENSG IDs via symbol/protein lookup and replaces obsolete
    EFO disease terms with their current equivalents before grouping into the
    per-target ``safetyLiabilities`` array.

    Args:
        safety_df: Pre-processed safety evidence parquet (from pts_target_safety).
        ensg_lookup: ENSG lookup from :func:`_build_ensg_lookup`.
        diseases_df: Disease index parquet from ``output/disease``.

    Returns:
        DataFrame with one row per safety liability record and columns
        ``targetId``, ``event``, ``eventId``, ``effects``, ``biosamples``,
        ``datasource``, ``literature``, ``url``, ``studies``.
    """
    # Resolve ENSG IDs for symbol-keyed entries (ToxCast provides targetFromSourceId)
    enriched = (
        safety_df
        .join(ensg_lookup, f.array_contains(f.col('name'), f.col('targetFromSourceId')), 'left_outer')
        .drop(*[c for c in ensg_lookup.columns if c != 'ensgId'])
        .withColumn('id', f.coalesce(f.col('id'), f.col('ensgId')))
        .drop('ensgId')
    )

    # Remap obsolete EFO terms to their current equivalents
    disease_mapping = (
        diseases_df
        .select(
            f.col('id').alias('diseaseId'),
            f.explode(f.col('obsoleteTerms')).alias('obsoleteTerm'),
        )
    )

    return (
        enriched
        .join(disease_mapping, enriched['eventId'] == disease_mapping['obsoleteTerm'], 'left_outer')
        .withColumn('eventId', f.coalesce(f.col('diseaseId'), f.col('eventId')))
        .drop('obsoleteTerm', 'diseaseId')
        .filter(f.col('id').isNotNull())
        .select(
            f.col('id').alias('targetId'),
            'event',
            'eventId',
            'effects',
            'biosamples',
            'datasource',
            'literature',
            'url',
            'studies',
        )
    )
