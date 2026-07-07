"""Target tractability dataset generation.

Processes the Open Targets tractability TSV and produces one row per
tractability bucket assessment (flat schema) across small molecule,
antibody, and other modalities. Only targets present in output/target are
retained.
"""

from __future__ import annotations

import re
from typing import Any

import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import DataFrame

from pts.pyspark.common.session import Session
from pts.pyspark.common.utils import maybe_coalesce


def target_tractability(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Build tractability assessments and write the output dataset.

    Args:
        source: Mapping with keys ``tractability`` (TSV) and ``target``
            (output/target parquet used for ID validation).
        destination: Output path for the tractability parquet.
        settings: Step settings; supports ``partition_count`` (int, default 2).
        properties: Spark properties passed to :class:`Session`.
    """
    spark = Session(app_name='target_tractability', properties=properties).spark

    logger.info('Reading tractability TSV')
    tractability_raw = spark.read.option('sep', '\t').option('header', 'true').csv(source['tractability'])

    logger.info('Reading target IDs for validation')
    target_ids = spark.read.parquet(source['target']).select('id')

    logger.info('Building tractability assessments')
    result = _build_tractability(tractability_raw, target_ids)

    partition_count = (settings or {}).get('partition_count', 2)
    logger.info(f'Writing target_tractability to {destination} ({partition_count} partitions)')
    maybe_coalesce(result, partition_count).write.mode('overwrite').parquet(destination)


def _build_tractability(df: DataFrame, target_ids: DataFrame) -> DataFrame:
    """Build tractability assessments from the tractability TSV.

    Columns matching the pattern ``*_B{N}_*`` are tractability bucket columns.
    Rows whose ``ensembl_gene_id`` is not present in the target universe are
    dropped.

    Args:
        df: Raw tractability TSV DataFrame.
        target_ids: DataFrame with a single ``id`` column from output/target.

    Returns:
        DataFrame with one row per tractability assessment and columns
        ``targetId``, ``modality``, ``id``, ``value``.
    """
    bucket_cols = [c for c in df.columns if re.match(r'.*_B\d+_.*', c)]
    tractability = df.select('ensembl_gene_id', *bucket_cols)
    data_cols = [c for c in tractability.columns if c != 'ensembl_gene_id']

    for col_name in data_cols:
        parts = col_name.split('_')
        tractability = tractability.withColumn(
            col_name,
            f.struct(
                f.lit(parts[0]).alias('modality'),
                f.lit(parts[-1]).alias('id'),
                f.when(f.col(f'`{col_name}`') == 1, True).otherwise(False).alias('value'),
            ),
        )

    return (
        tractability
        .select(
            f.col('ensembl_gene_id').alias('targetId'),
            f.array(*data_cols).alias('tractability'),
        )
        .join(target_ids, f.col('targetId') == f.col('id'), 'inner')
        .drop('id')
        .select('targetId', f.explode('tractability').alias('assessment'))
        .select('targetId', 'assessment.modality', 'assessment.id', 'assessment.value')
    )
