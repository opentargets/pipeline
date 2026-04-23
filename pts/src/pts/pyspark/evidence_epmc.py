"""EPMC literature evidence generation.

Ported from Epmc.scala in platform-etl-backend (evidence portion only,
EPMCCooccurrences is not migrated). Aggregates cooccurrence data into
disease-target evidence entries with text mining sentences.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql import functions as f
from pyspark.sql.types import StringType

from pts.pyspark.common.session import Session

_EXCLUDED_TARGET_TERMS = [
    'TEC',
    'TECS',
    'Tec',
    'tec',
    "'",
    '(',
    ')',
    '-',
    '-S',
    'S',
    'S-',
    'SS',
    'SSS',
    'Ss',
    'Ss-',
    's',
    's-',
    'ss',
    'U3',
    'U6',
    'u6',
    'SNORA70',
    'U2',
    'U8',
]

_SECTIONS_OF_INTEREST = [
    'title',
    'abstract',
    'intro',
    'case',
    'figure',
    'table',
    'discuss',
    'concl',
    'results',
    'appendix',
    'other',
]


def _compute_evidence(cooccurrences: DataFrame) -> DataFrame:
    """Aggregate cooccurrence data into scored disease-target evidence.

    Args:
        cooccurrences: Raw cooccurrence DataFrame.

    Returns:
        DataFrame with evidence columns including textMiningSentences.
    """
    return (
        cooccurrences
        .filter(f.col('section').isin(_SECTIONS_OF_INTEREST))
        .withColumn('pmid', f.trim(f.col('pmid').cast(StringType())))
        .withColumn('publicationIdentifier', f.coalesce(f.col('pmid'), f.col('pmcid')))
        .filter(
            (f.col('type') == 'GP-DS')
            & f.col('isMapped')
            & f.col('publicationIdentifier').isNotNull()
            & (f.length(f.col('text')) < 600)
            & ~f.col('label1').isin(_EXCLUDED_TARGET_TERMS)
        )
        .withColumnRenamed('keywordId1', 'targetFromSourceId')
        .withColumnRenamed('keywordId2', 'diseaseFromSourceMappedId')
        .groupBy('publicationIdentifier', 'targetFromSourceId', 'diseaseFromSourceMappedId', 'year')
        .agg(
            f.collect_set(f.col('pmcid')).alias('pmcIds'),
            f.collect_set(f.col('pmid')).alias('literature'),
            f.collect_set(
                f.struct(
                    f.col('text'),
                    f.col('start1').alias('tStart'),
                    f.col('end1').alias('tEnd'),
                    f.col('start2').alias('dStart'),
                    f.col('end2').alias('dEnd'),
                    f.col('section'),
                )
            ).alias('textMiningSentences'),
            f.sum(f.col('evidence_score')).alias('resourceScore'),
        )
        .withColumn('pmcIds', f.when(f.size(f.col('pmcIds')) != 0, f.col('pmcIds')))
        .filter(f.col('resourceScore') > 1)
        .select(
            f.lit('europepmc').alias('datasourceId'),
            f.lit('literature').alias('datatypeId'),
            f.col('targetFromSourceId'),
            f.col('diseaseFromSourceMappedId'),
            f.col('resourceScore'),
            f.col('literature'),
            f.col('textMiningSentences'),
            f.col('pmcIds'),
            f.col('year').alias('publicationYear'),
        )
    )


def evidence_epmc(
    source: dict[str, str] | str,
    destination: dict[str, str] | str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Generate EPMC disease-target evidence from cooccurrence data."""
    spark = Session(app_name='evidence_epmc', properties=properties).spark

    logger.info('Reading literature cooccurrences')
    cooccurrences = spark.read.parquet(source['cooccurrences'])

    logger.info('Computing EPMC evidence')
    evidence = _compute_evidence(cooccurrences)

    dest = destination['evidence'] if isinstance(destination, dict) else destination
    logger.info(f'Writing EPMC evidence to {dest}')
    evidence.write.mode('overwrite').parquet(dest)
