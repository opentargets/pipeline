"""Literature entity lookup table with harmonic relevance scoring.

Ported from Processing.filterMatchesForCH in platform-etl-backend.
Computes a harmonic relevance score per (pmid, keywordId) pair based on
the sections of the publication where the entity was mentioned.
"""

from __future__ import annotations

import operator
from typing import Any

from loguru import logger
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from pyspark.sql.window import Window

from pts.pyspark.common.session import Session

# Section importance ranks (from Scala reference.conf)
_SECTION_RANKS = [
    {'section': 'title', 'rank': 1, 'weight': 1.0},
    {'section': 'abstract', 'rank': 1, 'weight': 0.8},
    {'section': 'concl', 'rank': 1, 'weight': 0.7},
    {'section': 'results', 'rank': 2, 'weight': 0.6},
    {'section': 'discuss', 'rank': 2, 'weight': 0.5},
    {'section': 'methods', 'rank': 3, 'weight': 0.3},
    {'section': 'other', 'rank': 4, 'weight': 0.1},
]

_TITLE_WEIGHT = 1.0

_OUTPUT_COLUMNS = [
    'pmid',
    'pmcid',
    'date',
    'year',
    'month',
    'day',
    'keywordId',
    'relevance',
    'keywordType',
]


def _harmonic_fn(v: Column, s: Column) -> Column:
    """Compute harmonic relevance: sum(weight_i / i^2) for i in 1..n.

    Ports Processing.harmonicFn from Scala.
    """
    return f.when(
        s <= 0,
        f.lit(0.0),
    ).otherwise(
        f.aggregate(
            f.zip_with(v, f.sequence(f.lit(1), s), lambda e1, e2: e1 / f.pow(e2, 2)),
            f.lit(0.0),
            operator.add,
        )
    )


def _compute_relevance(matches: DataFrame) -> DataFrame:
    """Compute harmonic relevance scores for entity-publication pairs.

    Args:
        matches: DataFrame with literature match data (pmid, keywordId, section, type, etc.)

    Returns:
        DataFrame with columns: pmid, pmcid, date, year, month, day, keywordId, relevance, keywordType
    """
    spark = matches.sparkSession

    section_rank_table = f.broadcast(spark.createDataFrame(_SECTION_RANKS).orderBy(f.col('rank').asc()))

    w_by_section_keyword = Window.partitionBy('pmid', 'section', 'keywordId')

    # Step 1: join with section ranks and compute per-section weight vectors.
    # Title always gets a single fixed weight; other sections collect all
    # mention weights within that section.
    with_section_weights = (
        matches
        .withColumnRenamed('type', 'keywordType')
        .join(section_rank_table, on='section', how='left_outer')
        .na.fill(100, ['rank'])
        .na.fill(0.01, ['weight'])
        .withColumn(
            'keywordSectionV',
            f.when(
                f.col('section') != 'title',
                f.collect_list(f.col('weight')).over(w_by_section_keyword),
            ).otherwise(f.array(f.lit(_TITLE_WEIGHT))),
        )
        .dropDuplicates(['pmid', 'section', 'keywordId'])
    )

    # Step 2: aggregate across sections per (pmid, keywordId).
    # sort_array on rank ensures deterministic ordering (lower rank = more important).
    agg_cols = [
        f.first('pmcid').alias('pmcid'),
        f.first('date').alias('date'),
        f.first('year').alias('year'),
        f.first('month').alias('month'),
        f.first('day').alias('day'),
        f.first('keywordType').alias('keywordType'),
        f.flatten(
            f.sort_array(
                f.collect_list(
                    f.struct(
                        f.col('rank'),
                        (f.lit(0.0) - f.col('weight')).alias('neg_weight'),
                        f.col('keywordSectionV'),
                    )
                )
            ).getField('keywordSectionV')
        ).alias('relevanceV'),
    ]

    return (
        with_section_weights
        .groupBy('pmid', 'keywordId')
        .agg(*agg_cols)
        .withColumn('relevance', _harmonic_fn(f.col('relevanceV'), f.size(f.col('relevanceV'))))
        .select(*_OUTPUT_COLUMNS)
    )


def literature_entity_lut(
    source: dict[str, str] | str,
    destination: dict[str, str] | str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Generate literature entity lookup table with harmonic relevance scoring."""
    spark = Session(app_name='literature_entity_lut', properties=properties).spark

    logger.info('Reading literature matches')
    matches = spark.read.parquet(source['matches'])

    logger.info('Computing relevance scores')
    result = _compute_relevance(matches)

    dest = destination['literature_entity_lut'] if isinstance(destination, dict) else destination
    logger.info(f'Writing literature entity LUT to {dest}')
    result.write.mode('overwrite').parquet(dest)
