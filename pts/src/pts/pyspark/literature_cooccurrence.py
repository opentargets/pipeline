"""Literature cooccurrence step.

Generates target-disease cooccurrences from matches and writes them as parquet.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from pts.pyspark.common.session import Session
from pts.pyspark.common.utils import maybe_coalesce
from pts.pyspark.literature_utils.dataset.match_mapped import MatchMapped


def literature_cooccurrence(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Generate target-disease cooccurrences from the match dataset.

    Args:
        source: ``match``.
        destination: ``cooccurrence``.
        settings: optional ``cooccurrence_coalesce`` (int) for the output partition count.
        properties: Spark properties forwarded to the session.
    """
    spark = Session(app_name='literature', properties=properties)

    cooccurrence_coalesce = settings.get('cooccurrence_coalesce')

    logger.info(f'load matches from: {source["match"]}')
    match = spark.load_data(path=source['match'])
    logger.info(f'[DIAG] match partitions: {match.rdd.getNumPartitions()}')  # ty:ignore[call-non-callable]

    logger.info(f'write cooccurrences to {destination["cooccurrence"]} (coalesce={cooccurrence_coalesce})')
    cooccurrence = MatchMapped(match).generate_target_disease_cooccurrences().df
    logger.info(f'[DIAG] cooccurrence partitions: {cooccurrence.rdd.getNumPartitions()}')  # ty:ignore[call-non-callable]
    maybe_coalesce(cooccurrence, cooccurrence_coalesce).write.mode('overwrite').parquet(destination['cooccurrence'])
