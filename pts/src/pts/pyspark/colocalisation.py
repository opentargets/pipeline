"""Colocalisation step.

Finds the overlapping pairs of credible sets and scores each pair with coloc_pip_ecaviar.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from pts.pyspark.colocalisation_utils.coloc_pip_ecaviar import colocalise
from pts.pyspark.colocalisation_utils.overlap import aligned_tags, overlapping_pairs, tags
from pts.pyspark.common.session import Session
from pts.pyspark.common.utils import maybe_coalesce
from pts.schemas.colocalisation import CREDIBLE_SET_COLUMNS, CREDIBLE_SET_VARIANT_COLUMNS
from pts.schemas.credible_set import credible_set_schema, credible_set_variant_schema
from pts.schemas.spark import to_spark_schema


def colocalisation(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Colocalise the credible sets of the release.

    Args:
        source: ``credible_set`` and ``credible_set_variant``.
        destination: ``colocalisation``.
        settings: optional ``priorc1``, ``priorc2``, ``priorc12`` and ``output_partitions`` (int).
        properties: Spark properties forwarded to the session.
    """
    spark = Session(app_name='colocalisation', properties=properties)

    priors = {key: float(settings[key]) for key in ('priorc1', 'priorc2', 'priorc12') if key in settings}

    logger.info(f'load credible sets from: {source["credible_set"]}')
    credible_set = spark.load_data(
        source['credible_set'], schema=to_spark_schema(credible_set_schema, CREDIBLE_SET_COLUMNS)
    )
    logger.info(f'load credible set variants from: {source["credible_set_variant"]}')
    credible_set_variant = spark.load_data(
        source['credible_set_variant'],
        schema=to_spark_schema(credible_set_variant_schema, CREDIBLE_SET_VARIANT_COLUMNS),
    )

    all_tags = tags(credible_set, credible_set_variant).persist()
    pairs = overlapping_pairs(all_tags).persist()
    result = colocalise(aligned_tags(all_tags, pairs), **priors)

    logger.info(f'write colocalisation to {destination["colocalisation"]}')
    maybe_coalesce(result, settings.get('output_partitions')).write.mode('overwrite').parquet(
        destination['colocalisation']
    )
    pairs.unpersist()
    all_tags.unpersist()
