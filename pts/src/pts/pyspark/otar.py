"""OTAR projects dataset generation.

Ported from Otar.scala in platform-etl-backend.
Joins OTAR project metadata with disease EFO mappings and propagates
project info to disease ancestors.
"""

from __future__ import annotations

from typing import Any

import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import DataFrame

from pts.pyspark.common.session import Session


def _generate_otar_info(
    disease: DataFrame,
    otar_meta: DataFrame,
    efo_lookup: DataFrame,
) -> DataFrame:
    """Generate per-disease OTAR project info with ancestor propagation.

    Args:
        disease: Disease DataFrame with columns [id, ancestors].
        otar_meta: OTAR metadata with [otar_code, project_name, project_status, integrates_in_PPP].
        efo_lookup: Mapping from [otar_code, efo_disease_id].

    Returns:
        DataFrame with [efo_id, projects[{otar_code, status, project_name,
        integrates_data_PPP, reference}]].
    """
    joined = otar_meta.join(efo_lookup, 'otar_code', 'left_outer')
    return (
        joined
        .withColumnRenamed('efo_disease_id', 'efo_code')
        .join(disease, f.col('efo_code') == f.col('id'), 'inner')
        .withColumn('ancestor', f.explode(f.concat(f.array(f.col('id')), f.col('ancestors'))))
        .groupBy(f.col('ancestor').alias('efo_id'))
        .agg(
            f.collect_set(
                f.struct(
                    f.col('otar_code'),
                    f.col('project_status').alias('status'),
                    f.col('project_name'),
                    f.col('integrates_in_PPP').cast('boolean').alias('integrates_data_PPP'),
                    f.concat(f.lit('http://home.opentargets.org/'), f.col('otar_code')).alias('reference'),
                )
            ).alias('projects')
        )
    )


def otar(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Generate OTAR projects dataset."""
    spark = Session(app_name='otar', properties=properties).spark

    logger.info('Reading otar inputs')
    disease = spark.read.parquet(source['diseases'])
    meta = spark.read.option('sep', ',').option('header', 'true').csv(source['otar_meta'])
    lookup = spark.read.option('sep', ',').option('header', 'true').csv(source['otar_project_to_efo'])

    result = _generate_otar_info(disease, meta, lookup)

    logger.info(f'Writing otar output to {destination}')
    result.write.mode('overwrite').parquet(destination)
