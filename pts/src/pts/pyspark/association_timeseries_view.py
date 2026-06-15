"""Module to generate the association timeseries view.

Unions all four association datasets (overall and by-datasource, direct and
indirect), explodes the timeseries array, and writes a flat parquet dataset
suitable for time-series analysis.
"""

from typing import Any

from pyspark.sql import functions as f

from pts.pyspark.common import Session


def association_timeseries_view(
    source: dict[str, str], destination: str, settings: dict[str, Any], properties: dict[str, str]
) -> None:
    """Generate a flat association timeseries view from all four association datasets.

    Reads the four association datasets (overall and by-datasource, each in direct
    and indirect flavours), tags each row with an `isDirect` flag, unions them, and
    explodes the `timeseries` array column so that every year-level entry becomes its
    own row. The result is partitioned by a composite `partitionKey` column
    (``targetId_diseaseId_isDirect``) with rows sorted by `year` within each partition.

    Args:
        source: Dict with keys:
            - ``association_overall_direct``: path to the overall-direct association dataset.
            - ``association_overall_indirect``: path to the overall-indirect association dataset.
            - ``association_by_datasource_direct``: path to the by-datasource-direct dataset.
            - ``association_by_datasource_indirect``: path to the by-datasource-indirect dataset.
        destination: Output parquet path for the timeseries view.
        settings: Step-specific settings (unused, kept for interface consistency).
        properties: Spark configuration properties.
    """
    session = Session(app_name='Association timeseries view generation', properties=properties)

    # Load input datasets:
    association_overall_direct = session.load_data(source['association_overall_direct']).withColumn(
        'isDirect', f.lit(True)
    )
    association_overall_indirect = session.load_data(source['association_overall_indirect']).withColumn(
        'isDirect', f.lit(False)
    )

    association_by_datasource_direct = session.load_data(source['association_by_datasource_direct']).withColumn(
        'isDirect', f.lit(True)
    )

    association_by_datasource_indirect = session.load_data(source['association_by_datasource_indirect']).withColumn(
        'isDirect', f.lit(False)
    )

    (
        association_overall_direct
        .unionByName(association_overall_indirect)
        .unionByName(association_by_datasource_direct)
        .unionByName(association_by_datasource_indirect)
        .withColumn('col', f.explode('timeseries'))
        .select(
            'diseaseId',
            'targetId',
            'aggregationType',
            'aggregationValue',
            'col.*',
            'isDirect',
            f.concat_ws('_', 'targetId', 'diseaseId', 'isDirect').alias('partitionKey'),
        )
        .orderBy('partitionKey', 'aggregationValue', 'year')
        .write.mode('overwrite')
        .parquet(destination)
    )
