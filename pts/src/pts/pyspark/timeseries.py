"""Application to generate timeseries data."""

from typing import Any

from loguru import logger

from pts.pyspark.common.session import Session
from pts.pyspark.common.utils import parse_spark_schema
from pts.pyspark.timeseries_utils.evidence import Evidence


def timeseries(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Main function to generate timeseries data.

    Args:
        source (dict[str, str]): list of inputs.
        destination (dict[str, str]): list of outputs of this parser.
        settings (dict[str, Any]): list of settings for this step.
        properties (dict[str, Any]): list of properties for this step.
    """
    # Extract novelty parameters:
    novelty_scale = settings['novelty_scale']
    novelty_window = settings['novelty_window']
    novelty_shift = settings['novelty_shift']
    # start spark session
    session = Session(app_name='timeseries', properties=properties)

    # Reading evidence data:
    raw_evidence = session.load_data(source['evidence'], schema=parse_spark_schema('evidence.json'))

    # Reading disease data to generate indirect evidence:
    disease_df = session.load_data(source['disease'])

    # Extracting datasource weights:
    datasource_weights = session.spark.createDataFrame(settings['datasource_weights']).withColumnRenamed(
        'id', 'datasourceId'
    )

    # We might not request all outputs:
    for data_output, output_path in destination.items():
        # Timeseries for overall, direct associations:
        if data_output == 'overall_direct':
            logger.info('calculating overall direct timeseries')
            (
                Evidence.from_raw_evidence(raw_evidence)
                .apply_datasource_weight(datasource_weights)
                .aggregate_evidence(aggregation_type='overall')
                .compute_novelty(
                    novelty_scale=novelty_scale,
                    novelty_shift=novelty_shift,
                    novelty_window=novelty_window,
                )
                .write.mode('overwrite')
                .parquet(output_path)
            )

        # Timeseries for overall, indirect associations:
        elif data_output == 'overall_indirect':
            logger.info('calculating overall indirect timeseries')
            (
                Evidence.from_raw_evidence(raw_evidence)
                .expand_disease(disease_df)
                .apply_datasource_weight(datasource_weights)
                .aggregate_evidence(aggregation_type='overall')
                .compute_novelty(
                    novelty_scale=novelty_scale,
                    novelty_shift=novelty_shift,
                    novelty_window=novelty_window,
                )
                .write.mode('overwrite')
                .parquet(output_path)
            )

        # Timeseries for datasource specific, direct associations:
        elif data_output == 'by_datasource_direct':
            logger.info('calculating datasource specific direct timeseries')
            (
                Evidence.from_raw_evidence(raw_evidence)
                .aggregate_evidence(aggregation_type='datasourceId')
                .compute_novelty(
                    novelty_scale=novelty_scale,
                    novelty_shift=novelty_shift,
                    novelty_window=novelty_window,
                )
                .write.mode('overwrite')
                .parquet(output_path)
            )

        # Timeseries for datasource specific, indirect associations:
        elif data_output == 'by_datasource_indirect':
            logger.info('calculating datasource specific indirect timeseries')
            (
                Evidence.from_raw_evidence(raw_evidence)
                .expand_disease(disease_df)
                .aggregate_evidence(aggregation_type='datasourceId')
                .compute_novelty(
                    novelty_scale=novelty_scale,
                    novelty_shift=novelty_shift,
                    novelty_window=novelty_window,
                )
                .write.mode('overwrite')
                .parquet(output_path)
            )
