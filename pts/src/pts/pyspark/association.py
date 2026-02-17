"""Application to generate timeseries data."""

from typing import Any

from loguru import logger

from pts.pyspark.associations_utils.association import Association
from pts.pyspark.associations_utils.evidence import Evidence
from pts.pyspark.common.session import Session
from pts.pyspark.common.utils import parse_spark_schema


def association(
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
    datasource_weights = session.spark.createDataFrame(settings['datasource_weights'])

    # Processing direct association, as a first step aggregate evidence by datasource.
    # This is a re-used and persisted dataset.
    association_by_datasource = Evidence.from_raw_evidence(raw_evidence).aggregate_evidence_by_datasource(persist=True)

    # Save direct association by datasource:
    logger.info('Processing direct association stratified by datasource.')
    (
        association_by_datasource.compute_novelty(
            novelty_scale=novelty_scale,
            novelty_shift=novelty_shift,
            novelty_window=novelty_window,
        )
        .write.mode('overwrite')
        .parquet(destination['by_datasource_direct'])
    )

    # Save direct overall association:
    logger.info('Processing direct overall association.')
    (
        association_by_datasource.aggregate_overall(datasource_weights)
        .compute_novelty(
            novelty_scale=novelty_scale,
            novelty_shift=novelty_shift,
            novelty_window=novelty_window,
        )
        .write.mode('overwrite')
        .parquet(destination['overall_direct'])
    )

    # Save direct association by datatype:
    logger.info('Processing direct associations stratified by datatype.')
    (
        association_by_datasource.aggregate_by_datatype(datasource_weights)
        .compute_novelty(
            novelty_scale=novelty_scale,
            novelty_shift=novelty_shift,
            novelty_window=novelty_window,
        )
        .write.mode('overwrite')
        .parquet(destination['by_datatype_direct'])
    )

    # Unpersist temporary, datasource specific data:
    association_by_datasource.df.unpersist()

    # Processing indirect associations. This time the exploded dataset is saved as a temporary
    # parquet file and re-used for all downstream aggregation (instead of persisting):
    logger.info('Processing indirect associations...')
    (
        Evidence.from_raw_evidence(raw_evidence)
        .expand_disease(disease_index=disease_df)
        .aggregate_evidence_by_datasource()
        .df.write.mode('overwrite')
        .parquet(destination['temporary'])
    )

    # Save direct association by datasource:
    logger.info('Processing indirect associations stratified by datasource.')
    (
        Association(_df=session.load_data(destination['temporary']))
        .compute_novelty(
            novelty_scale=novelty_scale,
            novelty_shift=novelty_shift,
            novelty_window=novelty_window,
        )
        .write.mode('overwrite')
        .parquet(destination['by_datasource_indirect'])
    )

    # Save direct association by datasource:
    logger.info('Processing indirect associations stratified by datatype.')
    (
        Association(_df=session.load_data(destination['temporary']))
        .aggregate_by_datatype(datasource_weights)
        .compute_novelty(
            novelty_scale=novelty_scale,
            novelty_shift=novelty_shift,
            novelty_window=novelty_window,
        )
        .write.mode('overwrite')
        .parquet(destination['by_datatype_indirect'])
    )

    # Save direct association by datasource:
    logger.info('Processing indirect overall associations.')
    (
        Association(_df=session.load_data(destination['temporary']))
        .aggregate_overall(datasource_weights)
        .compute_novelty(
            novelty_scale=novelty_scale,
            novelty_shift=novelty_shift,
            novelty_window=novelty_window,
        )
        .write.mode('overwrite')
        .parquet(destination['overall_indirect'])
    )
