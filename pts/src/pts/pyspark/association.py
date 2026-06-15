"""Application to generate timeseries data."""

from typing import Any

from loguru import logger
from pyspark.storagelevel import StorageLevel

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
    partition_count = settings.get('partition_count') or {}
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

    def _write(df, key: str) -> None:
        n = partition_count.get(key)
        (df.coalesce(n) if n else df).write.mode('overwrite').parquet(destination[key])

    # Save direct association by datasource:
    logger.info('Processing direct association stratified by datasource.')
    _write(
        association_by_datasource.compute_novelty(
            novelty_scale=novelty_scale,
            novelty_shift=novelty_shift,
            novelty_window=novelty_window,
        ),
        'by_datasource_direct',
    )

    # Save direct overall association:
    logger.info('Processing direct overall association.')
    _write(
        association_by_datasource
        .aggregate_overall(datasource_weights)
        .compute_novelty(
            novelty_scale=novelty_scale,
            novelty_shift=novelty_shift,
            novelty_window=novelty_window,
        ),
        'overall_direct',
    )

    # Save direct association by datatype:
    logger.info('Processing direct associations stratified by datatype.')
    _write(
        association_by_datasource
        .aggregate_by_datatype(datasource_weights)
        .compute_novelty(
            novelty_scale=novelty_scale,
            novelty_shift=novelty_shift,
            novelty_window=novelty_window,
        ),
        'by_datatype_direct',
    )

    # Unpersist temporary, datasource specific data:
    association_by_datasource.df.unpersist()

    # Processing indirect associations. This time the exploded dataset is saved as a temporary
    # parquet file and re-used for all downstream aggregation (instead of persisting):
    logger.info('Processing indirect associations...')
    (
        Evidence
        .from_raw_evidence(raw_evidence)
        .expand_disease(disease_index=disease_df, datasource_weight=datasource_weights)
        .aggregate_evidence_by_datasource()
        .df.write.mode('overwrite')
        .parquet(destination['temporary'])
    )

    # Load the indirect intermediate once and persist for the three downstream uses,
    # so each downstream aggregation reads from cache rather than re-scanning GCS.
    indirect_intermediate = session.load_data(destination['temporary']).persist(StorageLevel.MEMORY_AND_DISK)

    # Save indirect association by datasource:
    logger.info('Processing indirect associations stratified by datasource.')
    _write(
        Association(_df=indirect_intermediate)
        .compute_novelty(
            novelty_scale=novelty_scale,
            novelty_shift=novelty_shift,
            novelty_window=novelty_window,
        ),
        'by_datasource_indirect',
    )

    # Save indirect association by datatype:
    logger.info('Processing indirect associations stratified by datatype.')
    _write(
        Association(_df=indirect_intermediate)
        .aggregate_by_datatype(datasource_weights)
        .compute_novelty(
            novelty_scale=novelty_scale,
            novelty_shift=novelty_shift,
            novelty_window=novelty_window,
        ),
        'by_datatype_indirect',
    )

    # Save indirect overall association:
    logger.info('Processing indirect overall associations.')
    _write(
        Association(_df=indirect_intermediate)
        .aggregate_overall(datasource_weights)
        .compute_novelty(
            novelty_scale=novelty_scale,
            novelty_shift=novelty_shift,
            novelty_window=novelty_window,
        ),
        'overall_indirect',
    )

    # Free the cache.
    indirect_intermediate.unpersist()
