"""Application to generate timeseries data."""

import os
from typing import Any
from venv import logger

from loguru import logger
from pyspark.sql import DataFrame

from pts.pyspark.common.session import Session

from .timeseries_utils.evidence import Evidence
from .timeseries_utils.utils import get_weight_for_datasource, read_yaml_config

from pyspark.sql import SparkSession


def timeseries(
    source: dict[str, str],
    destination: dict[str, str],
    properties: dict[str, Any],
) -> None:
    """Main function to generate timeseries data.

    Args:
        source (dict[str, str]): list of .
        destination (dict[str, str]): list of outputs of this parser.
        properties (dict[str, Any]): list of properties for this step.
    """
    # Extract novelty parameters:
    novelty_scale = properties['novelty_scale']
    novelty_window = properties['novelty_window']
    novelty_shift = properties['novelty_shift']
    # start spark session
    session = Session(app_name='timeseries', properties=properties)

    # Reading evidence data:
    raw_evidence = session.load_data(source['evidence'])

    # Reading disease data to generate indirect evidence:
    disease = session.load_data(source['disease'])

    # Extracting datasource weights:
    datasource_weights = session.spark.createDataFrame(properties['datasource_weights'])

    # Calculate novelty based on overall direct associations over the years:
    logger.info('Calculating overall direct timeseries:')
    (
        Evidence.from_raw_evidence(raw_evidence)
        .apply_datasource_weight(datasource_weights)
        .aggregate_evidence(aggregation_type='overall')
        .compute_novelty(
            novelty_scale=novelty_scale,
            novelty_shift=novelty_shift,
            novelty_window=novelty_shift,
        )
        .write.mode('overwrite')
        .parquet(destination['overall_direct'])
    )
