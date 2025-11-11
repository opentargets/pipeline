"""Module to apply validation and further post processing on Open Targets Platform evidence."""

from __future__ import annotations

from typing import Any

from loguru import logger

from pts.pyspark.common.session import Session
from pts.pyspark.common.utils import (
    linear_rescaling,
    pvalue_linear_rescaling,
)
from pts.pyspark.evidence_utils.evidence import Evidence
from pts.pyspark.evidence_utils.validation_lut import LookUpTables


def evidence_postprocess(
    source: dict[str, str], destination: dict[str, str], settings: dict[str, Any], properties: dict[str, str]
) -> None:
    """Post-process evidence.

    Process:
        1. Validate disease
        2. Validate target
        3. Validate datasource
        4. Assign evidence identifier
        5. Calculate evidence date
        6. Calculate direction of effect
        7. Write output.

    Args:
        source (dict[str, str]): dictionary with data sources.
        destination (dict[str, str]): dictionary with output location.
        settings (dict[str, str]): collection of step specific configuration.
        properties (dict[str, str]): collection of session specific configuration
    """
    datasource_id = settings['datasource_id']
    evidence_format = settings['evidence_format']
    score_expression = settings['score_expression']

    logger.info(f'processing "{datasource_id}" evidence')
    # Initialise session:
    session = Session(app_name='evidence', properties=properties)

    # Registering UDFs in the spark session:
    session.spark.udf.register('linear_rescale', linear_rescaling)
    session.spark.udf.register('pvalue_linear_score', pvalue_linear_rescaling)

    # Read input data:
    source_evidence = session.load_data(source['evidence_path'], format=evidence_format)

    # Reading look up tables and make surer all looks good:
    lookup_tables = LookUpTables(session, source)
    assert lookup_tables.disease_lut is not None, 'disease_lut not generated'
    assert lookup_tables.target_lut is not None, 'target_lut not generated'
    assert lookup_tables.publication_lut is not None, 'publication_lut not generated'

    # Processing evidence:
    processed_evidence = (
        Evidence(source_evidence)
        # Validate entities:
        .validate_diseases(lookup_tables.disease_lut)
        .validate_target(lookup_tables.target_lut, settings.get('excluded_biotypes'))
        .validate_datasource(settings['datasource_id'])
        # Assing evidence identifier + check duplication:
        .assign_evidence_identifier(settings['unique_fields'])
        .validate_uniqueness()
        # Resolving evidence date:
        .resolve_publication_date(lookup_tables.publication_lut)
        .resolve_evidence_date()
        # Calculating score:
        .calculate_evidence_score(score_expression)
        # Calculate direction of effect:
        .assign_direction_on_trait(settings.get('direction_on_trait_expression'))
        .assign_direction_on_target(
            direction_on_target_expression=settings.get('direction_on_target_expression'),
            mechanism_of_action_lut=lookup_tables.mechanism_of_action_lut,
            target_lut=lookup_tables.target_lut,
        )
        # Hash long variant identifiers:
        .hash_long_variant_identifiers()
    )

    # Writing outputs:
    processed_evidence.get_invalid_evidence().write.mode('overwrite').parquet(destination['failed_evidence'])
    processed_evidence.get_valid_evidence().write.mode('overwrite').parquet(destination['evidence'])
