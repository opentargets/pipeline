"""This module adds the category of why a clinical trial has stopped early to the ChEMBL evidence."""

from typing import Any

import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql.dataframe import DataFrame

from pts.pyspark.common.session import Session


def chembl(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """This module adds the studyStopReasonCategories to the ChEMBL evidence as a result of the.

    categorisation of the clinical trial reason to stop.

    The evidence from clinical trials is also filtered for Phase IV trials.
    """
    spark = Session(app_name='gene_burden', properties=properties)

    logger.info(f'load data from {source}')
    chembl_df = spark.load_data(source['chembl_evidence'], format='json')
    predictions_df = spark.load_data(source['stop_reasons'], format='json')
    chembl_indications_df = spark.load_data(source['drug_indications'], format='json')

    logger.info('Joining ChEMBL evidence with predicted stopped reasons')
    pretty_predictions_df = predictions_df.transform(prettify_subclasses).distinct()
    early_stopped_evd_df = (
        # Evidence with a given reason to stop is always supported by a single NCT ID
        chembl_df.filter(f.col('studyStopReason').isNotNull())
        .alias('chembl')
        .join(
            pretty_predictions_df.alias('predictions'),
            f.col('chembl.studyId') == f.col('predictions.nct_id'),
            how='left',
        )
        .drop('nct_id')
        .distinct()
    )
    # We expect that ~10% of evidence strings have a reason to stop assigned
    # It is asserted that this fraction is between 9 and 15% of the total count
    total_count = chembl_df.count()
    early_stopped_count = early_stopped_evd_df.count()
    if not 0.08 < early_stopped_count / total_count < 0.15:
        logger.warning(f'Fraction of early stopped evidence is not as expected ({early_stopped_count / total_count}).')
    chembl_df_w_predictions = chembl_df.filter(f.col('studyStopReason').isNull()).unionByName(
        early_stopped_evd_df, allowMissingColumns=True
    )
    assert chembl_df_w_predictions.count() == chembl_df.count()

    logger.info('Removing T/D evidence from Phase IV trials that have not demonstrated efficacy')
    drug_approvals_df = (
        chembl_indications_df.filter(f.col('max_phase_for_ind').cast('double').cast('int') == 4)
        .selectExpr('efo_id as diseaseFromSourceMappedId', 'molecule_chembl_id as drugId')
        .withColumn('diseaseFromSourceMappedId', f.translate('diseaseFromSourceMappedId', ':', '_'))
        .distinct()
    )
    filtered_chembl_df = remove_unvalidated_target_disease(chembl_df_w_predictions, drug_approvals_df)

    logger.info(f'write chembl evidence strings to {destination}')
    filtered_chembl_df.write.parquet(destination, mode='overwrite')


def prettify_subclasses(predictions_df: DataFrame) -> DataFrame:
    """List of categories must be converted formatted with a nice name."""
    categories_mappings = {
        'Business_Administrative': 'Business or administrative',
        'Logistics_Resources': 'Logistics or resources',
        'Covid19': 'COVID-19',
        'Safety_Sideeffects': 'Safety or side effects',
        'Endpoint_Met': 'Met endpoint',
        'Insufficient_Enrollment': 'Insufficient enrollment',
        'Negative': 'Negative',
        'Study_Design': 'Study design',
        'Invalid_Reason': 'Invalid reason',
        'Study_Staff_Moved': 'Study staff moved',
        'Another_Study': 'Another study',
        'No_Context': 'No context',
        'Regulatory': 'Regulatory',
        'Interim_Analysis': 'Interim analysis',
        'Success': 'Success',
        'Ethical_Reason': 'Ethical reason',
        'Insufficient_Data': 'Insufficient data',
        'Uncategorised': 'Uncategorised',
    }
    sub_mapping_col = f.map_from_entries(
        f.array(*[f.struct(f.lit(k), f.lit(v)) for k, v in categories_mappings.items()])
    )
    return (
        predictions_df.select('nct_id', 'subclasses', sub_mapping_col.alias('prettyStopReasonsMap'))
        # Create a MapType column to convert each element of the subclasses array
        .withColumn(
            'studyStopReasonCategories', f.expr('transform(subclasses, x -> element_at(prettyStopReasonsMap, x))')
        )
        .drop('subclasses', 'prettyStopReasonsMap')
    )


def remove_unvalidated_target_disease(evidence: DataFrame, drug_approvals_df: DataFrame) -> DataFrame:
    """Remove evidence from Phase IV trials where the target disease is not approved or investigational."""
    return (
        evidence.join(
            drug_approvals_df.select('*', f.lit(True).alias('isApproved')),
            on=['drugId', 'diseaseFromSourceMappedId'],
            how='left',
        )
        .filter(
            # Keep all <Phase IV evidence
            (f.col('clinicalPhase').cast('double').cast('int') != 4)
            |
            # Keep all Phase IV evidence clinically validated
            ((f.col('clinicalPhase').cast('double').cast('int') == 4) & (f.col('isApproved')))
        )
        .drop('isApproved')
    )
