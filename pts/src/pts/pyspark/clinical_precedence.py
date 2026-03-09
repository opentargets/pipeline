"""Generate clinical precedence evidence from clinical reports and drug mechanisms of action."""

from typing import Any

import pyspark.sql.functions as f
from loguru import logger

from pts.pyspark.common.session import Session


def clinical_precedence(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Generate clinical precedence evidence by joining clinical reports with drug mechanisms of action.

    Args:
        source: Dictionary with paths to:
            - clinical_report: Clinical report parquet from clinical_report step
            - drug_mechanism_of_action: Drug mechanism of action parquet
        destination: Dictionary with paths to:
            - output: Path to write the clinical precedence evidence parquet.
            - excluded: Path to write excluded clinical reports that failed QC.
        settings: Custom settings with:
            - invalid_clinical_report_qc: List of QC reason strings to exclude.
        properties: Spark configuration options.
    """
    spark = Session(app_name='clinical_precedence', properties=properties)

    logger.info(f'Loading data from {source}')
    clinical_report_df = spark.load_data(source['clinical_report'])
    drug_moa_df = spark.load_data(source['drug_mechanism_of_action'])

    # Filter out clinical reports that fail QC
    invalid_qc_reasons = settings.get('invalid_clinical_report_qc', [])

    if invalid_qc_reasons:
        invalid_qc_array = f.array([f.lit(reason) for reason in invalid_qc_reasons])
        has_invalid_qc = f.coalesce(
            f.arrays_overlap(f.col('qualityControls'), invalid_qc_array),
            f.lit(False),
        )
        excluded_cr = clinical_report_df.filter(has_invalid_qc)
        clinical_report_df = clinical_report_df.filter(~has_invalid_qc)
    else:
        excluded_cr = clinical_report_df.filter(f.lit(False))

    logger.info(f'Writing excluded clinical reports to {destination["excluded"]}')
    excluded_cr.write.mode('overwrite').parquet(destination['excluded'])

    # Explode clinical_report: diseases and drugs
    exploded_cr = (
        clinical_report_df
        .withColumnRenamed('id', 'clinicalReportId')
        .withColumn('studyStartDate', f.col('trialStartDate').cast('string'))
        .withColumn('literature', f.col('trialLiterature'))
        .withColumn('_disease', f.explode_outer('diseases'))
        .withColumn('diseaseFromSource', f.col('_disease.diseaseFromSource'))
        .withColumn('diseaseFromSourceMappedId', f.col('_disease.diseaseId'))
        .withColumn('_drug', f.explode_outer('drugs'))
        .withColumn('drugFromSource', f.col('_drug.drugFromSource'))
        .withColumn('drugId', f.col('_drug.drugId'))
        .filter(f.col('diseaseFromSourceMappedId').isNotNull() & f.col('drugId').isNotNull())  # ty:ignore[missing-argument]
    )

    # Prepare drug_mechanism_of_action: explode chemblIds to get drugId, keep targets
    drug_moa_exploded = drug_moa_df.select(f.explode('chemblIds').alias('drugId'), 'targets', 'actionType')

    # Join and explode targets
    evidence = (
        exploded_cr
        .join(drug_moa_exploded, on='drugId', how='inner')
        .withColumn('targetFromSourceId', f.explode_outer('targets'))
        .filter(f.col('targetFromSourceId').isNotNull())  # ty:ignore[missing-argument]
        .select(
            'clinicalReportId',
            'clinicalStage',
            'trialWhyStopped',
            'trialStopReasonCategories',
            'studyStartDate',
            'literature',
            'diseaseFromSource',
            'diseaseFromSourceMappedId',
            'drugFromSource',
            'drugId',
            'targetFromSourceId',
            f.lit('clinical_precedence').alias('datasourceId'),
            f.lit('clinical').alias('datatypeId'),
            'actionType',
        )
    )

    evidence = evidence.distinct()

    logger.info(f'Writing clinical precedence evidence to {destination["output"]}')
    evidence.write.mode('overwrite').parquet(destination['output'])
