"""Drug index generation.

Combines molecule data with clinical reports, mechanisms of action, and chemical probes
to produce the final drug index. Filters to include only molecules that qualify as
"drugs" and generates human-readable descriptions.
"""

from typing import Any

import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType

from pts.pyspark.common.session import Session

# Stage ranking: lower rank = more advanced stage
CLINICAL_STAGE_RANKS = {
    'APPROVAL': 1,
    'PREAPPROVAL': 2,
    'PHASE_3': 3,
    'PHASE_2_3': 4,
    'PHASE_2': 5,
    'PHASE_1_2': 6,
    'PHASE_1': 7,
    'EARLY_PHASE_1': 8,
    'IND': 9,
    'PRECLINICAL': 10,
    'UNKNOWN': 11,
}

# Stages that should be treated as APPROVAL when computing the max
STAGE_FOR_MAX_MAPPING = {'WITHDRAWAL': 'APPROVAL', 'PHASE_4': 'APPROVAL'}

STAGE_DISPLAY_NAMES = {
    'APPROVAL': 'approved',
    'PREAPPROVAL': 'pre-approval',
    'PHASE_3': 'phase III',
    'PHASE_2_3': 'phase II/III',
    'PHASE_2': 'phase II',
    'PHASE_1_2': 'phase I/II',
    'PHASE_1': 'phase I',
    'EARLY_PHASE_1': 'early phase I',
    'IND': 'IND',
    'PRECLINICAL': 'preclinical',
    'UNKNOWN': 'unknown',
}

# Inverse mapping: rank -> stage string
RANK_TO_STAGE = {v: k for k, v in CLINICAL_STAGE_RANKS.items()}


def drug_molecule(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Generate the drug molecule index.

    Args:
        source: Dictionary with paths to:
            - molecule: Processed molecule parquet
            - chemical_probes: Chemical probes parquet
            - mechanism_of_action: Mechanism of action parquet
            - clinical_report: Clinical report parquet from clinical_report step
            - disease: Disease/EFO parquet
        destination: Dictionary with paths to:
            - output: Path to write the output parquet file.
            - excluded: Path to write excluded clinical reports that failed QC.
        settings: Custom settings with:
            - invalid_clinical_report_qc: List of QC reason strings to exclude.
        properties: Spark configuration options.
    """
    spark = Session(app_name='drug_molecule', properties=properties)

    logger.info(f'Loading data from {source}')
    clinical_report_df = spark.load_data(source['clinical_report'])

    # Filter out clinical reports that fail QC (only if qualityControls column exists)
    invalid_qc_reasons = settings.get('invalid_clinical_report_qc', [])

    if invalid_qc_reasons and 'qualityControls' in clinical_report_df.columns:
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

    molecule_df = spark.load_data(source['molecule'])
    chemical_probes_df = spark.load_data(source['chemical_probes'])
    mechanism_df = spark.load_data(source['mechanism_of_action'])
    disease_df = spark.load_data(source['disease'])

    logger.info('Processing drug index')
    output_df = process_drug_index(
        molecule_df,
        chemical_probes_df,
        mechanism_df,
        clinical_report_df,
        disease_df,
    )

    logger.info(f'Writing drug index to {destination["output"]}')
    output_df.write.mode('overwrite').parquet(destination['output'])


def process_drug_index(
    molecule: DataFrame,
    chemical_probes: DataFrame,
    mechanism_of_action: DataFrame,
    clinical_report: DataFrame,
    disease: DataFrame,
) -> DataFrame:
    """Process and combine all drug data into the final index.

    Args:
        molecule: Processed molecule data.
        chemical_probes: Chemical probes data.
        mechanism_of_action: Mechanism of action data.
        clinical_report: Clinical report data with drugs, diseases, and clinicalStage.
        disease: Disease/EFO data for indication mapping.

    Returns:
        Final drug index DataFrame.
    """
    # Compute overall max clinical stage per drug
    max_phase = _compute_max_phase_per_drug(clinical_report)

    # Compute per-indication max stage for description generation
    indications = _process_clinical_report_indications(clinical_report, disease)

    # Get all chemical probe drug IDs (for is_drug filter)
    probe_drug_ids = (
        chemical_probes
        .filter(f.col('drugId').isNotNull())  # ty:ignore[missing-argument]
        .select(f.col('drugId').alias('chemicalProbeDrugId'))
        .distinct()
    )

    # Get probe compound IDs grouped per drug (for cross-references)
    probe_xrefs = (
        chemical_probes
        .filter(f.col('drugId').isNotNull())  # ty:ignore[missing-argument]
        .groupBy(f.col('drugId').alias('_probeXrefDrugId'))
        .agg(f.collect_set('drugId').alias('_probeIds'))
    )

    # Get molecules with mechanism of action
    has_mechanism = (
        mechanism_of_action
        .select(f.explode(f.col('chemblIds')).alias('id'))
        .distinct()
        .withColumn('hasMechanismOfAction', f.lit(True))
    )

    # Join all data together
    drug_df = (
        molecule
        .join(max_phase, on='id', how='left_outer')
        .join(indications, on='id', how='left_outer')
        .join(probe_drug_ids, molecule['id'] == probe_drug_ids['chemicalProbeDrugId'], 'left_outer')
        .join(probe_xrefs, molecule['id'] == probe_xrefs['_probeXrefDrugId'], 'left_outer')
        .join(has_mechanism, on='id', how='left_outer')
    )

    # Append probes&drugs cross-reference when the molecule is a chemical probe
    drug_df = drug_df.withColumn(
        'crossReferences',
        f.when(
            f.col('_probeIds').isNotNull(),  # ty:ignore[missing-argument]
            f.concat(
                f.coalesce(f.col('crossReferences'), f.array()),
                f.array(f.struct(f.lit('probes&drugs').alias('source'), f.col('_probeIds').alias('ids'))),
            ),
        ).otherwise(f.col('crossReferences')),
    )

    # Filter to only include "drugs" - molecules that have:
    # - a drugbank cross-reference, OR
    # - are present in clinical reports (maximumClinicalStage is not null), OR
    # - mechanism of action, OR
    # - are a chemical probe
    is_drug = (
        f.expr("array_contains(transform(crossReferences, x -> x.source), 'drugbank')")
        | f.col('maximumClinicalStage').isNotNull()  # ty:ignore[missing-argument]
        | f.col('hasMechanismOfAction').isNotNull()  # ty:ignore[missing-argument]
        | f.col('chemicalProbeDrugId').isNotNull()  # ty:ignore[missing-argument]
    )

    # Add description
    drug_df = _add_description(drug_df)

    # Filter and cleanup
    return (
        drug_df
        .filter(is_drug)
        .withColumn(
            'maximumClinicalStage',
            f.coalesce(f.col('maximumClinicalStage'), f.lit(STAGE_DISPLAY_NAMES['UNKNOWN'])),
        )
        .drop(
            'chemicalProbeDrugId',
            '_probeXrefDrugId',
            '_probeIds',
            'hasMechanismOfAction',
            'indications',
        )
        .transform(_cleanup)
        .dropDuplicates(['id'])
    )


def _compute_max_phase_per_drug(clinical_report: DataFrame) -> DataFrame:
    """Compute the overall maximum clinical stage for each drug across all clinical reports.

    Explodes the drugs array, maps WITHDRAWAL/PHASE_4 to APPROVAL, ranks stages,
    and returns the best (most advanced) stage per drug.

    Args:
        clinical_report: Clinical report DataFrame with drugs array and clinicalStage.

    Returns:
        DataFrame with columns: id (drugId), maximumClinicalStage (string display name).
    """
    # Build mapping expression for stage normalization
    stage_mapping = f.create_map(*[
        item for pair in STAGE_FOR_MAX_MAPPING.items() for item in (f.lit(pair[0]), f.lit(pair[1]))
    ])
    # Build mapping expression for stage -> rank
    rank_mapping = f.create_map(*[
        item for pair in CLINICAL_STAGE_RANKS.items() for item in (f.lit(pair[0]), f.lit(pair[1]))
    ])
    # Build mapping expression for rank -> display name
    rank_to_display = f.create_map(*[
        item for rank, stage in RANK_TO_STAGE.items() for item in (f.lit(rank), f.lit(STAGE_DISPLAY_NAMES[stage]))
    ])

    return (
        clinical_report
        .select(
            f.explode(f.col('drugs')).alias('drug'),
            f.col('clinicalStage'),
        )
        .select(
            f.col('drug.drugId').alias('id'),
            f.col('clinicalStage'),
        )
        .filter(f.col('id').isNotNull())  # ty:ignore[missing-argument]
        # Normalize: map WITHDRAWAL/PHASE_4 -> APPROVAL
        .withColumn(
            'normalizedStage',
            f.coalesce(stage_mapping[f.col('clinicalStage')], f.col('clinicalStage')),
        )
        # Map stage to rank
        .withColumn(
            'stageRank',
            f.coalesce(rank_mapping[f.col('normalizedStage')], f.lit(CLINICAL_STAGE_RANKS['UNKNOWN'])),
        )
        # Group by drug, take minimum rank (= best stage)
        .groupBy('id')
        .agg(f.min('stageRank').alias('bestRank'))
        # Map rank to display name
        .withColumn('maximumClinicalStage', rank_to_display[f.col('bestRank')])
        .drop('bestRank')
    )


def _process_clinical_report_indications(
    clinical_report: DataFrame,
    disease: DataFrame,
) -> DataFrame:
    """Process clinical reports to extract per-drug, per-indication max stage.

    Explodes both drugs and diseases arrays, computes the best clinical stage
    per (drugId, diseaseId) pair, joins with disease data for names, and
    aggregates into an array of indication structs per drug.

    Args:
        clinical_report: Clinical report DataFrame with drugs, diseases, clinicalStage.
        disease: Disease/EFO DataFrame with id and name columns.

    Returns:
        DataFrame with columns: id (drugId), indications (array of structs).
    """
    # Build mapping expressions
    stage_mapping = f.create_map(*[
        item for pair in STAGE_FOR_MAX_MAPPING.items() for item in (f.lit(pair[0]), f.lit(pair[1]))
    ])
    rank_mapping = f.create_map(*[
        item for pair in CLINICAL_STAGE_RANKS.items() for item in (f.lit(pair[0]), f.lit(pair[1]))
    ])
    rank_to_display = f.create_map(*[
        item for rank, stage in RANK_TO_STAGE.items() for item in (f.lit(rank), f.lit(STAGE_DISPLAY_NAMES[stage]))
    ])

    # Explode drugs and diseases, filter to rows with both IDs
    exploded = (
        clinical_report
        .select(
            f.explode(f.col('drugs')).alias('drug'),
            f.explode(f.col('diseases')).alias('disease'),
            f.col('clinicalStage'),
        )
        .select(
            f.col('drug.drugId').alias('drugId'),
            f.col('disease.diseaseId').alias('diseaseId'),
            f.col('clinicalStage'),
        )
        .filter(f.col('drugId').isNotNull() & f.col('diseaseId').isNotNull())  # ty:ignore[missing-argument]
        # Normalize stage
        .withColumn(
            'normalizedStage',
            f.coalesce(stage_mapping[f.col('clinicalStage')], f.col('clinicalStage')),
        )
        # Map to rank
        .withColumn(
            'stageRank',
            f.coalesce(rank_mapping[f.col('normalizedStage')], f.lit(CLINICAL_STAGE_RANKS['UNKNOWN'])),
        )
    )

    # Best stage per (drugId, diseaseId)
    per_indication = (
        exploded
        .groupBy('drugId', 'diseaseId')
        .agg(f.min('stageRank').alias('bestRank'))
        .withColumn('maxClinicalStage', rank_to_display[f.col('bestRank')])
        .drop('bestRank')
    )

    # Join with disease to get efoName
    disease_names = disease.select(
        f.col('id').alias('diseaseId'),
        f.trim(f.lower(f.col('name'))).alias('efoName'),
    )

    per_indication_with_names = per_indication.join(disease_names, on='diseaseId', how='left')

    # Group by drugId -> array of indication structs
    return (
        per_indication_with_names
        .withColumn(
            'indication',
            f.struct(
                f.col('diseaseId').alias('disease'),
                f.col('efoName'),
                f.col('maxClinicalStage'),
            ),
        )
        .groupBy(f.col('drugId').alias('id'))
        .agg(f.collect_set('indication').alias('indications'))
    )


def _add_description(df: DataFrame) -> DataFrame:
    """Add human-readable description to drug data.

    Args:
        df: Drug DataFrame with required columns.

    Returns:
        DataFrame with description column added.
    """
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession()
    if spark is None:
        msg = 'No active SparkSession found'
        raise RuntimeError(msg)
    spark.udf.register('generate_description', _generate_description, StringType())

    # Prepare indication data for description
    df = df.withColumn(
        '_indication_stages',
        f.expr('transform(coalesce(indications, array()), x -> x.maxClinicalStage)'),
    ).withColumn(
        '_indication_labels',
        f.expr('transform(coalesce(indications, array()), x -> x.efoName)'),
    )

    # Generate description using UDF
    df = df.withColumn(
        'description',
        f.expr(
            """generate_description(
            drugType,
            maximumClinicalStage,
            _indication_stages,
            _indication_labels
        )"""
        ),
    )

    return df.drop('_indication_stages', '_indication_labels')


def _generate_description(
    drug_type: str | None,
    max_phase: str | None,
    indication_stages: list[str] | None,
    indication_labels: list[str] | None,
) -> str:
    """Generate a human-readable description of a drug.

    Args:
        drug_type: Type of drug (e.g., "Small molecule").
        max_phase: Maximum clinical stage as a display name (e.g., "approved").
        indication_stages: List of per-indication max clinical stage display names.
        indication_labels: List of indication disease names.

    Returns:
        Human-readable description string.
    """
    if drug_type is None:
        drug_type = 'Unknown'

    approved_display = STAGE_DISPLAY_NAMES['APPROVAL']

    main_note = f'{drug_type.capitalize()} drug'

    # Clinical phase
    phase_str = ''
    if max_phase is not None:
        label_count = len(indication_labels) if indication_labels else 0
        multi_indication = ' (across all indications)' if label_count > 1 else ''
        phase_str = f' with a maximum clinical stage of {max_phase}{multi_indication}'

    # Process indications
    indication_str = ''
    if indication_stages is not None and indication_labels is not None:
        indications = list(zip(indication_stages, indication_labels, strict=False))
        indications = [(stage, label) for stage, label in indications if stage is not None and label is not None]
        indications = list(set(indications))

        approved = [label for stage, label in indications if stage == approved_display]
        investigational_count = sum(1 for stage, _ in indications if stage != approved_display)

        if approved and not investigational_count:
            if len(approved) <= 2:
                indication_str = f' and is indicated for {_join_semantic(approved)}'
            else:
                indication_str = f' and has {len(approved)} approved indications'
        elif not approved and investigational_count:
            s = 's' if investigational_count > 1 else ''
            indication_str = f' and has {investigational_count} investigational indication{s}'
        elif approved and investigational_count:
            s = 's' if investigational_count > 1 else ''
            if len(approved) <= 2:
                approved_str = _join_semantic(approved)
                indication_str = (
                    f' and is indicated for {approved_str}'
                    f' and has {investigational_count} investigational indication{s}'
                )
            else:
                indication_str = (
                    f' and has {len(approved)} approved and {investigational_count} investigational indication{s}'
                )

    return f'{main_note}{phase_str}{indication_str}.'


def _join_semantic(items: list[str]) -> str:
    """Join items in a grammatically correct way.

    Args:
        items: List of strings to join.

    Returns:
        Joined string (e.g., "a, b and c").
    """
    if not items:
        return ''
    if len(items) == 1:
        return items[0]
    return f'{", ".join(items[:-1])} and {items[-1]}'


def _cleanup(df: DataFrame) -> DataFrame:
    """Ensure array columns have empty arrays instead of nulls.

    Args:
        df: DataFrame to clean up.

    Returns:
        Cleaned DataFrame.
    """
    for column in ['tradeNames', 'synonyms']:
        if column in df.columns:
            df = df.withColumn(column, f.coalesce(f.col(column), f.array()))
    return df
