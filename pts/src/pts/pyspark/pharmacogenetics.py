"""Pharmacogenetics processing.

This module processes ClinPGx pharmacogenetics data:
1. Annotates phenotypes from the curated phenotypes lookup
2. Adds variant IDs from genotype information
3. Maps phenotypes to EFO disease ontology
4. Maps drug names to ChEMBL drug IDs
5. Enriches with isDirectTarget flag using mechanism of action data
"""

import json
from typing import Any

import pyspark.sql.functions as f
from loguru import logger
from otter.storage.synchronous.handle import StorageHandle
from pyspark.sql import DataFrame

from pts.pyspark.common.ontology import add_efo_mapping
from pts.pyspark.common.session import Session


def pharmacogenetics(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Process pharmacogenetics data from ClinPGx.

    Args:
        source: Dictionary with paths to:
            - clinpgx: ClinPGx annotation JSON
            - phenotypes: Phenotypes lookup JSON
            - ontoma_disease_label_lut: OnToma disease label lookup parquet
            - chembl_molecule: ChEMBL molecule parquet for drug ID mapping
            - drug_mechanism_of_action: Mechanism of action parquet for target lookup
        destination: Dictionary with output paths for:
            - associations: Output parquet path
            - unparsed_texts: JSON list of texts missing from the phenotypes lookup
        settings: Settings including partition_count
        properties: Spark configuration options
    """
    spark = Session(app_name='pharmacogenetics', properties=properties)

    logger.info(f'load data from {source}')
    pgx_phenotypes_df = spark.load_data(source['phenotypes'], format='json')
    pgx_df = spark.load_data(source['clinpgx'], format='json')
    chembl_molecule_df = spark.load_data(source['chembl_molecule'])
    moa_df = spark.load_data(source['drug_mechanism_of_action'])

    unparsed_texts = find_unparsed_texts(pgx_df, pgx_phenotypes_df)
    if unparsed_texts:
        logger.warning(f'{len(unparsed_texts)} genotypeAnnotationText values missing from the phenotypes lookup')
    else:
        logger.info('all phenotypes have been parsed')
    logger.info(f'save unparsed texts to {destination["unparsed_texts"]}')
    StorageHandle(destination['unparsed_texts']).write_text(json.dumps(unparsed_texts, indent=2))

    logger.info('overwrite phenotypeText column with parsed phenotypes')
    annotated_pgx_df = annotate_phenotype(pgx_df, pgx_phenotypes_df)

    logger.info('parse variantId')
    pgx_w_variantid_df = add_variantid_column(annotated_pgx_df)
    logger.info('add efo mappings')
    mapped_pgx_df = add_efo_mapping(
        spark=spark.spark,
        evidence_df=pgx_w_variantid_df,
        label_col_name='phenotypeText',
        disease_label_lut_path=source['ontoma_disease_label_lut'],
        id_col_name=None,
    ).withColumnRenamed('diseaseFromSourceMappedId', 'phenotypeFromSourceId')

    logger.info('map drug IDs and enrich with target information')
    enriched_pgx_df = enrich_with_drug_and_target_info(
        mapped_pgx_df,
        chembl_molecule_df,
        moa_df,
    )

    partition_count = settings.get('partition_count')

    logger.info(f'save associations to {destination["associations"]}')
    out = enriched_pgx_df.coalesce(partition_count) if partition_count is not None else enriched_pgx_df
    out.write.parquet(destination['associations'], mode='overwrite')


def find_unparsed_texts(pgx_df: DataFrame, extracted_phenotypes_df: DataFrame) -> list[str]:
    """Return the sorted `genotypeAnnotationText` values absent from the phenotypes lookup.

    A text present in the lookup counts as parsed even if its `phenotypeText` is empty.
    """
    unparsed_texts_df = (
        pgx_df
        .select('genotypeAnnotationText')
        .where(f.col('genotypeAnnotationText').isNotNull())
        .distinct()
        .join(
            extracted_phenotypes_df.select('genotypeAnnotationText').distinct(),
            on='genotypeAnnotationText',
            how='left_anti',
        )
    )
    return sorted(row.genotypeAnnotationText for row in unparsed_texts_df.collect())


def annotate_phenotype(pgx_evidence_df: DataFrame, extracted_phenotypes_df: DataFrame) -> DataFrame:
    """This module overwrites the `phenotypeText`, which comes from ClinPGx directly, but it is usually too verbose.

    Args:
        pgx_evidence_df: Dataframe with the PGx evidence submitted by EVA.
        extracted_phenotypes_df: Dataframe containing the phenotypes extracted from `genotypeAnnotationText`.
    """
    return (
        pgx_evidence_df
        .drop('phenotypeText', 'phenotypeFromSourceId')
        .join(extracted_phenotypes_df, on='genotypeAnnotationText', how='left')
        .withColumn('phenotypeText', f.explode_outer('phenotypeText'))
        .distinct()
    )


def add_variantid_column(input_df: DataFrame) -> DataFrame:
    """Based on the content of the genotypeId column, adds a variantId column to the dataset."""
    return (
        input_df
        # split genotypeId column into chr pos ref alt columns
        .select(
            'genotypeId',
            f.from_csv(
                f.col('genotypeId'),
                'chr string, pos string, ref string, alt string',
                {'sep': '_'},
            ).alias('genotype_split'),
        )
        .select('genotypeId', 'genotype_split.*')
        .toDF('genotypeId', 'chr', 'pos', 'ref', 'alt')
        # split alt column and explode
        .withColumn('alt', f.explode(f.split(f.col('alt'), ',')))
        .filter(~(f.col('ref') == f.col('alt')))
        .select(
            'genotypeId',
            f.concat_ws('_', f.col('chr'), f.col('pos'), f.col('ref'), f.col('alt')).alias('variantId'),
        )
        .join(input_df, on='genotypeId', how='right')
    )


def enrich_with_drug_and_target_info(
    pgx_df: DataFrame,
    molecule_df: DataFrame,
    moa_df: DataFrame,
) -> DataFrame:
    """Enrich pharmacogenetics data with drug IDs and isDirectTarget flag.

    This function:
    1. Explodes the drugs array to process each drug individually
    2. Maps drug names to ChEMBL drug IDs
    3. Determines if the variant target is a direct target of the drug (using mechanism of action)
    4. Groups back by the original row structure

    Args:
        pgx_df: Pharmacogenetics DataFrame with drugs array.
        molecule_df: ChEMBL molecule DataFrame for drug ID mapping.
        moa_df: Mechanism of action DataFrame for target lookup.

    Returns:
        Enriched DataFrame with drugId and isDirectTarget.
    """
    # Get drug name to ID lookup table
    drug_name_lut = _get_drug_name_lut(molecule_df)

    # Get drug to target lookup table from mechanism of action
    drug_target_lut = _get_drug_target_lut(moa_df)

    # Add operational row ID for grouping back later
    pgx_expanded = (
        pgx_df
        .withColumn('_operationalRowId', f.monotonically_increasing_id())
        .withColumn('drug', f.explode('drugs'))
        .withColumn('drugFromSource', f.col('drug.drugFromSource'))
        .drop('drugs')
    )

    # Map drug IDs using the lookup table
    pgx_with_drug_id = _map_drug_id(pgx_expanded, drug_name_lut)

    # Join with drug-target lookup and flag direct targets
    pgx_enriched = (
        pgx_with_drug_id
        .join(drug_target_lut, on='drugId', how='left')
        .withColumn(
            'isDirectTarget',
            f.when(
                f.array_contains(f.col('drugTargetIds'), f.col('targetFromSourceId')),
                f.lit(True),
            ).otherwise(f.lit(False)),
        )
        .drop('drugTargetIds')
        .distinct()
    )

    # Group back by the original row, collecting drugs into an array
    grouping_cols = [
        '_operationalRowId',
        'datasourceId',
        'datasourceVersion',
        'datatypeId',
        'directionality',
        'evidenceLevel',
        'genotype',
        'genotypeAnnotationText',
        'genotypeId',
        'haplotypeFromSourceId',
        'haplotypeId',
        'literature',
        'pgxCategory',
        'phenotypeFromSourceId',
        'phenotypeText',
        'variantAnnotation',
        'studyId',
        'targetFromSourceId',
        'variantFunctionalConsequenceId',
        'variantRsId',
        'variantId',
        'isDirectTarget',
    ]

    # Filter to only include columns that exist in the dataframe
    existing_cols = [c for c in grouping_cols if c in pgx_enriched.columns]

    return (
        pgx_enriched
        .groupBy(*existing_cols)
        .agg(f.collect_list(f.struct(f.col('drugFromSource'), f.col('drugId'))).alias('drugs'))
        .drop('_operationalRowId')
    )


def _get_drug_name_lut(molecule_df: DataFrame) -> DataFrame:
    """Create a lookup table mapping drug names to ChEMBL drug IDs.

    When multiple IDs exist for the same name, selects one deterministically
    by sorting and taking the last one (to match Scala behavior).

    Args:
        molecule_df: ChEMBL molecule DataFrame with id and name columns.

    Returns:
        DataFrame with drugFromSource and drugId columns.
    """
    return (
        molecule_df
        .select(f.col('id'), f.lower(f.col('name')).alias('drugFromSource'))
        .filter(f.col('drugFromSource').isNotNull())
        .groupBy('drugFromSource')
        .agg(f.collect_set('id').alias('ids'))
        .select(
            f.col('drugFromSource'),
            f.element_at(f.sort_array(f.col('ids'), asc=False), 1).alias('drugId'),
        )
    )


def _map_drug_id(pgx_df: DataFrame, drug_name_lut: DataFrame) -> DataFrame:
    """Map drug names to ChEMBL drug IDs.

    Args:
        pgx_df: Pharmacogenetics DataFrame with drugFromSource column.
        drug_name_lut: Lookup table with drugFromSource and drugId columns.

    Returns:
        DataFrame with drugId column added.
    """
    return pgx_df.withColumn('drugFromSource', f.lower(f.col('drugFromSource'))).join(
        drug_name_lut, on='drugFromSource', how='left'
    )


def _get_drug_target_lut(moa_df: DataFrame) -> DataFrame:
    """Create a lookup table mapping drug IDs to their target IDs.

    Extracts target information from the mechanism of action data,
    exploding chemblIds to get all drug IDs associated with each mechanism.

    Args:
        moa_df: Mechanism of action DataFrame with chemblIds and targets columns.

    Returns:
        DataFrame with drugId and drugTargetIds columns.
    """
    return (
        moa_df
        .filter(f.col('targets').isNotNull() & (f.size(f.col('targets')) >= 1))
        .select(
            f.explode(f.col('chemblIds')).alias('drugId'),
            f.col('targets'),
        )
        .groupBy('drugId')
        .agg(f.array_distinct(f.flatten(f.collect_list('targets'))).alias('drugTargetIds'))
    )
