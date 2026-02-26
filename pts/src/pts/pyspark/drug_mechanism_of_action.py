"""Mechanism of Action processing for drugs.

Prepares the mechanism of action section of the drug object by joining
ChEMBL mechanism data with target and gene information.
"""

from typing import Any

import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import DataFrame

from pts.pyspark.common.session import Session


def drug_mechanism_of_action(
    source: dict[str, str],
    destination: str,
    _settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Process mechanism of action data from ChEMBL.

    Args:
        source: Dictionary with paths to:
            - chembl_mechanism: ChEMBL mechanism JSONL
            - chembl_target: ChEMBL target JSONL
            - target: Target parquet (gene data)
        destination: Path to write the output parquet file.
        _settings: Custom settings (not used).
        properties: Spark configuration options.
    """
    spark = Session(app_name='drug_mechanism_of_action', properties=properties)

    logger.info(f'Loading data from {source}')
    mechanism_df = spark.load_data(source['chembl_mechanism'], format='json')
    target_df = spark.load_data(source['chembl_target'], format='json')
    gene_df = spark.load_data(source['target'])

    logger.info('Processing mechanisms of action')
    output_df = process_mechanism_of_action(mechanism_df, target_df, gene_df)

    logger.info(f'Writing mechanism of action to {destination}')
    output_df.write.parquet(destination, mode='overwrite')


def process_mechanism_of_action(
    mechanism_df: DataFrame,
    target_df: DataFrame,
    gene_df: DataFrame,
) -> DataFrame:
    """Process mechanism of action by joining mechanism, target, and gene data.

    Args:
        mechanism_df: Raw ChEMBL mechanism data.
        target_df: Raw ChEMBL target data.
        gene_df: Gene parquet data from target step.

    Returns:
        Processed mechanism of action DataFrame.
    """
    mechanism = (
        mechanism_df
        .withColumnRenamed('molecule_chembl_id', 'id')
        .withColumnRenamed('mechanism_of_action', 'mechanismOfAction')
        .withColumnRenamed('action_type', 'actionType')
        .withColumn('chemblIds', f.col('_metadata.all_molecule_chembl_ids'))
        .drop('_metadata', 'parent_molecule_chembl_id')
    )

    references = _chembl_mechanism_references(mechanism)
    target = _chembl_target(target_df, gene_df)

    result = (
        mechanism
        .join(references, on='id', how='outer')
        .join(target, on='target_chembl_id', how='outer')
        .drop('mechanism_refs', 'record_id', 'target_chembl_id')
        .filter(
            """
            mechanismOfAction is not null
            and (targets is not null or targetName is not null)
            and chemblIds is not null and size(chemblIds) > 0
            """
        )
        .drop('id')
        .distinct()
    )

    return _consolidate_duplicate_references(result)


def _chembl_mechanism_references(df: DataFrame) -> DataFrame:
    """Extract and structure references from mechanism data.

    Args:
        df: Mechanism DataFrame with id and mechanism_refs columns.

    Returns:
        DataFrame with id and references columns.
    """
    return (
        df
        .select(f.col('id'), f.explode('mechanism_refs').alias('ref'))
        .groupBy('id', f.col('ref.ref_type').alias('ref_type'))
        .agg(
            f.collect_list('ref.ref_id').alias('ref_id'),
            f.collect_list('ref.ref_url').alias('ref_url'),
        )
        .withColumn(
            'references',
            f.struct(
                f.col('ref_type').alias('source'),
                f.col('ref_id').alias('ids'),
                f.col('ref_url').alias('urls'),
            ),
        )
        .groupBy('id')
        .agg(f.collect_list('references').alias('references'))
    )


def _chembl_target(target_df: DataFrame, gene_df: DataFrame) -> DataFrame:
    """Process ChEMBL target data and join with gene information.

    Args:
        target_df: Raw ChEMBL target data.
        gene_df: Gene parquet data with proteinIds.

    Returns:
        DataFrame with target_chembl_id, targetName, targetType, and targets.
    """
    # Process target data - explode target_components and filter
    target = (
        target_df
        .withColumn('target_components', f.explode('target_components'))
        .filter(f.col('target_components.accession').isNotNull())  # ty:ignore[missing-argument]
        .select(
            f.col('pref_name').alias('targetName'),
            f.col('target_components.accession').alias('uniprot_id'),
            f.lower(f.col('target_type')).alias('targetType'),
            f.col('target_chembl_id'),
        )
    )

    # Get gene IDs from gene data - explode proteinIds
    genes = gene_df.select(
        f.col('id').alias('geneId'),
        f.explode('proteinIds.id').alias('uniprot_id'),
    )

    # Join target with genes on uniprot_id or geneId
    joined = target.join(
        genes,
        (target['uniprot_id'] == genes['uniprot_id']) | (target['uniprot_id'] == genes['geneId']),
        how='left_outer',
    )

    return joined.groupBy('target_chembl_id', 'targetName', 'targetType').agg(
        f.array_distinct(f.collect_list('geneId')).alias('targets')
    )


def _consolidate_duplicate_references(df: DataFrame) -> DataFrame:
    """Consolidate duplicate rows by merging their references.

    Args:
        df: DataFrame with potential duplicate rows.

    Returns:
        DataFrame with consolidated references.
    """
    cols = [c for c in df.columns if c != 'references']
    return (
        df.groupBy(*cols).agg(f.collect_set('references').alias('r')).withColumn('references', f.flatten('r')).drop('r')
    )
