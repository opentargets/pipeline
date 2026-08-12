"""Mechanism of Action processing for drugs.

Prepares the mechanism of action section of the drug object by joining raw
ChEMBL mechanism, target, and reference tables with target/gene information.
"""

from typing import Any

import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import DataFrame

from pts.pyspark.common.session import Session
from pts.pyspark.drug_utils.chembl_ids import chembl_ids as _chembl_ids


def drug_mechanism_of_action(
    source: dict[str, str],
    destination: str,
    _settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Process mechanism of action data from ChEMBL.

    Args:
        source: Dictionary with paths to:
            - drug_mechanism: Raw ChEMBL drug_mechanism parquet.
            - mechanism_refs: Raw ChEMBL mechanism_refs parquet.
            - molecule_dictionary: Raw ChEMBL molecule_dictionary parquet.
            - molecule_hierarchy: Raw ChEMBL molecule_hierarchy parquet.
            - target_dictionary: Raw ChEMBL target_dictionary parquet.
            - target_components: Raw ChEMBL target_components parquet.
            - component_sequences: Raw ChEMBL component_sequences parquet.
            - target: Target parquet (gene data).
        destination: Path to write the output parquet file.
        _settings: Custom settings (not used).
        properties: Spark configuration options.
    """
    spark = Session(app_name='drug_mechanism_of_action', properties=properties)

    logger.info(f'Loading data from {source}')
    drug_mechanism = spark.load_data(source['drug_mechanism'])
    mechanism_refs = spark.load_data(source['mechanism_refs'])
    molecule_dictionary = spark.load_data(source['molecule_dictionary'])
    molecule_hierarchy = spark.load_data(source['molecule_hierarchy'])
    target_dictionary = spark.load_data(source['target_dictionary'])
    target_components = spark.load_data(source['target_components'])
    component_sequences = spark.load_data(source['component_sequences'])
    gene_df = spark.load_data(source['target'])

    logger.info('Processing mechanisms of action')
    output_df = process_mechanism_of_action(
        drug_mechanism,
        mechanism_refs,
        molecule_dictionary,
        molecule_hierarchy,
        target_dictionary,
        target_components,
        component_sequences,
        gene_df,
    )

    logger.info(f'Writing mechanism of action to {destination}')
    output_df.write.parquet(destination, mode='overwrite')


def process_mechanism_of_action(
    drug_mechanism: DataFrame,
    mechanism_refs: DataFrame,
    molecule_dictionary: DataFrame,
    molecule_hierarchy: DataFrame,
    target_dictionary: DataFrame,
    target_components: DataFrame,
    component_sequences: DataFrame,
    gene_df: DataFrame,
) -> DataFrame:
    """Build mechanisms of action by joining raw ChEMBL tables with target and gene data.

    Args:
        drug_mechanism: Raw ChEMBL drug_mechanism table.
        mechanism_refs: Raw ChEMBL mechanism_refs table.
        molecule_dictionary: Raw ChEMBL molecule_dictionary table.
        molecule_hierarchy: Raw ChEMBL molecule_hierarchy table.
        target_dictionary: Raw ChEMBL target_dictionary table.
        target_components: Raw ChEMBL target_components table.
        component_sequences: Raw ChEMBL component_sequences table.
        gene_df: Gene parquet data from the target step.

    Returns:
        Processed mechanism of action DataFrame.
    """
    ids = _chembl_ids(drug_mechanism, molecule_dictionary, molecule_hierarchy, key='mec_id')
    mechanism_refs_agg = mechanism_refs.groupBy('mec_id').agg(
        f.collect_list(
            f.struct(
                f.col('ref_type'),
                f.col('ref_id'),
                f.col('ref_url'),
            )
        ).alias('mechanism_refs')
    )

    mechanism = (
        _with_target_chembl_id(drug_mechanism, target_dictionary)
        .join(molecule_dictionary.select('molregno', f.col('chembl_id').alias('id')), on='molregno', how='left')
        .join(ids, on='mec_id', how='left')
        .join(mechanism_refs_agg, on='mec_id', how='left')
        .withColumnRenamed('mechanism_of_action', 'mechanismOfAction')
        .withColumnRenamed('action_type', 'actionType')
        .drop('mec_id', 'molregno')
    )

    references = _chembl_mechanism_references(mechanism)
    target = _chembl_target(target_dictionary, target_components, component_sequences, gene_df)

    result = (
        mechanism
        .join(references, on='id', how='outer')
        .join(target, on='target_chembl_id', how='outer')
        .drop('mechanism_refs', 'record_id', 'target_chembl_id', 'id')
        .filter(
            """
            mechanismOfAction is not null
            and (targets is not null or targetName is not null)
            and chemblIds is not null and size(chemblIds) > 0
            """
        )
    )

    return _consolidate_duplicate_references(result)


def _with_target_chembl_id(mechanism: DataFrame, target_dictionary: DataFrame) -> DataFrame:
    """Resolve each mechanism's `tid` to a `target_chembl_id`.

    Uses a left join, so a null or unmatched `tid` yields a null `target_chembl_id`
    rather than dropping the mechanism row.

    Args:
        mechanism: Raw ChEMBL drug_mechanism table.
        target_dictionary: Raw ChEMBL target_dictionary table.

    Returns:
        `mechanism` with `tid` replaced by `target_chembl_id`.
    """
    return (
        mechanism
        .join(
            target_dictionary.select('tid', f.col('chembl_id').alias('target_chembl_id')),
            on='tid',
            how='left',
        )
        .drop('tid')
    )


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


def _chembl_target(
    target_dictionary: DataFrame,
    target_components: DataFrame,
    component_sequences: DataFrame,
    gene_df: DataFrame,
) -> DataFrame:
    """Process ChEMBL target data and join with gene information.

    Args:
        target_dictionary: Raw ChEMBL target_dictionary table.
        target_components: Raw ChEMBL target_components table.
        component_sequences: Raw ChEMBL component_sequences table.
        gene_df: Gene parquet data with proteinIds.

    Returns:
        DataFrame with target_chembl_id, targetName, targetType, and targets.
    """
    target_components_flat = (
        target_components
        .join(component_sequences, on='component_id', how='inner')
        .join(target_dictionary, on='tid', how='inner')
        .filter(f.col('accession').isNotNull())
        .select(
            f.col('pref_name').alias('targetName'),
            f.col('accession').alias('uniprot_id'),
            f.lower(f.col('target_type')).alias('targetType'),
            f.col('chembl_id').alias('target_chembl_id'),
        )
    )

    # Get gene IDs from gene data - explode proteinIds
    genes = gene_df.select(
        f.col('id').alias('geneId'),
        f.array_union(
            f.coalesce(f.col('uniprot_trembl'), f.array().cast('array<string>')),
            f.coalesce(f.col('uniprot_swissprot'), f.array().cast('array<string>')),
        ).alias('uniprotIds'),
    ).select('geneId', f.explode('uniprotIds').alias('uniprot_id'))

    # Join target with genes on uniprot_id or geneId
    joined = target_components_flat.join(
        genes,
        (target_components_flat['uniprot_id'] == genes['uniprot_id'])
        | (target_components_flat['uniprot_id'] == genes['geneId']),
        how='left_outer',
    )

    return joined.groupBy('target_chembl_id', 'targetName', 'targetType').agg(
        f.array_distinct(f.collect_list('geneId')).alias('targets')
    )


def _consolidate_duplicate_references(df: DataFrame) -> DataFrame:
    """Consolidate mechanism rows that are identical for the same drug.

    ChEMBL propagates the mechanism for their child/salt molecules in
    ``_metadata.all_molecule_chembl_ids`` to include the parent. When the parent has
    the same mechanism, these rows carry identical display information and differ only
    by ``chemblIds`` (and the per-molecule ``id``, already dropped). This step avoids the
    duplication on the mechanism for the parent once the data is exploded
    by ``chemblId`` downstream.
    """
    key_cols = [c for c in df.columns if c not in ('references', 'chemblIds')]
    return df.groupBy(*key_cols).agg(
        f.array_distinct(f.flatten(f.collect_list('chemblIds'))).alias('chemblIds'),
        f.flatten(f.collect_set('references')).alias('references'),
    )
