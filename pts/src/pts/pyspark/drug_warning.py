"""Drug warnings as produced by ChEMBL.

Drug warnings are manually curated by ChEMBL according to the methodology outlined
in https://pubs.acs.org/doi/pdf/10.1021/acs.chemrestox.0c00296
"""

from typing import Any

from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from pts.pyspark.common.session import Session


def drug_warning(
    source: dict[str, str],
    destination: str,
    _settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Transform ChEMBL drug warnings into the Open Targets format.

    Args:
        source: Dictionary with paths to:
            - drug_warning: Raw ChEMBL drug warnings parquet.
            - warning_refs: Raw ChEMBL warning references parquet.
            - molecule_dictionary: Raw ChEMBL molecule dictionary parquet.
            - molecule_hierarchy: Raw ChEMBL molecule hierarchy parquet.
        destination: Path to write the output parquet file.
        _settings: Custom settings (not used).
        properties: Spark configuration options.
    """
    spark = Session(app_name='drug_warning', properties=properties)

    logger.info(f'Loading data from {source}')
    warnings_df = spark.load_data(source['drug_warning'])
    refs_df = spark.load_data(source['warning_refs'])
    molecules_df = spark.load_data(source['molecule_dictionary'])
    hierarchy_df = spark.load_data(source['molecule_hierarchy'])

    logger.info('Preparing drug warnings')
    output_df = process_drug_warnings(warnings_df, refs_df, molecules_df, hierarchy_df)

    logger.info(f'Writing drug warnings to {destination}')
    output_df.write.parquet(destination, mode='overwrite')


def process_drug_warnings(
    warnings: DataFrame,
    refs: DataFrame,
    molecules: DataFrame,
    hierarchy: DataFrame,
) -> DataFrame:
    """Build drug warnings by joining raw ChEMBL tables.

    Args:
        warnings: Raw ChEMBL drug_warning table.
        refs: Raw ChEMBL warning_refs table.
        molecules: Raw ChEMBL molecule_dictionary table.
        hierarchy: Raw ChEMBL molecule_hierarchy table.

    Returns:
        One row per warning_id, in the Open Targets output format.
    """
    chembl_ids = _chembl_ids(warnings, molecules, hierarchy)
    references = _references(refs)

    # `warning_year` narrows int64 (BIGINT) -> int32 (postgres `integer`) versus the
    # released `year` column, the same ES-to-parquet type shift as the `protein_class_id`
    # cast at target.py's `_build_protein_classification`. Resolved the other way here:
    # the narrowing is accepted uncast because the values are identical, not cast back.
    return (
        warnings.join(chembl_ids, on='warning_id', how='left')
        .join(references, on='warning_id', how='left')
        .withColumn('references', f.coalesce(f.col('references'), f.array()))
        .selectExpr(
            'chemblIds',
            'warning_class as toxicityClass',
            'warning_country as country',
            'warning_description as description',
            'warning_id as id',
            'references',
            'warning_type as warningType',
            'warning_year as year',
            'efo_term as efoTerm',
            'efo_id as efoId',
            'efo_id_for_warning_class as efoIdForWarningClass',
        )
    )


def _chembl_ids(warnings: DataFrame, molecules: DataFrame, hierarchy: DataFrame) -> DataFrame:
    """Resolve the deduplicated {molecule, parent molecule} ChEMBL id pair per warning.

    Replaces the old `_metadata.all_molecule_chembl_ids` field from the Elasticsearch
    document, which was always exactly this deduplicated pair.

    Args:
        warnings: Raw ChEMBL drug_warning table.
        molecules: Raw ChEMBL molecule_dictionary table.
        hierarchy: Raw ChEMBL molecule_hierarchy table.

    Returns:
        DataFrame with warning_id and chemblIds columns.
    """
    parent = (
        hierarchy.join(
            molecules.withColumnRenamed('chembl_id', 'parent_chembl_id'),
            hierarchy['parent_molregno'] == molecules['molregno'],
            'left',
        ).select(hierarchy['molregno'], 'parent_chembl_id')
    )
    return (
        warnings.select('warning_id', 'molregno')
        .join(molecules, on='molregno', how='left')
        .join(parent, on='molregno', how='left')
        .select(
            'warning_id',
            f.array_distinct(f.array_compact(f.array(f.col('chembl_id'), f.col('parent_chembl_id')))).alias(
                'chemblIds'
            ),
        )
    )


def _references(refs: DataFrame) -> DataFrame:
    """Aggregate warning references into a struct array per warning.

    Args:
        refs: Raw ChEMBL warning_refs table.

    Returns:
        DataFrame with warning_id and references columns.
    """
    return refs.groupBy('warning_id').agg(
        f.collect_list(
            f.struct(
                f.col('ref_id').alias('id'),
                f.col('ref_type').alias('source'),
                f.col('ref_url').alias('url'),
            )
        ).alias('references')
    )
