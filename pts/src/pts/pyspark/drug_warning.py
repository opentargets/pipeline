"""Drug warnings as produced by ChEMBL.

Drug warnings are manually curated by ChEMBL according to the methodology outlined
in https://pubs.acs.org/doi/pdf/10.1021/acs.chemrestox.0c00296
"""

from typing import Any

from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from pts.pyspark.common.session import Session
from pts.pyspark.drug_utils.chembl_ids import chembl_ids as _chembl_ids


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

    logger.info('deduplicate warnings that resolve to the same drug')
    output_df = _deduplicate_warnings(output_df)

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
    ids = _chembl_ids(warnings, molecules, hierarchy, key='warning_id')
    references = _references(refs)

    # `warning_year` narrows int64 (BIGINT) -> int32 (postgres `integer`) versus the
    # released `year` column, the same ES-to-parquet type shift as the `protein_class_id`
    # cast at target.py's `_build_protein_classification`. Resolved the other way here:
    # the narrowing is accepted uncast because the values are identical, not cast back.
    return (
        warnings.join(ids, on='warning_id', how='left')
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


def _deduplicate_warnings(df: DataFrame) -> DataFrame:
    """Merge warnings that carry identical information for the same drug.

    ChEMBL propagates the warning for their child/salt molecules in
    ``_metadata.all_molecule_chembl_ids`` to include the parent. When the parent
    has the same warning, these rows carry identical display information and differ only
    by ``chemblIds`` (and the per-molecule ``id``, already dropped). This step avoids the
    duplication on the warning for the parent once the data is exploded
    by ``chemblId`` downstream.

    Group on every display field, union the ``chemblIds`` (so the merged warning
    still reaches every molecule it applied to) and keep the lowest ``id``.``references``
    stays in the grouping key: two warnings with genuinely different references are not
    duplicates and must not be merged.
    """
    group_cols = [c for c in df.columns if c not in ('id', 'chemblIds')]
    return df.groupBy(*group_cols).agg(
        f.min('id').alias('id'),
        f.array_distinct(f.flatten(f.collect_list('chemblIds'))).alias('chemblIds'),
    ).select(df.columns)
