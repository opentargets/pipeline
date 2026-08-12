"""Drug warnings as produced by ChEMBL.

Drug warnings are manually curated by ChEMBL according to the methodology outlined
in https://pubs.acs.org/doi/pdf/10.1021/acs.chemrestox.0c00296
"""

from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.postgres import read_dump_tables
from pts.transformers.utils import chembl_ids as _chembl_ids

SCHEMA_NAME = 'public'
"""Schema the ChEMBL tables live in inside the restored dump."""

TABLES = {
    'drug_warning': [
        'warning_id', 'molregno', 'warning_type', 'warning_class', 'warning_country',
        'warning_description', 'warning_year', 'efo_term', 'efo_id', 'efo_id_for_warning_class',
    ],
    'warning_refs': ['warnref_id', 'warning_id', 'ref_type', 'ref_id', 'ref_url'],
    'molecule_dictionary': ['molregno', 'chembl_id'],
    'molecule_hierarchy': ['molregno', 'parent_molregno'],
}
"""ChEMBL tables and columns this step needs, restored from the dump."""


def drug_warning(
    source: Path,
    destination: Path,
    _settings: dict[str, Any],
    _config: Config,
) -> None:
    """Transform ChEMBL drug warnings into the Open Targets format.

    Args:
        source: Path to the ChEMBL ``pg_dump`` archive.
        destination: Path to write the output parquet file.
        _settings: Custom settings (not used).
        _config: Config object (not used).
    """
    logger.info(f'Restoring {list(TABLES)} from {source}')
    tables = read_dump_tables(str(source), TABLES, schema_name=SCHEMA_NAME)

    logger.info('Preparing drug warnings')
    output_df = process_drug_warnings(
        tables['drug_warning'], tables['warning_refs'], tables['molecule_dictionary'], tables['molecule_hierarchy']
    )

    logger.info('deduplicate warnings that resolve to the same drug')
    output_df = _deduplicate_warnings(output_df)

    logger.info(f'Writing drug warnings to {destination}')
    output_df.write_parquet(destination, mkdir=True)


def process_drug_warnings(
    warnings: pl.DataFrame,
    refs: pl.DataFrame,
    molecules: pl.DataFrame,
    hierarchy: pl.DataFrame,
) -> pl.DataFrame:
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

    # `warning_year` is int32 (postgres `integer`), narrower than the released `year`
    # column's int64 (BIGINT), the same ES-to-parquet type shift as the `protein_class_id`
    # cast at target.py's `_build_protein_classification`. Resolved the other way here:
    # the narrowing is accepted uncast because the values are identical, not cast back.
    return (
        warnings.join(ids, on='warning_id', how='left')
        .join(references, on='warning_id', how='left')
        .with_columns(pl.col('references').fill_null([]))
        .select(
            'chemblIds',
            pl.col('warning_class').alias('toxicityClass'),
            pl.col('warning_country').alias('country'),
            pl.col('warning_description').alias('description'),
            pl.col('warning_id').alias('id'),
            'references',
            pl.col('warning_type').alias('warningType'),
            pl.col('warning_year').alias('year'),
            pl.col('efo_term').alias('efoTerm'),
            pl.col('efo_id').alias('efoId'),
            pl.col('efo_id_for_warning_class').alias('efoIdForWarningClass'),
        )
    )


def _references(refs: pl.DataFrame) -> pl.DataFrame:
    """Aggregate warning references into a struct array per warning.

    Args:
        refs: Raw ChEMBL warning_refs table.

    Returns:
        DataFrame with warning_id and references columns.
    """
    return refs.group_by('warning_id').agg(
        pl.struct(
            pl.col('ref_id').alias('id'),
            pl.col('ref_type').alias('source'),
            pl.col('ref_url').alias('url'),
        ).alias('references')
    )


def _deduplicate_warnings(df: pl.DataFrame) -> pl.DataFrame:
    """Merge warnings that carry identical information for the same drug.

    ChEMBL propagates the warning for their child/salt molecules in
    ``_metadata.all_molecule_chembl_ids`` to include the parent. When the parent
    has the same warning, these rows carry identical display information and differ only
    by ``chemblIds`` (and the per-molecule ``id``, already dropped). This step avoids the
    duplication on the warning for the parent once the data is exploded
    by ``chemblId`` downstream.

    Group on every display field, union the ``chemblIds`` (so the merged warning
    still reaches every molecule it applied to) and keep the lowest ``id``. ``references``
    stays in the grouping key: two warnings with genuinely different references are not
    duplicates and must not be merged.
    """
    group_cols = [c for c in df.columns if c not in ('id', 'chemblIds')]
    return (
        df.group_by(group_cols, maintain_order=True)
        .agg(
            pl.col('id').min().alias('id'),
            pl.col('chemblIds').list.explode(keep_nulls=False, empty_as_null=False).unique(maintain_order=True).alias(
                'chemblIds'
            ),
        )
        .select(df.columns)
    )
