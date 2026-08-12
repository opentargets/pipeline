"""Mechanism of Action processing for drugs.

Prepares the mechanism of action section of the drug object by joining raw
ChEMBL mechanism, target, and reference tables with target/gene information.
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
    'drug_mechanism': ['mec_id', 'record_id', 'molregno', 'mechanism_of_action', 'tid', 'action_type'],
    'mechanism_refs': ['mec_id', 'ref_type', 'ref_id', 'ref_url'],
    'molecule_dictionary': ['molregno', 'chembl_id'],
    'molecule_hierarchy': ['molregno', 'parent_molregno'],
    'target_dictionary': ['tid', 'chembl_id', 'pref_name', 'target_type'],
    'target_components': ['tid', 'component_id'],
    'component_sequences': ['component_id', 'accession'],
}
"""ChEMBL tables and columns this step needs, restored from the dump."""


def drug_mechanism_of_action(
    source: dict[str, Path],
    destination: Path,
    _settings: dict[str, Any],
    _config: Config,
) -> None:
    """Transform ChEMBL mechanism of action data into the Open Targets format.

    Args:
        source: Dictionary with paths to:
            - chembl: Path to the ChEMBL ``pg_dump`` archive.
            - target: Path to the target parquet (gene data).
        destination: Path to write the output parquet file.
        _settings: Custom settings (not used).
        _config: Config object (not used).
    """
    logger.info(f'Restoring {list(TABLES)} from {source["chembl"]}')
    tables = read_dump_tables(str(source['chembl']), TABLES, schema_name=SCHEMA_NAME)

    logger.info(f'Reading target data from {source["target"]}')
    gene_df = pl.read_parquet(source['target'])

    logger.info('Processing mechanisms of action')
    output_df = process_mechanism_of_action(
        tables['drug_mechanism'],
        tables['mechanism_refs'],
        tables['molecule_dictionary'],
        tables['molecule_hierarchy'],
        tables['target_dictionary'],
        tables['target_components'],
        tables['component_sequences'],
        gene_df,
    )

    logger.info(f'Writing mechanism of action to {destination}')
    output_df.write_parquet(destination, mkdir=True)


def process_mechanism_of_action(
    drug_mechanism: pl.DataFrame,
    mechanism_refs: pl.DataFrame,
    molecule_dictionary: pl.DataFrame,
    molecule_hierarchy: pl.DataFrame,
    target_dictionary: pl.DataFrame,
    target_components: pl.DataFrame,
    component_sequences: pl.DataFrame,
    gene_df: pl.DataFrame,
) -> pl.DataFrame:
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
        Processed mechanism of action DataFrame, one row per surviving `mec_id`.
    """
    ids = _chembl_ids(drug_mechanism, molecule_dictionary, molecule_hierarchy, key='mec_id')
    mechanism_refs_agg = mechanism_refs.group_by('mec_id').agg(
        pl.struct(
            pl.col('ref_type'),
            pl.col('ref_id'),
            pl.col('ref_url'),
        ).alias('mechanism_refs')
    )

    mechanism = (
        _with_target_chembl_id(drug_mechanism, target_dictionary)
        .join(molecule_dictionary.select('molregno', pl.col('chembl_id').alias('id')), on='molregno', how='left')
        .join(ids, on='mec_id', how='left')
        .join(mechanism_refs_agg, on='mec_id', how='left')
        .rename({'mechanism_of_action': 'mechanismOfAction', 'action_type': 'actionType'})
        .drop('mec_id', 'molregno')
    )

    references = _chembl_mechanism_references(mechanism)
    target = _chembl_target(target_dictionary, target_components, component_sequences, gene_df)

    result = (
        mechanism.join(references, on='id', how='full', coalesce=True)
        .join(target, on='target_chembl_id', how='full', coalesce=True)
        .with_columns(pl.col('references').fill_null([]))
        .drop('mechanism_refs', 'record_id', 'target_chembl_id', 'id')
        .filter(
            pl.col('mechanismOfAction').is_not_null()
            & (pl.col('targets').is_not_null() | pl.col('targetName').is_not_null())
            & pl.col('chemblIds').is_not_null()
            & (pl.col('chemblIds').list.len() > 0)
        )
    )

    return _consolidate_duplicate_references(result)


def _with_target_chembl_id(mechanism: pl.DataFrame, target_dictionary: pl.DataFrame) -> pl.DataFrame:
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
        mechanism.join(
            target_dictionary.select('tid', pl.col('chembl_id').alias('target_chembl_id')),
            on='tid',
            how='left',
        ).drop('tid')
    )


def _chembl_mechanism_references(df: pl.DataFrame) -> pl.DataFrame:
    """Extract and structure references from mechanism data.

    Args:
        df: Mechanism DataFrame with id and mechanism_refs columns.

    Returns:
        DataFrame with id and references columns. Only ids that have at least one
        reference are present -- the caller is responsible for filling the gap
        left by ids with none.
    """
    return (
        df.select('id', 'mechanism_refs')
        .explode('mechanism_refs')
        .filter(pl.col('mechanism_refs').is_not_null())
        .unnest('mechanism_refs')
        .group_by('id', 'ref_type')
        .agg(
            # collect_list on the pyspark side drops nulls; polars' bare list aggregation
            # does not, so a NULL ref_id/ref_url (e.g. an 'Expert' or 'KEGG' reference with
            # no id) would otherwise survive as `None` inside the array instead of being
            # dropped, and would desynchronise `ids` and `urls` differently than pyspark does.
            pl.col('ref_id').drop_nulls().alias('ids'),
            pl.col('ref_url').drop_nulls().alias('urls'),
        )
        .with_columns(
            references=pl.struct(
                pl.col('ref_type').alias('source'),
                pl.col('ids'),
                pl.col('urls'),
            )
        )
        .group_by('id')
        .agg(pl.col('references'))
    )


def _chembl_target(
    target_dictionary: pl.DataFrame,
    target_components: pl.DataFrame,
    component_sequences: pl.DataFrame,
    gene_df: pl.DataFrame,
) -> pl.DataFrame:
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
        target_components.join(component_sequences, on='component_id', how='inner')
        .join(target_dictionary, on='tid', how='inner')
        .filter(pl.col('accession').is_not_null())
        .select(
            pl.col('pref_name').alias('targetName'),
            pl.col('accession').alias('uniprot_id'),
            pl.col('target_type').str.to_lowercase().alias('targetType'),
            pl.col('chembl_id').alias('target_chembl_id'),
        )
    )

    # Gene lookup keyed by uniprot accession, plus an identity mapping keyed by the gene id
    # itself -- the original pyspark join matched on `uniprot_id == genes.uniprot_id OR
    # uniprot_id == genes.geneId`, which a single equi-join cannot express directly.
    genes_by_uniprot = (
        gene_df.select(
            pl.col('id').alias('geneId'),
            pl.concat_list(
                pl.col('uniprot_trembl').fill_null([]),
                pl.col('uniprot_swissprot').fill_null([]),
            ).alias('uniprotIds'),
        )
        .explode('uniprotIds')
        .rename({'uniprotIds': 'uniprot_id'})
        .drop_nulls('uniprot_id')
    )
    genes_by_id = gene_df.select(pl.col('id').alias('geneId'), pl.col('id').alias('uniprot_id'))
    gene_lookup = pl.concat([genes_by_uniprot, genes_by_id]).unique()

    joined = target_components_flat.join(gene_lookup, on='uniprot_id', how='left')

    return joined.group_by('target_chembl_id', 'targetName', 'targetType').agg(
        pl.col('geneId').drop_nulls().unique().alias('targets')
    )


def _consolidate_duplicate_references(df: pl.DataFrame) -> pl.DataFrame:
    """Consolidate mechanism rows that are identical for the same drug.

    ChEMBL propagates the mechanism for their child/salt molecules in
    ``_metadata.all_molecule_chembl_ids`` to include the parent. When the parent has
    the same mechanism, these rows carry identical display information and differ only
    by ``chemblIds`` (and the per-molecule ``id``, already dropped). This step avoids the
    duplication on the mechanism for the parent once the data is exploded
    by ``chemblId`` downstream.

    ``chemblIds`` and ``references`` are deduplicated at different granularities, matching
    the pyspark reference exactly: ``chemblIds`` unions the individual ids across the group
    (element-level distinct after flattening, mirroring ``array_distinct(flatten(collect_list(...)))``).
    ``references`` instead dedupes whole per-row lists first and only then concatenates the
    survivors (mirroring ``flatten(collect_set(...))``), so two rows whose reference lists are
    not byte-identical can still contribute overlapping/duplicate reference structs to the
    result -- that mismatch is in the published data and not accidental here.
    """
    key_cols = [c for c in df.columns if c not in ('references', 'chemblIds')]
    return (
        df.group_by(key_cols, maintain_order=True)
        .agg(
            pl.col('chemblIds').list.explode(keep_nulls=False, empty_as_null=False).unique(maintain_order=True).alias(
                'chemblIds'
            ),
            pl.col('references').unique(maintain_order=True).list.explode(keep_nulls=False, empty_as_null=False).alias(
                'references'
            ),
        )
        .select(df.columns)
    )
