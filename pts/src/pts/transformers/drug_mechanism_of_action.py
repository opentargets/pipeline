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
from pts.transformers.utils.dataset import scan_dataset, write_dataset

SCHEMA_NAME = 'public'
"""Schema the ChEMBL tables live in inside the restored dump."""

TABLES = {
    'drug_mechanism': ['mec_id', 'record_id', 'molregno', 'mechanism_of_action', 'tid', 'action_type'],
    # mecref_id is the table's key and is otherwise unused: it is here only so the
    # read can be ordered by it, see ORDER_BY. The key is unique, so including it
    # cannot change what the SELECT DISTINCT returns.
    'mechanism_refs': ['mecref_id', 'mec_id', 'ref_type', 'ref_id', 'ref_url'],
    'molecule_dictionary': ['molregno', 'chembl_id'],
    'molecule_hierarchy': ['molregno', 'parent_molregno'],
    'target_dictionary': ['tid', 'chembl_id', 'pref_name', 'target_type'],
    'target_components': ['tid', 'component_id'],
    'component_sequences': ['component_id', 'accession'],
}
"""ChEMBL tables and columns this step needs, restored from the dump."""

ORDER_BY = {
    'drug_mechanism': ['mec_id'],
    'mechanism_refs': ['mecref_id'],
    'target_dictionary': ['tid'],
    'target_components': ['tid', 'component_id'],
    'component_sequences': ['component_id', 'accession'],
}
"""Reads whose row order reaches the output and therefore must not float.

``_chembl_mechanism_references`` collects ``mechanism_refs`` into the published
``references`` array in scan order, and ``_chembl_target`` collects ``targets``
the same way, so an unordered read leaves that array order at the mercy of the
query plan and the physical layout of the restored dump. ``drug_mechanism`` is in
here too because it drives the row order the references are grouped in, and
because ``_consolidate_duplicate_references`` groups with ``maintain_order=True``.

Where the table's key is in the projection (``mecref_id``, ``mec_id``, ``tid``)
that is what it is ordered by; the other two are ordered by their whole
projection, which the ``SELECT DISTINCT`` makes a total order. Every one of these
is a small table, so the sort is free.

Ordering the reads is only half of it. Polars makes no promise about row order out
of a join, a ``group_by`` or a ``unique`` unless asked, so ``maintain_order`` is
pinned everywhere the order can reach the output. Without both halves the step does
not reproduce its own output on byte-identical inputs, and a step that cannot
reproduce itself cannot be diffed at all.

``molecule_dictionary`` and ``molecule_hierarchy`` are deliberately absent: they
are joined row-wise on ``molregno`` by joins that keep the left frame's order, so
their scan order reaches nothing, and they are the large tables here.
"""


def drug_mechanism_of_action(
    source: dict[str, Path],
    destination: Path,
    _settings: dict[str, Any],
    config: Config,
) -> None:
    """Transform ChEMBL mechanism of action data into the Open Targets format.

    Args:
        source: Dictionary with paths to:
            - chembl: Path to the ChEMBL ``pg_dump`` archive.
            - target: Path to the target parquet (gene data).
        destination: Path to write the output parquet file.
        _settings: Custom settings (not used).
        config: Config object, for ``work_path``.
    """
    logger.info(f'Restoring {list(TABLES)} from {source["chembl"]}')
    # scratch_root: the restore needs gigabytes, and `work_path` is the work disk.
    # See the note in drug_warning.
    tables = read_dump_tables(
        str(source['chembl']), TABLES, schema_name=SCHEMA_NAME, order_by=ORDER_BY, scratch_root=config.work_path
    )

    logger.info(f'Reading target data from {source["target"]}')
    gene_df = scan_dataset(str(source['target'])).collect()

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
    write_dataset(output_df, str(destination))


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
    mechanism_refs_agg = mechanism_refs.group_by('mec_id', maintain_order=True).agg(
        pl.struct(
            pl.col('ref_type'),
            pl.col('ref_id'),
            pl.col('ref_url'),
        ).alias('mechanism_refs')
    )

    mechanism = (
        _with_target_chembl_id(drug_mechanism, target_dictionary)
        .join(
            molecule_dictionary.select('molregno', pl.col('chembl_id').alias('id')),
            on='molregno',
            how='left',
            maintain_order='left',
        )
        .join(ids, on='mec_id', how='left', maintain_order='left')
        .join(mechanism_refs_agg, on='mec_id', how='left', maintain_order='left')
        .rename({'mechanism_of_action': 'mechanismOfAction', 'action_type': 'actionType'})
        .drop('mec_id', 'molregno')
    )

    references = _chembl_mechanism_references(mechanism)
    target = _chembl_target(target_dictionary, target_components, component_sequences, gene_df)

    result = (
        mechanism
        .join(references, on='id', how='full', coalesce=True, maintain_order='left_right')
        .join(target, on='target_chembl_id', how='full', coalesce=True, maintain_order='left_right')
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
    return mechanism.join(
        target_dictionary.select('tid', pl.col('chembl_id').alias('target_chembl_id')),
        on='tid',
        how='left',
        maintain_order='left',
    ).drop('tid')


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
        df
        .select('id', 'mechanism_refs')
        .explode('mechanism_refs')
        .filter(pl.col('mechanism_refs').is_not_null())
        .unnest('mechanism_refs')
        .group_by('id', 'ref_type', maintain_order=True)
        .agg(
            # drop_nulls is load-bearing: a reference with no id or url (an 'Expert' or
            # 'KEGG' entry, say) would otherwise contribute a `None` element, and since
            # the two columns are null in different rows, `ids` and `urls` would come out
            # positionally desynchronised inside the published struct.
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
        .group_by('id', maintain_order=True)
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
        target_components
        .join(component_sequences, on='component_id', how='inner', maintain_order='left')
        .join(target_dictionary, on='tid', how='inner', maintain_order='left')
        .filter(pl.col('accession').is_not_null())
        .select(
            pl.col('pref_name').alias('targetName'),
            pl.col('accession').alias('uniprot_id'),
            pl.col('target_type').str.to_lowercase().alias('targetType'),
            pl.col('chembl_id').alias('target_chembl_id'),
        )
    )

    # ChEMBL spells a target either as a uniprot accession or as an Ensembl gene id, so
    # resolution needs two lookups: one keyed by accession, one keyed by the gene id
    # itself. `genes_by_id` is deliberately built from the whole of `gene_df` rather than
    # from the rows that survive the explode below -- the explode drops genes whose
    # accession lists are both empty, and those genes must still be resolvable by id.
    # Most genes have no uniprot accession at all, so restricting the id lookup to
    # exploded rows would silently drop every mechanism ChEMBL states as a gene id.
    genes_by_uniprot = (
        gene_df
        .select(
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
    gene_lookup = pl.concat([genes_by_uniprot, genes_by_id]).unique(maintain_order=True)

    joined = target_components_flat.join(gene_lookup, on='uniprot_id', how='left', maintain_order='left')

    return joined.group_by('target_chembl_id', 'targetName', 'targetType', maintain_order=True).agg(
        pl.col('geneId').drop_nulls().unique(maintain_order=True).alias('targets')
    )


def _consolidate_duplicate_references(df: pl.DataFrame) -> pl.DataFrame:
    """Consolidate mechanism rows that are identical for the same drug.

    ChEMBL propagates the mechanism for their child/salt molecules in
    ``_metadata.all_molecule_chembl_ids`` to include the parent. When the parent has
    the same mechanism, these rows carry identical display information and differ only
    by ``chemblIds`` (and the per-molecule ``id``, already dropped). This step avoids the
    duplication on the mechanism for the parent once the data is exploded
    by ``chemblId`` downstream.

    ``chemblIds`` and ``references`` are deduplicated at different granularities, which is
    deliberate: ``chemblIds`` is flattened and then distinct-ed element by element, while
    ``references`` dedupes whole per-row lists first and only then concatenates the
    survivors. Two rows whose reference lists are not byte-identical can therefore still
    contribute overlapping reference structs. That asymmetry is present in the published
    data, so it is preserved rather than tidied up.
    """
    key_cols = [c for c in df.columns if c not in ('references', 'chemblIds')]
    return (
        df
        .group_by(key_cols, maintain_order=True)
        .agg(
            pl
            .col('chemblIds')
            .list.explode(keep_nulls=False, empty_as_null=False)
            .unique(maintain_order=True)
            .alias('chemblIds'),
            pl
            .col('references')
            .unique(maintain_order=True)
            .list.explode(keep_nulls=False, empty_as_null=False)
            .alias('references'),
        )
        .select(df.columns)
    )
