"""Shared `chemblIds` resolution for ChEMBL drug datasets.

Used by both ``drug_warning`` and ``drug_mechanism_of_action``, which each key a
molecule/parent ChEMBL id pair off a different id column but otherwise apply the
identical join and dedup logic -- kept here once so the release-equivalence claim
about what `_metadata.all_molecule_chembl_ids` used to contain cannot drift
between the two copies.
"""

import polars as pl


def chembl_ids(records: pl.DataFrame, molecules: pl.DataFrame, hierarchy: pl.DataFrame, key: str) -> pl.DataFrame:
    """Resolve the deduplicated {molecule, parent molecule} ChEMBL id pair per record.

    Replaces the old `_metadata.all_molecule_chembl_ids` field from the Elasticsearch
    document, which was always exactly this deduplicated pair.

    Args:
        records: Raw ChEMBL table with `key` and `molregno` columns.
        molecules: Raw ChEMBL molecule_dictionary table.
        hierarchy: Raw ChEMBL molecule_hierarchy table.
        key: Name of the id column in `records` to key the result by
            (e.g. `warning_id` or `mec_id`).

    Returns:
        DataFrame with `key` and `chemblIds` columns.
    """
    parent = (
        hierarchy.join(
            molecules.rename({'chembl_id': 'parent_chembl_id'}),
            left_on='parent_molregno',
            right_on='molregno',
            how='left',
        ).select('molregno', 'parent_chembl_id')
    )
    return (
        records.select(key, 'molregno')
        .join(molecules, on='molregno', how='left')
        .join(parent, on='molregno', how='left')
        .select(
            key,
            pl.concat_list('chembl_id', 'parent_chembl_id')
            .list.drop_nulls()
            .list.unique(maintain_order=True)
            .alias('chemblIds'),
        )
    )
