"""Shared `chemblIds` resolution for ChEMBL drug datasets.

Used by both ``drug_warning`` and ``drug_mechanism_of_action``, which each key a
molecule/parent ChEMBL id pair off a different id column but otherwise apply
identical join and dedup logic. Kept here once so the two cannot drift apart:
``chemblIds`` is published by both datasets and has to mean the same thing in each.
"""

import polars as pl


def add_parent_chembl_ids(
    rows: pl.DataFrame, molecules: pl.DataFrame, hierarchy: pl.DataFrame, key: str
) -> pl.DataFrame:
    """Resolve the deduplicated {molecule, parent molecule} ChEMBL id pair per record.

    A record states one molregno; the published `chemblIds` carries that molecule and
    its parent, deduplicated, so a record attached to a salt is also found under the
    parent drug. A molecule that is its own parent yields a single id.

    Args:
        rows: Raw ChEMBL table with `key` and `molregno` columns.
        molecules: Raw ChEMBL molecule_dictionary table.
        hierarchy: Raw ChEMBL molecule_hierarchy table.
        key: Name of the id column in `rows` to key the result by
            (e.g. `warning_id` or `mec_id`).

    Returns:
        DataFrame with `key` and `chemblIds` columns.
    """
    # `maintain_order='left'` throughout: a polars join makes no promise about row
    # order by default, and the caller's row order is what ends up deciding the
    # element order of published arrays such as `references`.
    parent = hierarchy.join(
        molecules.rename({'chembl_id': 'parent_chembl_id'}),
        left_on='parent_molregno',
        right_on='molregno',
        how='left',
        maintain_order='left',
    ).select('molregno', 'parent_chembl_id')
    return (
        rows
        .select(key, 'molregno')
        .join(molecules, on='molregno', how='left', maintain_order='left')
        .join(parent, on='molregno', how='left', maintain_order='left')
        .select(
            key,
            pl
            .concat_list('chembl_id', 'parent_chembl_id')
            .list.drop_nulls()
            .list.unique(maintain_order=True)
            .alias('chemblIds'),
        )
    )
