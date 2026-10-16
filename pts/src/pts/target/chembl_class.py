"""ChEMBL protein classification by UniProt accession, from the ChEMBL postgres dump tables."""

from __future__ import annotations

import polars as pl
from loguru import logger

SCHEMA_NAME = 'public'

TABLES = {
    'target_dictionary': ['tid', 'chembl_id', 'pref_name', 'target_type'],
    'target_components': ['targcomp_id', 'tid', 'component_id'],
    'component_sequences': ['component_id', 'accession'],
    'component_class': ['comp_class_id', 'component_id', 'protein_class_id'],
    'protein_classification': ['protein_class_id', 'parent_id', 'pref_name', 'class_level'],
}

MAX_CLASS_LEVEL = 6


def flatten_protein_classification(protein_classification: pl.DataFrame) -> pl.DataFrame:
    """Each class's ancestor chain, one column per level.

    Walks up `parent_id` a fixed `MAX_CLASS_LEVEL` times. Each ancestor's `pref_name` lands at its
    own `class_level`; the root, at level 0, falls out.

    Returns:
        [leaf_id, l1, ..., l6].
    """
    max_level = protein_classification.select(pl.col('class_level').max()).item()
    if max_level is not None and max_level > MAX_CLASS_LEVEL:
        logger.warning(
            f'protein_classification has class_level up to {max_level}, above the '
            f'MAX_CLASS_LEVEL={MAX_CLASS_LEVEL} this function walks to. Labels at levels beyond '
            f'{MAX_CLASS_LEVEL} will be silently dropped from targetClass.'
        )

    parents = protein_classification.select(
        pl.col('protein_class_id').alias('node_id'),
        pl.col('parent_id').alias('node_parent_id'),
        pl.col('pref_name').alias('node_pref_name'),
        pl.col('class_level').alias('node_class_level'),
    )
    frontier = protein_classification.select(
        pl.col('protein_class_id').alias('leaf_id'), 'parent_id', 'pref_name', 'class_level'
    )
    chain = [frontier]
    for _ in range(MAX_CLASS_LEVEL):
        frontier = (
            frontier
            .select('leaf_id', pl.col('parent_id').alias('node_id'))
            .join(parents, on='node_id', how='inner')
            .select(
                'leaf_id',
                pl.col('node_parent_id').alias('parent_id'),
                pl.col('node_pref_name').alias('pref_name'),
                pl.col('node_class_level').alias('class_level'),
            )
        )
        chain.append(frontier)

    return (
        pl
        .concat(chain)
        .group_by('leaf_id')
        .agg([
            pl.col('pref_name').filter(pl.col('class_level') == i).max().alias(f'l{i}')
            for i in range(1, MAX_CLASS_LEVEL + 1)
        ])
        .sort('leaf_id')
    )


def build_protein_classification(
    target_dictionary: pl.DataFrame,
    target_components: pl.DataFrame,
    component_sequences: pl.DataFrame,
    component_class: pl.DataFrame,
    protein_classification: pl.DataFrame,
) -> pl.DataFrame:
    """Protein classes per UniProt accession.

    Only single-component targets contribute, and each component keeps only its lowest
    `protein_class_id`: both reproduce the published dataset.

    Returns:
        [accession, targetClass], `targetClass` a sorted list of distinct {id, label, level}.
    """
    levels = flatten_protein_classification(protein_classification)
    single_component_tids = (
        target_components.group_by('tid').agg(pl.len().alias('n')).filter(pl.col('n') == 1).select('tid')
    )
    class_per_component = component_class.group_by('component_id').agg(pl.col('protein_class_id').min())

    accession_class = (
        target_components
        .select('tid', 'component_id')
        .join(target_dictionary.select('tid'), on='tid', how='inner')
        .join(single_component_tids, on='tid', how='inner')
        .join(component_sequences.select('component_id', 'accession'), on='component_id', how='inner')
        .join(class_per_component, on='component_id', how='inner')
        .filter(pl.col('accession').is_not_null())
        .select('accession', 'protein_class_id')
        .join(levels, left_on='protein_class_id', right_on='leaf_id', how='left')
    )

    return (
        pl
        .concat([
            accession_class.select(
                'accession',
                pl.col('protein_class_id').cast(pl.Int64).alias('id'),
                pl.col(f'l{i}').alias('label'),
                pl.lit(f'l{i}').alias('level'),
            )
            for i in range(1, MAX_CLASS_LEVEL + 1)
        ])
        .filter(pl.col('label').is_not_null())
        .unique()
        .sort('accession', 'id', 'level', 'label')
        .group_by('accession', maintain_order=True)
        .agg(pl.struct('id', 'label', 'level').alias('targetClass'))
    )
