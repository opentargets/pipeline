"""Gene sets dataset generation.

One gene set per GO term, Reactome pathway and ChEMBL protein class, holding the targets annotated
to it or to any of its descendants.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.postgres import read_dump_tables
from pts.schemas.gene_sets import gene_sets_schema
from pts.transformers.target import chembl_class
from pts.transformers.utils.dataset import scan_dataset, write_dataset
from pts.transformers.utils.lists import collect_set, explode

GO_SOURCE = 'go'
REACTOME_SOURCE = 'reactome'
CHEMBL_TARGET_CLASS_SOURCE = 'chembl_target_class'

GO_NAMESPACES = {
    'molecular_function': 'GO:MF',
    'biological_process': 'GO:BP',
    'cellular_component': 'GO:CC',
}
REACTOME_CATEGORY = 'Reactome'
CHEMBL_TARGET_CLASS_CATEGORY = 'ChEMBL Target Class'


def with_ancestors(lf: pl.LazyFrame, reference: pl.LazyFrame) -> pl.LazyFrame:
    """Replace each id with itself and its ancestors from the reference."""
    ancestors = reference.select('id', 'ancestors')
    return (
        lf
        .filter(pl.col('id').is_not_null())
        .join(ancestors, on='id', how='left')
        .with_columns(
            pl.concat_list(pl.col('id'), pl.col('ancestors').fill_null([])).list.unique(maintain_order=True).alias('id')
        )
        .drop('ancestors')
        .pipe(explode, 'id')
    )


def _go_hierarchy(go: pl.LazyFrame) -> pl.LazyFrame:
    """Live GO terms, with parents from `isA` and `partOf` that are also ancestors.

    Ancestors stay within the term's namespace, so this drops parents from another namespace.
    """
    return go.filter(~pl.col('isObsolete')).select(
        'id',
        'label',
        pl.lit(GO_SOURCE).alias('source'),
        pl.col('namespace').replace_strict(GO_NAMESPACES).alias('category'),
        pl.concat_list('isA', 'partOf').list.set_intersection('ancestors').list.sort().alias('parentIds'),
        'ancestors',
    )


def _reactome_hierarchy(reactome: pl.LazyFrame) -> pl.LazyFrame:
    """Reactome pathways, with parents and ancestors as computed by the reactome step."""
    return reactome.select(
        'id',
        'label',
        pl.lit(REACTOME_SOURCE).alias('source'),
        pl.lit(REACTOME_CATEGORY).alias('category'),
        pl.col('parents').alias('parentIds'),
        'ancestors',
    )


def _chembl_class_hierarchy(classes: pl.DataFrame) -> pl.LazyFrame:
    """Protein classes below the ChEMBL root, with parents and ancestors from `parent_id`."""
    classes = classes.filter(pl.col('class_level') > 0).select(
        pl.col('protein_class_id').cast(pl.String).alias('id'),
        pl.col('parent_id').cast(pl.String),
        pl.col('pref_name').alias('label'),
    )
    graph = nx.DiGraph()
    graph.add_nodes_from(classes['id'])
    graph.add_edges_from(
        (child, parent) for child, parent in classes.select('id', 'parent_id').iter_rows() if parent in graph
    )
    ids = classes['id']
    return (
        classes
        .drop('parent_id')
        .with_columns(
            pl.lit(CHEMBL_TARGET_CLASS_SOURCE).alias('source'),
            pl.lit(CHEMBL_TARGET_CLASS_CATEGORY).alias('category'),
            pl.Series('parentIds', [sorted(graph.successors(i)) for i in ids], dtype=pl.List(pl.String)),
            pl.Series('ancestors', [sorted(nx.descendants(graph, i)) for i in ids], dtype=pl.List(pl.String)),
        )
        .lazy()
    )


def _annotations(target: pl.LazyFrame, column: str, field: str) -> pl.LazyFrame:
    """One row per target and term id found in `column[].field`."""
    return explode(target.select(pl.col('id').alias('targetId'), column), column).select(
        'targetId', pl.col(column).struct.field(field).cast(pl.String).alias('id')
    )


def _propagate(annotations: pl.LazyFrame, hierarchy: pl.LazyFrame) -> pl.LazyFrame:
    """Each hierarchy term with the targets annotated to it or a descendant."""
    targets = with_ancestors(annotations, hierarchy).group_by('id').agg(collect_set('targetId').alias('entityIds'))
    return hierarchy.drop('ancestors').join(targets, on='id', how='inner')


def gene_sets(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate the gene sets dataset.

    Args:
        source: paths keyed by targets, go, reactome and chembl (the ChEMBL postgres dump).
        destination: path of the gene sets dataset.
        settings: unused.
        config: for `work_path`, where the ChEMBL restore puts its scratch.
    """
    target = scan_dataset(source['targets']).select('id', 'go', 'pathways', 'targetClass')
    go = scan_dataset(source['go'])
    reactome = scan_dataset(source['reactome'])
    logger.info(f'restoring the protein classification table from {source["chembl"]}')
    # scratch_root: the restore needs gigabytes, and `work_path` is the work disk.
    classes = read_dump_tables(
        source['chembl'],
        {'protein_classification': chembl_class.TABLES['protein_classification']},
        schema_name=chembl_class.SCHEMA_NAME,
        scratch_root=config.work_path,
    )['protein_classification']

    logger.info('computing gene sets')
    result = (
        pl
        .concat([
            _propagate(_annotations(target, 'go', 'id'), _go_hierarchy(go)),
            _propagate(_annotations(target, 'pathways', 'pathwayId'), _reactome_hierarchy(reactome)),
            _propagate(_annotations(target, 'targetClass', 'id'), _chembl_class_hierarchy(classes)),
        ])
        .select(list(gene_sets_schema))
        .sort('source', 'id')
        .collect()
    )

    logger.info(f'writing {result.height} gene sets to {destination}')
    write_dataset(result, destination)
