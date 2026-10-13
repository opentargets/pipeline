"""Search facet dataset generation.

Ported from the platform-etl-backend searchFacet step (SearchFacet.scala, TargetFacets.scala,
DiseaseFacets.scala, Helpers.scala). Computes the target and disease facets used by the Open Targets
Platform search.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.transformers.utils.dataset import scan_dataset, write_dataset

DEFAULT_CATEGORIES: dict[str, str] = {
    'disease_name': 'Disease',
    'therapeutic_area': 'Therapeutic Area',
    'sm': 'Tractability Small Molecule',
    'ab': 'Tractability Antibody',
    'pr': 'Tractability PROTAC',
    'oc': 'Tractability Other Modalities',
    'target_id': 'Target ID',
    'approved_symbol': 'Approved Symbol',
    'approved_name': 'Approved Name',
    'subcellular_location': 'Subcellular Location',
    'target_class': 'ChEMBL Target Class',
    'pathways': 'Reactome',
    'go_f': 'GO:MF',
    'go_p': 'GO:BP',
    'go_c': 'GO:CC',
}

TARGET_COLUMNS = ['label', 'category', 'entityIds', 'datasourceId']
DISEASE_COLUMNS = ['label', 'category', 'datasourceId', 'entityIds']


def explode(lf: pl.LazyFrame, column: str) -> pl.LazyFrame:
    """Explode a list column, dropping rows whose list is null or empty."""
    return lf.filter(pl.col(column).list.len() > 0).explode(column)


def collect_set(column: str) -> pl.Expr:
    """Distinct non-null values of a column, sorted."""
    return pl.col(column).drop_nulls().unique().sort()


def facet(lf: pl.LazyFrame, keys: list[str], entity: str) -> pl.LazyFrame:
    """Group by the facet keys, collecting the distinct entity ids."""
    return lf.group_by(keys).agg(collect_set(entity).alias('entityIds'))


def null_datasource(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add a null datasourceId."""
    return lf.with_columns(pl.lit(None, dtype=pl.String).alias('datasourceId'))


def map_or_keep(column: str, mapping: dict[str, str]) -> pl.Expr:
    """Map a code to its label, keeping the code when it has none."""
    return pl.col(column).replace_strict(mapping, default=pl.col(column), return_dtype=pl.String)


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


def with_labels(lf: pl.LazyFrame, reference: pl.LazyFrame) -> pl.LazyFrame:
    """Join each id to its reference label, dropping ids without one."""
    labels = reference.select('id', 'label').filter(pl.col('label').is_not_null())
    return lf.join(labels, on='id', how='inner')


def disease_name_facets(disease: pl.LazyFrame, categories: dict[str, str]) -> pl.LazyFrame:
    """One facet per disease, labelled with its name."""
    return facet(
        disease.select(
            pl.col('name').alias('label'),
            pl.lit(categories['disease_name']).alias('category'),
            pl.col('id').alias('datasourceId'),
            'id',
        ),
        ['label', 'category', 'datasourceId'],
        'id',
    )


def therapeutic_area_facets(disease: pl.LazyFrame, categories: dict[str, str]) -> pl.LazyFrame:
    """One facet per therapeutic area, holding the diseases under it."""
    names = disease.select(pl.col('id').alias('datasourceId'), pl.col('name').alias('label'))
    areas = disease.select(pl.col('id').alias('diseaseId'), pl.col('therapeuticAreas').alias('datasourceId'))
    return facet(
        explode(areas, 'datasourceId')
        .join(names, on='datasourceId', how='inner')
        .with_columns(pl.lit(categories['therapeutic_area']).alias('category')),
        ['label', 'category', 'datasourceId'],
        'diseaseId',
    )


def tractability_facets(tractability: pl.LazyFrame, categories: dict[str, str]) -> pl.LazyFrame:
    """One facet per modality and tractability bucket that holds."""
    modalities = {'SM': categories['sm'], 'AB': categories['ab'], 'PR': categories['pr'], 'OC': categories['oc']}
    return (
        facet(
            tractability.filter(pl.col('value')).select('targetId', 'modality', pl.col('category').alias('label')),
            ['modality', 'label'],
            'targetId',
        )
        .with_columns(map_or_keep('modality', modalities).alias('category'))
        .drop('modality')
        .pipe(null_datasource)
    )


def go_facets(target: pl.LazyFrame, go: pl.LazyFrame, categories: dict[str, str]) -> pl.LazyFrame:
    """One facet per GO term and aspect, holding the targets annotated to it or a descendant."""
    aspects = {'F': categories['go_f'], 'P': categories['go_p'], 'C': categories['go_c']}
    annotations = explode(target.select(pl.col('id').alias('targetId'), 'go'), 'go').select(
        'targetId',
        pl.col('go').struct.field('id'),
        pl.col('go').struct.field('aspect'),
    )
    return (
        facet(
            with_labels(with_ancestors(annotations, go), go).rename({'id': 'datasourceId'}),
            ['label', 'aspect', 'datasourceId'],
            'targetId',
        )
        .with_columns(map_or_keep('aspect', aspects).alias('category'))
        .drop('aspect')
    )


def subcellular_location_facets(target: pl.LazyFrame, categories: dict[str, str]) -> pl.LazyFrame:
    """One facet per subcellular location."""
    locations = explode(target.select('id', 'subcellularLocations'), 'subcellularLocations')
    return facet(
        locations.select(
            'id',
            pl.col('subcellularLocations').struct.field('location').alias('label'),
            pl.lit(categories['subcellular_location']).alias('category'),
            pl.col('subcellularLocations').struct.field('termSL').alias('datasourceId'),
        ),
        ['label', 'category', 'datasourceId'],
        'id',
    )


def target_class_facets(target: pl.LazyFrame, categories: dict[str, str]) -> pl.LazyFrame:
    """One facet per ChEMBL target class label."""
    classes = explode(target.select('id', 'targetClass'), 'targetClass')
    return facet(
        classes.select(
            'id',
            pl.col('targetClass').struct.field('label'),
            pl.lit(categories['target_class']).alias('category'),
        ),
        ['label', 'category'],
        'id',
    ).pipe(null_datasource)


def pathway_facets(target: pl.LazyFrame, reactome: pl.LazyFrame, categories: dict[str, str]) -> pl.LazyFrame:
    """One facet per Reactome pathway, holding the targets in it or a descendant."""
    pathways = explode(target.select(pl.col('id').alias('targetId'), 'pathways'), 'pathways').select(
        'targetId', pl.col('pathways').struct.field('pathwayId').alias('id')
    )
    return facet(
        with_labels(with_ancestors(pathways, reactome), reactome).select(
            'targetId',
            'label',
            pl.lit(categories['pathways']).alias('category'),
            pl.col('id').alias('datasourceId'),
        ),
        ['label', 'category', 'datasourceId'],
        'targetId',
    )


def simple_facet(target: pl.LazyFrame, label: str, category: str) -> pl.LazyFrame:
    """One facet per value of a target column."""
    return facet(
        target.select(pl.col(label).alias('label'), pl.lit(category).alias('category'), pl.col('id').alias('_eid')),
        ['label', 'category'],
        '_eid',
    ).pipe(null_datasource)


def target_facets(
    target: pl.LazyFrame,
    go: pl.LazyFrame,
    reactome: pl.LazyFrame,
    tractability: pl.LazyFrame,
    categories: dict[str, str],
) -> pl.LazyFrame:
    """All target facets."""
    frames = [
        simple_facet(target, 'id', categories['target_id']),
        simple_facet(target, 'approvedSymbol', categories['approved_symbol']),
        simple_facet(target, 'approvedName', categories['approved_name']),
        go_facets(target, go, categories),
        subcellular_location_facets(target, categories),
        target_class_facets(target, categories),
        pathway_facets(target, reactome, categories),
        tractability_facets(tractability, categories),
    ]
    return sort_facets(pl.concat([lf.select(TARGET_COLUMNS) for lf in frames]))


def disease_facets(disease: pl.LazyFrame, categories: dict[str, str]) -> pl.LazyFrame:
    """All disease facets."""
    frames = [disease_name_facets(disease, categories), therapeutic_area_facets(disease, categories)]
    return sort_facets(pl.concat([lf.select(DISEASE_COLUMNS) for lf in frames]))


def sort_facets(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Order facets by category, label and datasourceId."""
    return lf.sort('category', 'label', 'datasourceId', nulls_last=True, maintain_order=True)


def search_facet(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Compute the search facets for targets and diseases.

    Args:
        source: paths keyed by diseases, targets, go, reactome and tractability.
        destination: paths keyed by targets and diseases.
        settings: an optional categories mapping overriding the default category labels.
        config: unused.
    """
    categories = {**DEFAULT_CATEGORIES, **(settings.get('categories') or {})}

    disease = scan_dataset(source['diseases']).select('id', 'name', 'therapeuticAreas')
    target = scan_dataset(source['targets']).select(
        'id', 'approvedSymbol', 'approvedName', 'go', 'subcellularLocations', 'targetClass', 'pathways'
    )
    go = scan_dataset(source['go']).select('id', 'label', 'ancestors')
    reactome = scan_dataset(source['reactome']).select('id', 'label', 'ancestors')
    tractability = scan_dataset(source['tractability']).select('targetId', 'modality', 'category', 'value')

    logger.info('computing target facets')
    targets = target_facets(target, go, reactome, tractability, categories).collect()
    logger.info('computing disease facets')
    diseases = disease_facets(disease, categories).collect()

    logger.info(f'writing {targets.height} target facets to {destination["targets"]}')
    write_dataset(targets, destination['targets'])
    logger.info(f'writing {diseases.height} disease facets to {destination["diseases"]}')
    write_dataset(diseases, destination['diseases'])
