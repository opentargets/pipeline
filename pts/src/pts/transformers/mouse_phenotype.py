"""Mouse phenotype dataset generation.

The phenotypes of IMPC mouse models, grouped by mouse gene, human target and MP term, with the
models and the high-level MP classes of each term. Rows whose human target is not in `target` are
excluded.
"""

from __future__ import annotations

import json
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas.mouse_phenotype import MOUSE_PHENOTYPE
from pts.transformers.impc_common import (
    MODEL_ID,
    biological_model_id,
    literature,
    model_mouse_phenotypes,
    ontology_terms,
    read_gene_mapping,
    read_impc_csv,
    read_mgi_pubmed,
)
from pts.transformers.utils.dataset import scan_dataset, write_dataset

MP_ROOT = 'MP:0000001'

GROUP_KEYS = [
    'targetInModel',
    'targetInModelMgiId',
    'targetInModelEnsemblId',
    'targetFromSourceId',
    'modelPhenotypeId',
    'modelPhenotypeLabel',
]


def normalise_ontology_id(iri: str) -> str:
    """Turn an ontology IRI into its short id: `.../obo/MP_0000001` becomes `MP:0000001`."""
    if iri.startswith('MP:'):
        return iri
    if '/' not in iri:
        return iri
    last = iri.rsplit('/', maxsplit=1)[-1].split('#', maxsplit=1)[0]
    if last.startswith('MP_'):
        return last.replace('MP_', 'MP:', 1)
    return last


def mp_classes(ontology: dict) -> pl.DataFrame:
    """Map every MP term to each of the direct children of the root that it descends from, itself included."""
    graph = ontology['graphs'][0]

    children: dict[str, set[str]] = {}
    for edge in graph['edges']:
        if edge.get('pred') == 'is_a':
            children.setdefault(normalise_ontology_id(edge['obj']), set()).add(normalise_ontology_id(edge['sub']))

    labels: dict[str, str | None] = {}
    for node in graph['nodes']:
        term = normalise_ontology_id(node['id'])
        if term not in labels or (not labels[term] and node.get('lbl')):
            labels[term] = node.get('lbl', '')

    if MP_ROOT not in labels:
        raise ValueError(f'Could not find root node {MP_ROOT} in ontology')

    rows = []
    for cls in children.get(MP_ROOT, set()) - {MP_ROOT}:
        label = labels.get(cls, cls)
        seen: set[str] = set()
        stack = [cls]
        while stack:
            term = stack.pop()
            if term not in seen:
                seen.add(term)
                stack.extend(children.get(term, ()))
        rows.extend((term, cls, label) for term in seen)

    return pl.DataFrame(
        rows,
        schema={
            'modelPhenotypeId': pl.String,
            'modelPhenotypeClassId': pl.String,
            'modelPhenotypeClassLabel': pl.String,
        },
        orient='row',
    ).sort('modelPhenotypeId', 'modelPhenotypeClassId')


def read_mp_classes(path: str) -> pl.DataFrame:
    """Read the MP ontology JSON and return its phenotype classes."""
    logger.debug('loading mp ontology')
    blob, _ = StorageHandle(path).read()
    return mp_classes(json.loads(blob))


def generate_mouse_phenotypes(
    summary: pl.DataFrame,
    gene_mapping: pl.DataFrame,
    model_mouse: pl.DataFrame,
    mp_terms: pl.DataFrame,
    model_literature: pl.DataFrame,
    classes: pl.DataFrame,
) -> pl.DataFrame:
    """One row per mouse gene, human target and MP term, with its models and the term's classes.

    Args:
        summary: `MODEL_ID`, `biologicalModelAllelicComposition`, `biologicalModelGeneticBackground`
            and `targetInModelMgiId` of the disease model summary.
        gene_mapping: `gene_mapping` of the MGI and HGNC files.
        model_mouse: `model_mouse_phenotypes`.
        mp_terms: `mp_id` and `mp_term` of the IMPC ontology.
        model_literature: the PMIDs per `MODEL_ID` and `targetInModelMgiId`.
        classes: `mp_classes`.

    Returns:
        `MOUSE_PHENOTYPE` rows sorted by `GROUP_KEYS`, their lists sorted.
    """
    model = pl.struct(
        allelicComposition='biologicalModelAllelicComposition',
        geneticBackground='biologicalModelGeneticBackground',
        id='biologicalModelId',
        literature='literature',
    )
    phenotype_class = pl.struct(id='modelPhenotypeClassId', label='modelPhenotypeClassLabel')
    return (
        summary
        .select(MODEL_ID, 'biologicalModelAllelicComposition', 'biologicalModelGeneticBackground', 'targetInModelMgiId')
        .unique(maintain_order=True)
        .join(gene_mapping, on='targetInModelMgiId', how='inner', maintain_order='left')
        .join(model_mouse, on=MODEL_ID, how='inner', maintain_order='left')
        .join(mp_terms, on='mp_id', how='inner', maintain_order='left')
        .join(model_literature, on=[MODEL_ID, 'targetInModelMgiId'], how='left', maintain_order='left')
        .rename({'mp_id': 'modelPhenotypeId', 'mp_term': 'modelPhenotypeLabel'})
        .join(classes, on='modelPhenotypeId', how='inner', maintain_order='left')
        .with_columns(biologicalModelId=biological_model_id(pl.col(MODEL_ID)))
        .group_by(GROUP_KEYS)
        .agg(
            biologicalModels=model.unique().sort(),
            modelPhenotypeClasses=phenotype_class.unique().sort(),
        )
        .sort(GROUP_KEYS)
        .select(list(MOUSE_PHENOTYPE))
    )


def split_by_target(mouse_phenotypes: pl.DataFrame, target_ids: pl.Series) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The rows whose `targetFromSourceId` is in `target_ids`, and the others with that column first."""
    targets = target_ids.alias('targetFromSourceId').to_frame()
    known = mouse_phenotypes.join(targets, on='targetFromSourceId', how='semi', maintain_order='left')
    unknown = mouse_phenotypes.join(targets, on='targetFromSourceId', how='anti', maintain_order='left')
    return known, unknown.select('targetFromSourceId', pl.exclude('targetFromSourceId'))


def mouse_phenotype(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Generate the mouse phenotypes from the IMPC files and split them by target.

    Args:
        source: `solr_disease_model_summary`, `solr_mouse_model`, `solr_ontology`, `solr_gene_gene`,
            `hgnc_gene_mappings`, `mouse_gene_mappings`, `mouse_pubmed_refs`, `mp_ontology` and
            `target` locations.
        destination: `output` and `excluded` directories.
        settings: unused.
        config: unused.
    """
    logger.info('load impc data for mouse phenotypes')
    summary = read_impc_csv(
        source['solr_disease_model_summary'],
        columns=['model_id', 'model_description', 'model_genetic_background', 'marker_id'],
    ).select(
        MODEL_ID,
        biologicalModelAllelicComposition=pl.col('model_description'),
        biologicalModelGeneticBackground=pl.col('model_genetic_background'),
        targetInModelMgiId=pl.col('marker_id'),
    )
    model_mouse = model_mouse_phenotypes(source['solr_mouse_model'])
    summary, gene_mapping, model_mouse, mp_terms, model_literature, targets = pl.collect_all([
        summary,
        read_gene_mapping(source),
        model_mouse,
        ontology_terms(read_impc_csv(source['solr_ontology']), 'MP'),
        literature(read_mgi_pubmed(source['mouse_pubmed_refs']), summary, model_mouse),
        scan_dataset(source['target']).select('id'),
    ])

    mouse_phenotypes = generate_mouse_phenotypes(
        summary, gene_mapping, model_mouse, mp_terms, model_literature, read_mp_classes(source['mp_ontology'])
    )
    output, excluded = split_by_target(mouse_phenotypes, targets['id'])
    logger.info(f'{output.height} mouse phenotypes, {excluded.height} excluded')
    write_dataset(output, destination['output'])
    write_dataset(excluded, destination['excluded'])
