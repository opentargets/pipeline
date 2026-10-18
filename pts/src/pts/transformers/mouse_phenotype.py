"""Mouse phenotype dataset generation."""

import json

import polars as pl
from loguru import logger
from otter.storage.synchronous.handle import StorageHandle

MP_ROOT = 'MP:0000001'


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
