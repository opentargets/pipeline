"""Documentation JSON for dataset models.

One document per dataset, built from the same field tree the pandera converter reads, nested to
any depth in the shape of PySpark's `StructType.jsonValue()`::

    {"name": "diseaseCellLines", "description": "...", "required": true, "nullable": false,
     "type": {"type": "list", "element": {
         "type": {"type": "struct", "fields": [
             {"name": "tissueId", "description": "...", "nullable": false, "type": "String",
              "reference": {"dataset": "biosample", "column": "biosampleId"}, "bioregistry": "uberon"},
             ...]},
         "nullable": false}}}

A primitive's `type` is its Polars dtype name. Optional keys -- `description`, `constraints`,
`reference`, `bioregistry`, `note`, `primary_key` -- appear only when set.

The documents of every discovered dataset are kept under version control (see
`pts/docs/schemas/`), so a change to any dataset's contract shows up as a diff in review.
Regenerate them with::

    uv run python -m pts.schemas.dataset.docs docs/schemas
"""

import json
import sys
from pathlib import Path
from typing import Any

from pts.schemas.dataset.base import DatasetModel
from pts.schemas.dataset.checks import registered_checks
from pts.schemas.dataset.discovery import discover_datasets
from pts.schemas.dataset.spec import FieldSpec, Node, dataset_spec


def dataset_doc(model: type[DatasetModel]) -> dict[str, Any]:
    """The documentation of one dataset, JSON-ready."""
    spec = dataset_spec(model)
    return {
        'name': spec.name,
        'description': spec.description,
        'primary_key': list(spec.primary_key),
        'strict': True,
        'fields': [_field_doc(f, top_level=True) for f in spec.fields],
        'checks': [
            {'name': check.name, 'column': check.column, 'description': check.description}
            for check in registered_checks(model)
        ],
        'example': spec.example,
    }


def write_docs(directory: Path) -> list[Path]:
    """Write the documentation of every discovered dataset to `directory`, one file per dataset.

    Returns:
        The written paths.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, model in sorted(discover_datasets().items()):
        path = directory / f'{name}.json'
        path.write_text(render(dataset_doc(model)))
        written.append(path)
    return written


def render(doc: dict[str, Any]) -> str:
    """Serialize a document the way it is stored on disk."""
    return json.dumps(doc, indent=2, ensure_ascii=False) + '\n'


def _field_doc(f: FieldSpec, *, top_level: bool) -> dict[str, Any]:
    doc: dict[str, Any] = {'name': f.name}
    if f.description is not None:
        doc['description'] = f.description
    if top_level:
        doc['required'] = f.required
    if f.primary_key:
        doc['primary_key'] = True
    if f.note is not None:
        doc['note'] = f.note
    return doc | _node_doc(f.node)


def _node_doc(node: Node) -> dict[str, Any]:
    doc: dict[str, Any] = {'nullable': node.nullable, 'type': _type_doc(node)}
    if constraints := node.constraints.as_dict():
        doc['constraints'] = constraints
    if node.reference is not None:
        doc['reference'] = {'dataset': node.reference.dataset, 'column': node.reference.column}
    if node.bioregistry is not None:
        doc['bioregistry'] = node.bioregistry
    return doc


def _type_doc(node: Node) -> str | dict[str, Any]:
    if node.element is not None:
        return {'type': 'list', 'element': _node_doc(node.element)}
    if node.kind == 'struct':
        return {'type': 'struct', 'fields': [_field_doc(f, top_level=False) for f in node.fields]}
    return str(node.dtype)


if __name__ == '__main__':
    for written in write_docs(Path(sys.argv[1])):
        print(written)  # noqa: T201
