"""Tests over every discovered dataset model -- a new model is covered without being listed here."""

import json
from pathlib import Path

import pytest

from ot_datamodels import DatasetModel, Reference
from ot_datamodels.discovery import discover_datasets
from ot_datamodels.docs import write_docs
from ot_datamodels.spec import Node, dataset_spec, walk

DATASETS = discover_datasets()
REFERENCES = [
    (model, path, node)
    for model in sorted(DATASETS.values(), key=lambda model: model.dataset_name)
    for path, node in walk(dataset_spec(model))
    if node.reference is not None
]


def test_discovers_the_migrated_datasets() -> None:
    assert {'biosample', 'evidence_encore', 'evidence_gwas_credible_sets'} <= set(DATASETS)


def test_documentation_is_written_for_every_dataset(tmp_path: Path) -> None:
    written = write_docs(tmp_path)

    assert sorted(path.stem for path in written) == sorted(DATASETS)
    assert all(json.loads(path.read_text())['fields'] for path in written)


@pytest.mark.parametrize(
    ('model', 'path', 'node'),
    REFERENCES,
    ids=[f'{m.dataset_name}:{p}' for m, p, _ in REFERENCES],
)
def test_reference_resolves_to_a_column(model: type[DatasetModel], path: str, node: Node) -> None:
    """A reference is a look-up into another dataset: often its primary key, but not necessarily (croissant's
    foreign keys also point at a column of a composite key, or at a dataset with no key)."""
    reference = node.reference
    assert isinstance(reference, Reference)
    target = DATASETS.get(reference.dataset)
    if target is None:
        pytest.skip(f'{reference.dataset!r} has no model yet')

    target_spec = dataset_spec(target)
    target_field = next((f for f in target_spec.fields if f.name == reference.column), None)

    assert target_field is not None, f'{reference.dataset} has no column {reference.column!r}'
    assert target_field.node.dtype == node.dtype


def test_every_dataset_has_a_title() -> None:
    for name, model in DATASETS.items():
        assert model.dataset_title, name
