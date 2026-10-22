"""Tests over every discovered dataset model -- a new model is covered without being listed here.

`test_real_output_is_valid` validates actual step outputs, which no fixture here can stand in for
(e.g. struct field order, which only real data fixes). Point `PTS_OUTPUT_DIR` at a local output
directory holding one folder per dataset to run it::

    PTS_OUTPUT_DIR=../work/output uv run pytest tests/test_schema_outputs.py
"""

import json
import os
from pathlib import Path
from typing import get_args

import polars as pl
import pytest

from pts.schemas.dataset import DatasetModel, Reference, to_pandera, to_polars_schema
from pts.schemas.dataset.discovery import discover_datasets
from pts.schemas.dataset.docs import write_docs
from pts.schemas.dataset.spec import Node, dataset_spec, walk
from pts.schemas.outputs.evidence_base import QualityFlag
from pts.transformers.evidence.utils.evidence import EvidenceFlags

DATASETS = discover_datasets()
MODELS = sorted(DATASETS.values(), key=lambda model: model.dataset_name)
REFERENCES = [
    (model, path, node) for model in MODELS for path, node in walk(dataset_spec(model)) if node.reference is not None
]


def _name(model: type[DatasetModel]) -> str:
    return model.dataset_name


def test_discovers_the_migrated_datasets() -> None:
    assert {'biosample', 'evidence_encore', 'evidence_gwas_credible_sets'} <= set(DATASETS)


@pytest.mark.parametrize('model', MODELS, ids=_name)
def test_example_is_valid(model: type[DatasetModel]) -> None:
    model.model_validate(model.example)
    to_pandera(model).validate(pl.DataFrame([model.example], schema=to_polars_schema(model)))


def test_documentation_is_written_for_every_dataset(tmp_path: Path) -> None:
    written = write_docs(tmp_path)

    assert sorted(path.stem for path in written) == sorted(DATASETS)
    assert all(json.loads(path.read_text())['fields'] for path in written)


@pytest.mark.parametrize(
    ('model', 'path', 'node'),
    REFERENCES,
    ids=[f'{m.dataset_name}:{p}' for m, p, _ in REFERENCES],
)
def test_reference_resolves_to_a_primary_key(model: type[DatasetModel], path: str, node: Node) -> None:
    reference = node.reference
    assert isinstance(reference, Reference)
    target = DATASETS.get(reference.dataset)
    if target is None:
        pytest.skip(f'{reference.dataset!r} has no model yet')

    target_spec = dataset_spec(target)
    target_field = next((f for f in target_spec.fields if f.name == reference.column), None)

    assert target_field is not None, f'{reference.dataset} has no column {reference.column!r}'
    assert target_spec.primary_key == (reference.column,), (
        f'{reference.dataset}.{reference.column} is not its primary key'
    )
    assert target_field.node.dtype == node.dtype


def test_quality_flag_vocabulary_matches_the_evidence_flags() -> None:
    assert set(get_args(QualityFlag)) == {flag.value for flag in EvidenceFlags}


@pytest.mark.parametrize('model', MODELS, ids=_name)
def test_real_output_is_valid(model: type[DatasetModel]) -> None:
    output_dir = os.environ.get('PTS_OUTPUT_DIR')
    if not output_dir:
        pytest.skip('PTS_OUTPUT_DIR is not set')
    location = Path(output_dir) / model.dataset_name
    if not location.exists():
        pytest.skip(f'no {location}')

    to_pandera(model).validate(pl.read_parquet(location), lazy=True)


def test_every_dataset_has_a_title_and_tags() -> None:
    for name, model in discover_datasets().items():
        assert model.dataset_title, name
        assert model.dataset_tags, name
