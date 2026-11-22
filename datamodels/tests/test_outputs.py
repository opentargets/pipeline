"""Tests over every discovered dataset model -- a new model is covered without being listed here."""

import json
from pathlib import Path

import pytest

from ot_datamodels import DatasetModel, Reference
from ot_datamodels.discovery import discover_datasets, discover_models
from ot_datamodels.docs import write_docs
from ot_datamodels.spec import Node, dataset_spec, walk

DATASETS = discover_datasets()
MODELS = discover_models()
REFERENCES = [
    (model, path, node)
    for model in sorted(MODELS.values(), key=lambda model: model.dataset_name)
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


def test_products_default_and_ppp_only() -> None:
    assert DATASETS['disease'].products == ('platform', 'ppp')
    ppp_only = {name for name, model in DATASETS.items() if model.products == ('ppp',)}
    assert ppp_only == {'evidence_encore', 'evidence_ot_crispr', 'evidence_ot_crispr_validation', 'otar'}
    for model in DATASETS.values():
        assert model.products
        assert set(model.products) <= {'platform', 'ppp'}


INTEGER_RANGES = {
    'UInt8': (0, 2**8 - 1),
    'UInt16': (0, 2**16 - 1),
    'UInt32': (0, 2**32 - 1),
    'UInt64': (0, 2**64 - 1),
    'Int8': (-(2**7), 2**7 - 1),
    'Int16': (-(2**15), 2**15 - 1),
    'Int32': (-(2**31), 2**31 - 1),
    'Int64': (-(2**63), 2**63 - 1),
}
INTEGER_MARKERS = [
    (model, path, node)
    for model in sorted(MODELS.values(), key=lambda model: model.dataset_name)
    for path, node in walk(dataset_spec(model))
    if node.clickhouse in INTEGER_RANGES
]


@pytest.mark.parametrize(('model', 'path', 'node'), INTEGER_MARKERS)
def test_integer_markers_have_bounds(model: type[DatasetModel], path: str, node: Node) -> None:
    """An integer marker whose range does not cover its dtype's has bounds inside the marker's range."""
    assert node.clickhouse is not None
    low, high = INTEGER_RANGES[node.clickhouse]
    dtype_low, dtype_high = INTEGER_RANGES[str(node.dtype)]
    if low <= dtype_low and dtype_high <= high:
        return
    c = node.constraints
    where = f'{model.dataset_name} {path}'
    if c.allowed is not None:
        assert all(isinstance(v, int) and low <= v <= high for v in c.allowed), where
        return
    lower = dtype_low
    upper = dtype_high
    for bound, value in (('ge', 0), ('gt', 1)):
        if (limit := getattr(c, bound)) is not None:
            lower = max(lower, int(limit) + value)
    for bound, value in (('le', 0), ('lt', -1)):
        if (limit := getattr(c, bound)) is not None:
            upper = min(upper, int(limit) + value)
    assert lower >= low, f'{where}: lower bound below {low}'
    assert upper <= high, f'{where}: upper bound above {high}'
