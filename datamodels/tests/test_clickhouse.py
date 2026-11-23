"""Tests for the ClickHouse declarations a dataset model carries."""

from typing import ClassVar

from ot_datamodels import ClickHouseDictionary, ClickHouseTable, DatasetModel, validate_declarations
from ot_datamodels.discovery import discover_models


def _model(*declarations: ClickHouseTable | ClickHouseDictionary) -> type[DatasetModel]:
    class Sample(DatasetModel):
        dataset_name: ClassVar[str] = 'sample'
        clickhouse: ClassVar[tuple[ClickHouseTable | ClickHouseDictionary, ...]] = declarations
        id: str
        name: str
        extra: str

    return Sample


def test_valid_declarations_have_no_problems():
    model = _model(
        ClickHouseTable('sample', 'rocksdb', key='id', columns=('id', 'name'), renames={'name': 'label'}),
        ClickHouseTable('sample_by_name', 'mergetree', order_by=('name', 'id'), primary_key=('name',)),
        ClickHouseDictionary(
            'sample_names', 'sample', key='id', attributes=('label',), source_columns={'label': 'name'}
        ),
    )
    assert validate_declarations(model) == []


def test_rocksdb_needs_a_key():
    assert validate_declarations(_model(ClickHouseTable('t', 'rocksdb'))) == ['Sample.t: rocksdb needs a key']


def test_mergetree_needs_an_order_by():
    assert validate_declarations(_model(ClickHouseTable('t', 'mergetree'))) == ['Sample.t: mergetree needs an order_by']


def test_unknown_columns_are_named():
    table = ClickHouseTable('t', 'rocksdb', key='nope', order_by=('nada',), columns=('id', 'zip'), renames={'zap': 'x'})
    problems = validate_declarations(_model(table))
    for column in ('nope', 'nada', 'zip', 'zap'):
        assert any(f"'{column}'" in p and p.startswith('Sample.t: ') for p in problems), column


def test_renamed_column_is_not_a_source_column_any_more():
    table = ClickHouseTable('t', 'rocksdb', key='name', renames={'name': 'label'})
    assert validate_declarations(_model(table)) == ["Sample.t: key 'name' is not a column of the table"]


def test_primary_key_must_prefix_order_by():
    table = ClickHouseTable('t', 'mergetree', order_by=('id', 'name'), primary_key=('name',))
    assert validate_declarations(_model(table)) == ['Sample.t: primary_key is not a prefix of order_by']


def test_dictionary_source_must_be_declared_on_the_model():
    problems = validate_declarations(_model(ClickHouseDictionary('d', 'missing', key='id', attributes=('name',))))
    assert problems == ["Sample.d: source_table 'missing' is not declared on the model"]


def test_every_declaration_is_valid():
    problems = [p for model in discover_models().values() for p in validate_declarations(model)]
    assert problems == []
