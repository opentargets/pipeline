"""Tests for the ClickHouse declarations a dataset model carries."""

from typing import Annotated, ClassVar

import pytest

from ot_datamodels import ClickHouseDictionary, ClickHouseTable, DatasetModel, PrimaryKey, validate_declarations
from ot_datamodels.clickhouse import where_columns
from ot_datamodels.discovery import discover_models


def _model(*declarations: ClickHouseTable | ClickHouseDictionary) -> type[DatasetModel]:
    class Sample(DatasetModel):
        dataset_name: ClassVar[str] = 'sample'
        clickhouse: ClassVar[tuple[ClickHouseTable | ClickHouseDictionary, ...]] = declarations
        id: Annotated[str, PrimaryKey()]
        name: str
        extra: str | None = None

    return Sample


def _keyless(*declarations: ClickHouseTable) -> type[DatasetModel]:
    class Keyless(DatasetModel):
        dataset_name: ClassVar[str] = 'keyless'
        clickhouse: ClassVar[tuple[ClickHouseTable, ...]] = declarations
        id: str

    return Keyless


def _two_keys(*declarations: ClickHouseTable) -> type[DatasetModel]:
    class TwoKeys(DatasetModel):
        dataset_name: ClassVar[str] = 'two_keys'
        clickhouse: ClassVar[tuple[ClickHouseTable, ...]] = declarations
        a: Annotated[str, PrimaryKey()]
        b: Annotated[str, PrimaryKey()]

    return TwoKeys


def test_valid_declarations_have_no_problems():
    model = _model(
        ClickHouseTable('sample', 'rocksdb', columns=('id', 'name'), where="name != 'x' AND extra IS NOT NULL"),
        ClickHouseTable('sample_by_name', 'mergetree', order_by=('name', 'id')),
        ClickHouseDictionary(
            'sample_names', 'sample', key='id', attributes=('title',), source_columns={'title': 'name'}
        ),
    )
    assert validate_declarations(model) == []


def test_a_table_takes_only_name_engine_order_by_columns_and_where():
    with pytest.raises(TypeError):
        ClickHouseTable('t', 'rocksdb', key='id')  # ty: ignore[unknown-argument]
    assert ClickHouseTable.__dataclass_fields__.keys() == {'name', 'engine', 'order_by', 'columns', 'where'}


def test_rocksdb_needs_one_primary_key_column():
    assert validate_declarations(_keyless(ClickHouseTable('t', 'rocksdb'))) == [
        'Keyless.t: rocksdb needs one PrimaryKey column on the model, it has none'
    ]
    assert validate_declarations(_two_keys(ClickHouseTable('t', 'rocksdb'))) == [
        "TwoKeys.t: rocksdb needs one PrimaryKey column on the model, it has ('a', 'b')"
    ]


def test_rocksdb_refuses_an_order_by():
    table = ClickHouseTable('t', 'rocksdb', order_by=('id',))
    assert validate_declarations(_model(table)) == ['Sample.t: rocksdb takes no order_by']


def test_rocksdb_keeps_its_key_column():
    table = ClickHouseTable('t', 'rocksdb', columns=('name',))
    assert validate_declarations(_model(table)) == ["Sample.t: key 'id' is not a column of the table"]


def test_mergetree_needs_an_order_by():
    assert validate_declarations(_model(ClickHouseTable('t', 'mergetree'))) == ['Sample.t: mergetree needs an order_by']


def test_mergetree_needs_no_primary_key_column():
    assert validate_declarations(_keyless(ClickHouseTable('t', 'mergetree', order_by=('id',)))) == []


def test_unknown_columns_are_named():
    table = ClickHouseTable('t', 'mergetree', order_by=('nada',), columns=('id', 'zip'), where='zap IS NULL')
    problems = validate_declarations(_model(table))
    for column in ('nada', 'zip', 'zap'):
        assert any(f"'{column}'" in p and p.startswith('Sample.t: ') for p in problems), column


def test_order_by_dropped_by_columns_is_a_problem():
    table = ClickHouseTable('t', 'mergetree', order_by=('name',), columns=('id',))
    assert validate_declarations(_model(table)) == ["Sample.t: order_by 'name' is not a column of the table"]


def test_where_columns_are_the_identifiers_that_are_not_keywords_functions_or_strings():
    where = "lower(`name`) != 'Id here' AND extra IS NOT NULL OR id IN ('a', 'b') AND NOT empty(extra)"
    assert where_columns(where) == {'name', 'extra', 'id'}


def test_where_may_name_a_column_the_table_drops():
    table = ClickHouseTable('t', 'rocksdb', columns=('id',), where='extra IS NOT NULL')
    assert validate_declarations(_model(table)) == []


def _dictionary_problems(**dictionary) -> list[str]:
    table = ClickHouseTable('sample', 'rocksdb', columns=('id', 'name'))
    return validate_declarations(_model(table, ClickHouseDictionary('d', 'sample', **dictionary)))


def test_dictionary_column_dropped_by_columns_is_a_problem():
    problems = _dictionary_problems(key='id', attributes=('extra',))
    assert problems == ["Sample.d: column 'extra' is not a column of 'sample'"]


def test_dictionary_source_columns_name_the_source_table_columns():
    assert _dictionary_problems(key='id', attributes=('label',), source_columns={'label': 'name'}) == []
    problems = _dictionary_problems(key='id', attributes=('title',), source_columns={'title': 'extra'})
    assert problems == ["Sample.d: column 'title' is not a column of 'sample'"]


def test_dictionary_source_must_be_declared_on_the_model():
    problems = validate_declarations(_model(ClickHouseDictionary('d', 'missing', key='id', attributes=('name',))))
    assert problems == ["Sample.d: source_table 'missing' is not declared on the model"]


def test_every_declaration_is_valid():
    problems = [p for model in discover_models().values() for p in validate_declarations(model)]
    assert problems == []
