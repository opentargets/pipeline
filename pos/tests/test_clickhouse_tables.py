"""pos.clickhouse_tables: the tables and dictionaries a `datasets.yaml` entry declares, and their checks."""

from typing import Annotated

import pytest
from ot_datamodels.base import DatasetModel, PrimaryKey

from pos import clickhouse_ddl
from pos.clickhouse_ddl import entry_problems
from pos.clickhouse_tables import (
    ClickHouseDictionary,
    ClickHouseTable,
    entry_dictionaries,
    entry_tables,
    where_columns,
)


class Sample(DatasetModel):
    dataset_name = 'sample'
    dataset_title = 'Sample'
    dataset_description = 'Sample.'
    example = {}

    id: Annotated[str, PrimaryKey()]
    name: str
    extra: str | None = None


class Keyless(DatasetModel):
    dataset_name = 'keyless'
    dataset_title = 'Keyless'
    dataset_description = 'Keyless.'
    example = {}

    id: str


class TwoKeys(DatasetModel):
    dataset_name = 'two_keys'
    dataset_title = 'Two keys'
    dataset_description = 'Two keys.'
    example = {}

    a: Annotated[str, PrimaryKey()]
    b: Annotated[str, PrimaryKey()]


@pytest.fixture(autouse=True)
def _models(monkeypatch):
    models = {'sample': Sample, 'keyless': Keyless, 'two_keys': TwoKeys}
    monkeypatch.setattr(clickhouse_ddl, 'discover_models', lambda: models)


def _problems(*tables: dict, model: str = 'sample', dictionaries: tuple[dict, ...] = ()) -> list[str]:
    entry = {'table': f'{model}_log', 'model': model, 'tables': list(tables), 'dictionaries': list(dictionaries)}
    return entry_problems(entry, 'platform')


def test_entry_tables_and_dictionaries_parse_the_yaml_lists():
    entry = {
        'table': 'sample_log',
        'tables': [
            {'name': 'sample', 'engine': 'rocksdb'},
            {'name': 'by_name', 'engine': 'mergetree', 'order_by': ['name'], 'columns': ['name'], 'where': 'x'},
        ],
        'dictionaries': [
            {'name': 'd', 'source': 'sample', 'key': 'id', 'attributes': ['title'], 'source_columns': {'title': 'name'}}
        ],
    }
    assert entry_tables(entry) == [
        ClickHouseTable('sample', 'rocksdb'),
        ClickHouseTable('by_name', 'mergetree', order_by=('name',), columns=('name',), where='x'),
    ]
    assert entry_dictionaries(entry) == [
        ClickHouseDictionary('d', 'sample', key='id', attributes=('title',), source_columns={'title': 'name'})
    ]
    assert entry_tables({'table': 't'}) == []
    assert entry_dictionaries({'table': 't'}) == []


def test_entry_dictionaries_take_only_the_dictionary_fields():
    entry = {'table': 'e_log', 'dictionaries': [{'name': 'd', 'source_table': 's', 'key': 'id', 'attributes': []}]}
    with pytest.raises(ValueError, match=r"e_log: unknown ClickHouseDictionary fields \['source_table'\]"):
        entry_dictionaries(entry)


def test_valid_declarations_have_no_problems():
    tables = (
        {
            'name': 'sample',
            'engine': 'rocksdb',
            'columns': ['id', 'name'],
            'where': "name != 'x' AND extra IS NOT NULL",
        },
        {'name': 'sample_by_name', 'engine': 'mergetree', 'order_by': ['name', 'id']},
    )
    dictionary = {
        'name': 'sample_names',
        'source': 'sample',
        'key': 'id',
        'attributes': ['title'],
        'source_columns': {'title': 'name'},
    }
    assert _problems(*tables, dictionaries=(dictionary,)) == []


def test_rocksdb_needs_one_primary_key_column():
    assert _problems({'name': 't', 'engine': 'rocksdb'}, model='keyless') == [
        'keyless_log.t: rocksdb needs one PrimaryKey column on the model, it has none'
    ]
    assert _problems({'name': 't', 'engine': 'rocksdb'}, model='two_keys') == [
        "two_keys_log.t: rocksdb needs one PrimaryKey column on the model, it has ('a', 'b')"
    ]


def test_rocksdb_refuses_an_order_by():
    assert _problems({'name': 't', 'engine': 'rocksdb', 'order_by': ['id']}) == [
        'sample_log.t: rocksdb takes no order_by'
    ]


def test_rocksdb_keeps_its_key_column():
    assert _problems({'name': 't', 'engine': 'rocksdb', 'columns': ['name']}) == [
        "sample_log.t: key 'id' is not a column of the table"
    ]


def test_mergetree_needs_an_order_by():
    assert _problems({'name': 't', 'engine': 'mergetree'}) == ['sample_log.t: mergetree needs an order_by']


def test_mergetree_needs_no_primary_key_column():
    assert _problems({'name': 't', 'engine': 'mergetree', 'order_by': ['id']}, model='keyless') == []


def test_unknown_columns_are_named():
    table = {'name': 't', 'engine': 'mergetree', 'order_by': ['nada'], 'columns': ['id', 'zip'], 'where': 'zap IS NULL'}
    problems = _problems(table)
    for column in ('nada', 'zip', 'zap'):
        assert any(f"'{column}'" in p and p.startswith('sample_log.t: ') for p in problems), column


def test_order_by_dropped_by_columns_is_a_problem():
    assert _problems({'name': 't', 'engine': 'mergetree', 'order_by': ['name'], 'columns': ['id']}) == [
        "sample_log.t: order_by 'name' is not a column of the table"
    ]


def test_where_columns_are_the_identifiers_that_are_not_keywords_functions_or_strings():
    where = "lower(`name`) != 'Id here' AND extra IS NOT NULL OR id IN ('a', 'b') AND NOT empty(extra)"
    assert where_columns(where) == {'name', 'extra', 'id'}


def test_where_may_name_a_column_the_table_drops():
    assert _problems({'name': 't', 'engine': 'rocksdb', 'columns': ['id'], 'where': 'extra IS NOT NULL'}) == []


def _dictionary_problems(**dictionary) -> list[str]:
    table = {'name': 'sample', 'engine': 'rocksdb', 'columns': ['id', 'name']}
    return _problems(table, dictionaries=({'name': 'd', 'source': 'sample'} | dictionary,))


def test_dictionary_column_dropped_by_columns_is_a_problem():
    problems = _dictionary_problems(key='id', attributes=['extra'])
    assert problems == ["sample_log.d: column 'extra' is not a column of 'sample'"]


def test_dictionary_source_columns_name_the_source_table_columns():
    assert _dictionary_problems(key='id', attributes=['label'], source_columns={'label': 'name'}) == []
    problems = _dictionary_problems(key='id', attributes=['title'], source_columns={'title': 'extra'})
    assert problems == ["sample_log.d: column 'title' is not a column of 'sample'"]


def test_dictionary_source_must_be_a_table_of_the_entry():
    dictionary = {'name': 'd', 'source': 'missing', 'key': 'id', 'attributes': ['name']}
    assert _problems(dictionaries=(dictionary,)) == ["sample_log.d: source 'missing' is not a table of the entry"]


def test_tables_with_a_postload_script_are_a_problem():
    entry = {
        'table': 'sample_log',
        'model': 'sample',
        'postload_script': 'post.sql',
        'tables': [{'name': 'sample', 'engine': 'rocksdb'}],
    }
    assert entry_problems(entry, 'platform') == ['sample_log: declares ClickHouse tables and has a postload_script']
