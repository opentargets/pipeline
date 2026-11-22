"""pos.clickhouse_ddl: ClickHouse landing DDL generated from the dataset models."""

import datetime
from pathlib import Path
from typing import Annotated, Literal

import polars as pl
import pytest
import yaml
from ot_datamodels.base import ClickHouseType, DatasetModel, PolarsDtype, StructModel
from ot_datamodels.discovery import discover_datasets
from ot_datamodels.spec import dataset_spec

from pos import clickhouse_ddl
from pos.clickhouse_ddl import column_type, columns, create_table, members, parse_table, render, union_columns

ROOT = Path(__file__).parents[1]


class Foo(StructModel):
    a: str
    b: int


class Bar(StructModel):
    a: str


class Toy(DatasetModel):
    dataset_name = 'toy'
    dataset_title = 'Toy'
    dataset_description = 'Toy.'
    example = {}

    marked: Annotated[int, ClickHouseType('UInt32')]
    marked_nullable: Annotated[int | None, ClickHouseType('UInt32')]
    literal: Literal['a', 'b']
    enum: Annotated[Literal['x', 'y'], PolarsDtype(pl.Enum)]
    flag: bool
    day: datetime.date
    score: float | None
    tags: list[str]
    maybe_tags: list[str | None]
    nested: list[list[int]]
    struct: Foo
    nullable_struct: Foo | None
    low_cardinality: Annotated[str | None, ClickHouseType('LowCardinality(String)')]
    bars: list[Bar | None]


def _rendered(name: str) -> str:
    node = next(f.node for f in dataset_spec(Toy).fields if f.name == name)
    return render(column_type(node))


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        ('marked', 'UInt32'),
        ('marked_nullable', 'Nullable(UInt32)'),
        ('literal', 'String'),
        ('enum', "Enum8('x' = 1, 'y' = 2)"),
        ('flag', 'Bool'),
        ('day', 'Date'),
        ('score', 'Nullable(Float64)'),
        ('tags', 'Array(String)'),
        ('maybe_tags', 'Array(Nullable(String))'),
        ('nested', 'Array(Array(Int64))'),
        ('struct', 'Tuple(a String, b Int64)'),
        ('nullable_struct', 'Tuple(a Nullable(String), b Nullable(Int64))'),
    ],
)
def test_column_type(name, expected):
    assert _rendered(name) == expected


def test_low_cardinality_nullable_order():
    assert _rendered('low_cardinality') == 'LowCardinality(Nullable(String))'


def test_list_of_nullable_structs():
    assert _rendered('bars') == 'Array(Tuple(a Nullable(String)))'


def test_columns_in_model_order():
    assert [name for name, _ in columns(Toy)] == [f.name for f in dataset_spec(Toy).fields]


class MemberA(DatasetModel):
    dataset_name = 'member_a'
    dataset_title = 'A'
    dataset_description = 'A.'
    example = {}

    id: str
    s: Bar
    l: list[str]  # noqa: E741
    only_a: int


class MemberB(DatasetModel):
    dataset_name = 'member_b'
    dataset_title = 'B'
    dataset_description = 'B.'
    example = {}

    id: str
    only_b: float | None


class ClashA(DatasetModel):
    dataset_name = 'clash_a'
    dataset_title = 'A'
    dataset_description = 'A.'
    example = {}

    x: int


class ClashB(DatasetModel):
    dataset_name = 'clash_b'
    dataset_title = 'B'
    dataset_description = 'B.'
    example = {}

    x: str


def test_union_absent_struct_and_list():
    union = {name: render(t) for name, t in union_columns([MemberA, MemberB])}
    assert list(union) == ['id', 's', 'l', 'only_a', 'only_b']
    assert union == {
        'id': 'String',
        's': 'Tuple(a Nullable(String))',
        'l': 'Array(String)',
        'only_a': 'Nullable(Int64)',
        'only_b': 'Nullable(Float64)',
    }


def test_union_type_clash_raises():
    with pytest.raises(ValueError, match='x') as info:
        union_columns([ClashA, ClashB])
    assert 'clash_a' in str(info.value)
    assert 'clash_b' in str(info.value)


class PppOnly(DatasetModel):
    dataset_name = 'toy_ppp'
    dataset_title = 'PPP'
    dataset_description = 'PPP.'
    products = ('ppp',)
    example = {}

    id: str


def test_members_filters_product(monkeypatch):
    models = {m.dataset_name: m for m in (MemberB, MemberA, PppOnly, Toy)}
    monkeypatch.setattr(clickhouse_ddl, 'discover_datasets', lambda: models)
    assert members('*_*', 'platform') == [MemberA, MemberB]
    assert members('*_*', 'ppp') == [MemberA, MemberB, PppOnly]


def test_create_table_quotes_identifiers():
    sql = create_table('toy_log', [('id', column_type(dataset_spec(MemberA).fields[0].node))])
    assert sql == 'CREATE TABLE IF NOT EXISTS `toy_log` (`id` String) ENGINE = Log'


def test_every_entry_round_trips():
    entries = yaml.safe_load((ROOT / 'config' / 'datasets.yaml').read_text())['clickhouse']
    models = discover_datasets()
    checked = 0
    for name, entry in entries.items():
        input_dir = entry.get('input_dir', '')
        model = models.get(input_dir.removeprefix('output/')) if input_dir.startswith('output/') else None
        if name == 'evidence' or model is None:
            continue
        cols = columns(model)
        assert parse_table(create_table(entry['table'], cols)) == dict(cols), name
        checked += 1
    assert checked > 20
