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
from pos.clickhouse_ddl import (
    CHType,
    column_type,
    columns,
    create_table,
    ddl_for,
    landing_columns,
    members,
    parse_table,
    parse_type,
    render,
    type_at,
    union_columns,
)

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
        ('struct', 'Tuple(`a` String, `b` Int64)'),
        ('nullable_struct', 'Tuple(`a` Nullable(String), `b` Nullable(Int64))'),
    ],
)
def test_column_type(name, expected):
    assert _rendered(name) == expected


def test_low_cardinality_nullable_order():
    assert _rendered('low_cardinality') == 'LowCardinality(Nullable(String))'


def test_list_of_nullable_structs():
    assert _rendered('bars') == 'Array(Tuple(`a` Nullable(String)))'


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
        's': 'Tuple(`a` Nullable(String))',
        'l': 'Array(String)',
        'only_a': 'Nullable(Int64)',
        'only_b': 'Nullable(Float64)',
    }


def test_union_type_clash_raises():
    with pytest.raises(ValueError, match=r'^column x:') as info:
        union_columns([ClashA, ClashB])
    assert 'clash_a' in str(info.value)
    assert 'clash_b' in str(info.value)


class NullableBar(DatasetModel):
    dataset_name = 'nullable_bar'
    dataset_title = 'N'
    dataset_description = 'N.'
    example = {}

    s: Bar | None


class PlainBar(DatasetModel):
    dataset_name = 'plain_bar'
    dataset_title = 'P'
    dataset_description = 'P.'
    example = {}

    s: Bar


def test_union_nested_field_nullability():
    union = union_columns([PlainBar, NullableBar])
    assert [(n, render(t)) for n, t in union] == [('s', 'Tuple(`a` Nullable(String))')]


class Quoted(DatasetModel):
    dataset_name = 'quoted'
    dataset_title = 'Q'
    dataset_description = 'Q.'
    example = {}

    label: Annotated[Literal["it's", 'back\\slash'], PolarsDtype(pl.Enum)]


def test_enum_labels_round_trip():
    t = columns(Quoted)[0][1]
    assert render(t) == r"Enum8('it\'s' = 1, 'back\\slash' = 2)"
    assert parse_type(render(t)) == t


BigLiteral = Literal[tuple(f'v{i}' for i in range(128))]  # ty: ignore[invalid-type-form]


class BigEnum(StructModel):
    a: Annotated[BigLiteral, PolarsDtype(pl.Enum)]


class BigEnumDataset(DatasetModel):
    dataset_name = 'big_enum'
    dataset_title = 'B'
    dataset_description = 'B.'
    example = {}

    s: BigEnum


def test_enum_too_large_names_the_path():
    with pytest.raises(ValueError, match=r'^s/a: 128 Enum categories'):
        columns(BigEnumDataset)


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


def test_landing_columns(monkeypatch):
    models = {m.dataset_name: m for m in (MemberB, MemberA, PppOnly, Toy)}
    monkeypatch.setattr(clickhouse_ddl, 'discover_datasets', lambda: models)
    glob = {'input_dir': 'output', 'models': 'member_*'}
    assert landing_columns(glob, 'platform') == union_columns([MemberA, MemberB])
    assert landing_columns({'input_dir': 'output/x', 'model': 'toy_ppp'}, 'platform') == columns(PppOnly)
    assert landing_columns({'input_dir': 'view/toy', 'preload_script': 'toy.sql'}, 'platform') is None
    assert landing_columns({'input_dir': 'output/toy'}, 'platform') is None


def test_ddl_for(monkeypatch):
    models = {m.dataset_name: m for m in (MemberB, MemberA, PppOnly, Toy)}
    monkeypatch.setattr(clickhouse_ddl, 'discover_datasets', lambda: models)
    one = {'table': 't_log', 'model': 'toy_ppp', 'engine': 'MergeTree ORDER BY id'}
    assert ddl_for(one, 'platform') == [('t_log', create_table('t_log', columns(PppOnly), 'MergeTree ORDER BY id'))]
    union = {'table': 'u_log', 'models': 'member_*'}
    assert ddl_for(union, 'platform') == [
        ('member_a_log', create_table('member_a_log', columns(MemberA))),
        ('member_b_log', create_table('member_b_log', columns(MemberB))),
        ('u_log', create_table('u_log', union_columns([MemberA, MemberB]))),
    ]
    assert ddl_for({'table': 'v', 'preload_script': 'v.sql'}, 'platform') == []
    with pytest.raises(ValueError, match=r"^p_log: models 'toy_\*' matches no model for platform$"):
        ddl_for({'table': 'p_log', 'models': 'toy_*'}, 'platform')


def test_pos_ddl_prints_one_dataset(monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr('sys.argv', ['pos-ddl', 'disease'])
    clickhouse_ddl.main()
    out = capsys.readouterr().out
    assert out.startswith('CREATE TABLE IF NOT EXISTS `disease_log` (')


def test_pos_ddl_evidence_follows_the_product(monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr('sys.argv', ['pos-ddl', 'evidence', '--product', 'ppp'])
    clickhouse_ddl.main()
    ppp = capsys.readouterr().out
    monkeypatch.setattr('sys.argv', ['pos-ddl', 'evidence'])
    clickhouse_ddl.main()
    platform = capsys.readouterr().out
    assert '`assays`' in ppp
    assert '`assays`' not in platform
    assert '`evidence_europepmc_log`' in platform
    assert platform.count('CREATE TABLE') == len(members('evidence_*', 'platform')) + 1


def test_pos_ddl_all_prints_every_modelled_entry(monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr('sys.argv', ['pos-ddl', '--all'])
    clickhouse_ddl.main()
    out = capsys.readouterr().out
    assert '`disease_log`' in out
    assert '`evidence_log`' in out
    assert '`targets_log`' not in out  # hand-written


def test_type_at():
    cols = columns(Toy)
    assert type_at(cols, 'tags') == dict(cols)['tags']
    assert type_at(cols, 'tags[]') == CHType('String')
    assert type_at(cols, 'nested[][]') == CHType('Int64')
    assert type_at(cols, 'struct/b') == CHType('Int64')
    assert type_at(cols, 'nullable_struct/a') == CHType('String', nullable=True)
    assert type_at(cols, 'bars[]/a') == CHType('String', nullable=True)
    assert type_at(cols, '') == CHType('Tuple', fields=tuple(cols))
    assert type_at(cols, 'absent') is None
    assert type_at(cols, 'struct/absent') is None
    assert type_at(cols, 'absent[]/a') is None


def test_create_table_quotes_identifiers():
    sql = create_table('toy_log', [('id', column_type(dataset_spec(MemberA).fields[0].node))])
    assert sql == 'CREATE TABLE IF NOT EXISTS `toy_log` (`id` String) ENGINE = Log'


def test_every_entry_round_trips():
    entries = yaml.safe_load((ROOT / 'config' / 'datasets.yaml').read_text())['clickhouse']
    models = discover_datasets()
    checked = 0
    for name, entry in entries.items():
        if 'model' not in entry:
            continue
        cols = columns(models[entry['model']])
        assert list(parse_table(create_table(entry['table'], cols)).items()) == cols, name
        checked += 1
    assert checked > 20
