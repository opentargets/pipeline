"""pos.clickhouse_ddl: ClickHouse landing DDL generated from the dataset models."""

import datetime
from pathlib import Path
from typing import Annotated, Literal

import polars as pl
import pytest
import yaml
from ot_datamodels.base import ClickHouseType, DatasetModel, PolarsDtype, StructModel
from ot_datamodels.clickhouse import ClickHouseDictionary, ClickHouseTable
from ot_datamodels.discovery import VIEWS_PACKAGE, discover_datasets, discover_models
from ot_datamodels.spec import dataset_spec
from ot_datamodels.views.target_view import TargetView

from pos import clickhouse_ddl
from pos.clickhouse_ddl import (
    CHType,
    canonical,
    column_type,
    columns,
    create_table,
    ddl_for,
    declarations,
    declared_tables,
    dictionary_sql,
    drop_dictionary_sql,
    landing_columns,
    member_table,
    members,
    model_type,
    parse_table,
    parse_type,
    render,
    table_sql,
    type_at,
    union_columns,
    uniqueness_sql,
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
    assert model_type(dataset_spec(Quoted).fields[0].node) == t


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
    monkeypatch.setattr(clickhouse_ddl, 'discover_models', lambda: models)
    assert members('*_*', 'platform') == [MemberA, MemberB]
    assert members('*_*', 'ppp') == [MemberA, MemberB, PppOnly]


def test_landing_columns(monkeypatch):
    models = {m.dataset_name: m for m in (MemberB, MemberA, PppOnly, Toy)}
    monkeypatch.setattr(clickhouse_ddl, 'discover_models', lambda: models)
    glob = {'input_dir': 'output', 'models': 'member_*'}
    assert landing_columns(glob, 'platform') == union_columns([MemberA, MemberB])
    assert landing_columns({'input_dir': 'output/x', 'model': 'toy_ppp'}, 'platform') == columns(PppOnly)
    assert landing_columns({'input_dir': 'view/toy', 'preload_script': 'toy.sql'}, 'platform') is None
    assert landing_columns({'input_dir': 'output/toy'}, 'platform') is None


def test_ddl_for(monkeypatch):
    models = {m.dataset_name: m for m in (MemberB, MemberA, PppOnly, Toy)}
    monkeypatch.setattr(clickhouse_ddl, 'discover_models', lambda: models)
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
    assert '`targets_log`' in out


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


def test_canonical_numbers_enums():
    numbered = parse_type("Enum8('a' = 1, 'b' = 2)")
    assert canonical(parse_type("Enum('a', 'b')")) == numbered
    assert canonical(numbered) == numbered
    assert canonical(parse_type("Enum8('b' = 1, 'a' = 2)")) != numbered
    assert canonical(parse_type("Enum('a' = 5, 'b')")) == parse_type("Enum8('a' = 5, 'b' = 6)")
    assert canonical(parse_type("Enum('a' = 200)")) == parse_type("Enum16('a' = 200)")
    nested = canonical(parse_type("Array(Tuple(x Nullable(Enum('a', 'b'))))"))
    assert nested == parse_type("Array(Tuple(x Nullable(Enum8('a' = 1, 'b' = 2))))")


def test_create_table_quotes_identifiers():
    sql = create_table('toy_log', [('id', column_type(dataset_spec(MemberA).fields[0].node))])
    assert sql == 'CREATE TABLE IF NOT EXISTS `toy_log` (`id` String) ENGINE = Log'


def test_every_entry_round_trips():
    entries = yaml.safe_load((ROOT / 'config' / 'datasets.yaml').read_text())['clickhouse']
    models = discover_models()
    checked = 0
    for name, entry in entries.items():
        if 'model' not in entry:
            continue
        cols = columns(models[entry['model']])
        assert list(parse_table(create_table(entry['table'], cols)).items()) == cols, name
        checked += 1
    assert checked > 20
    views = {entries[n]['model'] for n in ('aotf', 'association_time_series', 'protein_coding_coords', 'targets')}
    assert views == set(discover_datasets(VIEWS_PACKAGE))
    for product in ('platform', 'ppp'):
        found = members(entries['evidence']['models'], product)
        assert found, product
        tables = [(member_table(m), columns(m)) for m in found]
        tables.append((entries['evidence']['table'], union_columns(found)))
        for table, cols in tables:
            assert list(parse_table(create_table(table, cols)).items()) == cols, (product, table)


#: The `targets` landing types the markers on the `target` model keep.
TARGET_VIEW_TYPES = {
    'genomicLocation/start': 'UInt32',
    'genomicLocation/end': 'UInt32',
    'genomicLocation/strand': "Enum8('-1' = -1, '0' = 0, '1' = 1)",
    'constraint[]/exp': 'Nullable(Float64)',
    'constraint[]/oe': 'Nullable(Float64)',
    'constraint[]/oeLower': 'Nullable(Float64)',
    'constraint[]/oeUpper': 'Nullable(Float64)',
    'constraint[]/score': 'Nullable(Float64)',
    'constraint[]/obs': 'Nullable(UInt32)',
    'constraint[]/upperBin': 'Nullable(UInt32)',
    'constraint[]/upperBin6': 'Nullable(UInt32)',
    'constraint[]/upperRank': 'Nullable(UInt32)',
    'hallmarks/attributes[]/pmid': 'Nullable(UInt32)',
    'hallmarks/cancerHallmarks[]/pmid': 'UInt32',
    'targetClass[]/id': 'UInt32',
}


@pytest.mark.parametrize(('path', 'expected'), TARGET_VIEW_TYPES.items())
def test_target_view_keeps_landing_types(path, expected):
    t = type_at(columns(TargetView), path)
    assert t is not None
    assert render(t) == expected


# declared tables

STR = CHType('String')
LANDING = [
    ('id', STR),
    ('drugId', STR),
    ('targetId', CHType('String', nullable=True)),
    ('score', CHType('Float64', nullable=True)),
    ('tags', CHType('Array', element=STR)),
]


def test_rocksdb_table_sql():
    table = ClickHouseTable('gene_ontology', 'rocksdb', key='id')
    assert table_sql(table, [('id', STR), ('name', STR)], 'gene_ontology_log') == [
        'DROP TABLE IF EXISTS `gene_ontology` SYNC',
        (
            'CREATE TABLE `gene_ontology` (`id` String, `name` String) '
            'ENGINE = EmbeddedRocksDB() PRIMARY KEY `id`'
        ),
        'INSERT INTO `gene_ontology` SELECT `id`, `name` FROM `gene_ontology_log`',
    ]


def test_mergetree_nullable_order_by_allows_nullable_key():
    table = ClickHouseTable('t', 'mergetree', order_by=('targetId', 'id'), columns=('id', 'targetId'))
    assert table_sql(table, LANDING, 't_log') == [
        'DROP TABLE IF EXISTS `t` SYNC',
        (
            'CREATE TABLE `t` (`id` String, `targetId` Nullable(String)) '
            'ENGINE = MergeTree() ORDER BY (`targetId`, `id`) SETTINGS allow_nullable_key = 1'
        ),
        'INSERT INTO `t` SELECT `id`, `targetId` FROM `t_log`',
    ]


def test_mergetree_primary_key_prefix_and_where_with_optimize():
    table = ClickHouseTable(
        'clinical_target',
        'mergetree',
        order_by=('targetId', 'id'),
        primary_key=('targetId',),
        columns=('id', 'drugId', 'targetId'),
        where='targetId IS NOT NULL',
        optimize=True,
    )
    assert table_sql(table, LANDING, 'clinical_target_log') == [
        'DROP TABLE IF EXISTS `clinical_target` SYNC',
        (
            'CREATE TABLE `clinical_target` (`id` String, `drugId` String, `targetId` Nullable(String)) '
            'ENGINE = MergeTree() ORDER BY (`targetId`, `id`) PRIMARY KEY (`targetId`) SETTINGS allow_nullable_key = 1'
        ),
        (
            'INSERT INTO `clinical_target` SELECT `id`, `drugId`, `targetId` FROM `clinical_target_log` '
            'WHERE targetId IS NOT NULL'
        ),
        'OPTIMIZE TABLE `clinical_target` FINAL',
    ]


def test_subset_rename_follow_columns_order():
    table = ClickHouseTable('t', 'mergetree', order_by=('gene',), columns=('score', 'id'), renames={'id': 'gene'})
    assert table_sql(table, LANDING, 'l') == [
        'DROP TABLE IF EXISTS `t` SYNC',
        (
            'CREATE TABLE `t` (`score` Nullable(Float64), `gene` String) '
            'ENGINE = MergeTree() ORDER BY (`gene`)'
        ),
        'INSERT INTO `t` SELECT `score`, `id` AS `gene` FROM `l`',
    ]


def test_no_columns_keeps_every_landing_column():
    table = ClickHouseTable('t', 'rocksdb', key='id')
    assert table_sql(table, LANDING, 'l')[2] == (
        'INSERT INTO `t` SELECT `id`, `drugId`, `targetId`, `score`, `tags` FROM `l`'
    )


def test_dictionary_sql_renames_through_expression():
    table = ClickHouseTable('biosample', 'rocksdb', key='biosampleId')
    d = ClickHouseDictionary(
        'biosample_names',
        'biosample',
        key='id',
        attributes=('name',),
        source_columns={'id': 'biosampleId', 'name': 'biosampleName'},
    )
    cols = [('biosampleId', STR), ('biosampleName', STR), ('other', STR)]
    assert dictionary_sql(d, cols) == (
        'CREATE OR REPLACE DICTIONARY `biosample_names` '
        '(`id` String EXPRESSION `biosampleId`, `name` String EXPRESSION `biosampleName`) PRIMARY KEY `id` '
        "SOURCE(CLICKHOUSE(TABLE 'biosample')) LAYOUT(COMPLEX_KEY_HASHED()) LIFETIME(0)"
    )
    assert table.name == d.source_table


def test_dictionary_sql_identity_columns_have_no_expression():
    d = ClickHouseDictionary('d', 't', key='id', attributes=('name',))
    assert dictionary_sql(d, [('id', STR), ('name', STR)]) == (
        'CREATE OR REPLACE DICTIONARY `d` (`id` String, `name` String) PRIMARY KEY `id` '
        "SOURCE(CLICKHOUSE(TABLE 't')) LAYOUT(COMPLEX_KEY_HASHED()) LIFETIME(0)"
    )


def test_dictionary_sql_drops_low_cardinality():
    d = ClickHouseDictionary('sequence_ontology', 'sequence_ontology_source', key='id', attributes=('label',))
    low = CHType('String', low_cardinality=True)
    assert dictionary_sql(d, [('id', low), ('label', low)]).startswith(
        'CREATE OR REPLACE DICTIONARY `sequence_ontology` (`id` String, `label` String) PRIMARY KEY `id` '
    )


def test_declared_allow_nullable_key_without_a_nullable_order_by():
    table = ClickHouseTable('t', 'mergetree', order_by=('id',), columns=('id',), allow_nullable_key=True)
    assert table_sql(table, LANDING, 'l')[1] == (
        'CREATE TABLE `t` (`id` String) ENGINE = MergeTree() ORDER BY (`id`) SETTINGS allow_nullable_key = 1'
    )


@pytest.mark.parametrize('product', ['platform', 'ppp'])
def test_union_declarations_name_union_columns(product):
    entries = yaml.safe_load((ROOT / 'config' / 'datasets.yaml').read_text())['clickhouse']
    for name, entry in entries.items():
        tables, _ = declarations(entry, product)
        if 'models' not in entry or not tables:
            continue
        union = {c for c, _ in landing_columns(entry, product) or []}
        for table in tables:
            used = {*(table.columns or ()), *table.order_by, *table.primary_key, *((table.key,) if table.key else ())}
            assert used <= union, f'{name}.{table.name}: {sorted(used - union)} not in the union'


def test_drop_dictionary_sql():
    d = ClickHouseDictionary('biosample_names', 'biosample', key='id', attributes=('name',))
    assert drop_dictionary_sql(d) == 'DROP DICTIONARY IF EXISTS `biosample_names` SYNC'


def test_uniqueness_sql():
    table = ClickHouseTable('biosample', 'rocksdb', key='biosampleId')
    assert uniqueness_sql(table, 'biosample_log') == (
        "SELECT throwIf(count() != uniqExact(tuple(`biosampleId`)), 'duplicate biosampleId in biosample') "
        'FROM `biosample_log`'
    )


def test_uniqueness_sql_reads_the_renamed_key_through_the_filter():
    table = ClickHouseTable('t', 'rocksdb', key='gene', renames={'id': 'gene'}, where='id != 3')
    assert uniqueness_sql(table, 't_log') == (
        "SELECT throwIf(count() != uniqExact(tuple(`id`)), 'duplicate gene in t') FROM `t_log` WHERE id != 3"
    )


class Declared(DatasetModel):
    dataset_name = 'declared'
    dataset_title = 'Declared'
    dataset_description = 'Declared.'
    example = {}
    products = ('platform', 'ppp')
    clickhouse = (
        ClickHouseTable('both', 'rocksdb', key='id'),
        ClickHouseTable('ppp_only', 'rocksdb', key='id', products=('ppp',)),
        ClickHouseDictionary('both_names', 'both', key='id', attributes=('id',)),
    )

    id: str


def test_declared_tables_filter_by_product():
    assert [t.name for t in declared_tables(Declared, 'platform')] == ['both']
    assert [t.name for t in declared_tables(Declared, 'ppp')] == ['both', 'ppp_only']


def test_table_sql_rerun_keeps_one_copy_of_the_rows(clickhouse_database):
    client = clickhouse_database
    table = ClickHouseTable('ddl_rerun', 'mergetree', order_by=('id',), where='id != 3', optimize=True)
    cols = [('id', CHType('Int64')), ('name', STR)]
    client.command('DROP TABLE IF EXISTS ddl_rerun_log SYNC')
    client.command('CREATE TABLE ddl_rerun_log (id Int64, name String) ENGINE = Log')
    client.command("INSERT INTO ddl_rerun_log VALUES (1, 'a'), (2, 'b'), (3, 'c')")
    try:
        for _ in range(2):
            for statement in table_sql(table, cols, 'ddl_rerun_log'):
                client.command(statement)
        assert client.command('SELECT count() FROM ddl_rerun') == 2
    finally:
        client.command('DROP TABLE IF EXISTS ddl_rerun SYNC')
        client.command('DROP TABLE IF EXISTS ddl_rerun_log SYNC')


def test_dictionary_sql_rerun_reads_the_rebuilt_table(clickhouse_database):
    client = clickhouse_database
    table = ClickHouseTable('ddl_dict_src', 'rocksdb', key='k')
    cols = [('k', STR), ('v', STR)]
    d = ClickHouseDictionary(
        'ddl_dict', 'ddl_dict_src', key='id', attributes=('name',), source_columns={'id': 'k', 'name': 'v'}
    )
    client.command('DROP TABLE IF EXISTS ddl_dict_log SYNC')
    client.command('CREATE TABLE ddl_dict_log (k String, v String) ENGINE = Log')
    client.command("INSERT INTO ddl_dict_log VALUES ('x', 'nx')")
    try:
        for _ in range(2):
            client.command(drop_dictionary_sql(d))
            for statement in table_sql(table, cols, 'ddl_dict_log'):
                client.command(statement)
            client.command(dictionary_sql(d, cols))
        assert client.command("SELECT dictGet('ddl_dict', 'name', tuple('x'))") == 'nx'
    finally:
        client.command('DROP DICTIONARY IF EXISTS ddl_dict')
        client.command('DROP TABLE IF EXISTS ddl_dict_src SYNC')
        client.command('DROP TABLE IF EXISTS ddl_dict_log SYNC')
