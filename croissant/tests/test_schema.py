"""Tests for reading dataset schemas from parquet footers."""

import json
from pathlib import Path

import mlcroissant as mlc
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ot_croissant.crumbs.distribution import PlatformOutputDistribution
from ot_croissant.crumbs.record_sets import PlatformOutputRecordSets
from ot_croissant.schema import SPARK_METADATA_KEY, SchemaField, SchemaType, partition_columns, read_schema, type_name


@pytest.mark.parametrize(
    ('arrow_type', 'expected'),
    [
        (pa.string(), 'string'),
        (pa.large_string(), 'string'),
        (pa.bool_(), 'boolean'),
        (pa.int8(), 'byte'),
        (pa.int16(), 'short'),
        (pa.int32(), 'integer'),
        (pa.int64(), 'long'),
        (pa.uint8(), 'short'),
        (pa.uint16(), 'integer'),
        (pa.uint32(), 'long'),
        (pa.uint64(), 'decimal'),
        (pa.float32(), 'float'),
        (pa.float64(), 'double'),
        (pa.decimal128(10, 2), 'decimal'),
        (pa.date32(), 'date'),
        (pa.timestamp('us', tz='UTC'), 'timestamp'),
        (pa.timestamp('us'), 'timestamp_ntz'),
        (pa.binary(), 'binary'),
        (pa.list_(pa.int64()), 'array'),
        (pa.large_list(pa.string()), 'array'),
        (pa.struct([('a', pa.int64())]), 'struct'),
        (pa.map_(pa.string(), pa.int64()), 'map'),
    ],
)
def test_type_name_follows_spark(arrow_type: pa.DataType, expected: str) -> None:
    assert type_name(arrow_type) == expected


def _write(path: Path, table: pa.Table) -> str:
    path.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path / 'part-00000.parquet')
    return str(path)


NESTED = pa.schema([
    ('id', pa.large_string()),
    ('score', pa.float64()),
    ('tags', pa.large_list(pa.large_string())),
    ('loc', pa.struct([('chromosome', pa.string()), ('start', pa.int64())])),
    ('xrefs', pa.list_(pa.struct([('id', pa.string()), ('source', pa.string())]))),
    ('grid', pa.list_(pa.list_(pa.int32()))),
    ('counts', pa.map_(pa.string(), pa.int64())),
])


def test_read_schema_walks_nested_types(tmp_path: Path) -> None:
    path = _write(tmp_path / 'ds', NESTED.empty_table())
    fields = read_schema(path)
    assert [f.name for f in fields] == NESTED.names
    types = {f.name: f.type for f in fields}
    text = SchemaType('string')
    assert types['tags'] == SchemaType('array', element=text)
    assert types['loc'] == SchemaType(
        'struct', fields=[SchemaField('chromosome', text), SchemaField('start', SchemaType('long'))]
    )
    assert types['xrefs'] == SchemaType(
        'array', element=SchemaType('struct', fields=[SchemaField('id', text), SchemaField('source', text)])
    )
    assert types['grid'] == SchemaType('array', element=SchemaType('array', element=SchemaType('integer')))
    assert types['counts'] == SchemaType('map', key=text, value=SchemaType('long'))


def test_read_schema_takes_spark_field_metadata(tmp_path: Path) -> None:
    spark_schema = {
        'type': 'struct',
        'fields': [
            {'name': 'id', 'type': 'string', 'nullable': True, 'metadata': {'description': 'The id'}},
            {
                'name': 'loc',
                'type': {
                    'type': 'struct',
                    'fields': [
                        {'name': 'chromosome', 'type': 'string', 'nullable': True, 'metadata': {'foreign_key': 'x/y'}}
                    ],
                },
                'nullable': True,
                'metadata': {},
            },
        ],
    }
    schema = pa.schema([('id', pa.string()), ('loc', pa.struct([('chromosome', pa.string())]))]).with_metadata({
        SPARK_METADATA_KEY: json.dumps(spark_schema)
    })
    fields = read_schema(_write(tmp_path / 'ds', schema.empty_table()))
    assert fields[0].metadata == {'description': 'The id'}
    assert fields[1].metadata == {}
    assert fields[1].type.fields[0].metadata == {'foreign_key': 'x/y'}


def test_read_schema_without_spark_metadata_has_empty_metadata(tmp_path: Path) -> None:
    fields = read_schema(_write(tmp_path / 'ds', pa.schema([('id', pa.string())]).empty_table()))
    assert fields == [SchemaField(name='id', type=fields[0].type, metadata={})]


def test_read_schema_reads_parts_below_partition_folders(tmp_path: Path) -> None:
    _write(tmp_path / 'ds' / 'chromosome=1', pa.schema([('id', pa.string())]).empty_table())
    (tmp_path / 'ds' / '_SUCCESS').write_text('')
    assert [f.name for f in read_schema(str(tmp_path / 'ds'))] == ['id']


def test_read_schema_without_parquet_raises(tmp_path: Path) -> None:
    (tmp_path / 'ds').mkdir()
    with pytest.raises(ValueError, match='Could not read dataset'):
        read_schema(str(tmp_path / 'ds'))


def test_partition_columns_are_the_hive_folders(tmp_path: Path) -> None:
    for d in ('chromosome=1', 'chromosome=2', 'source=x', 'not_a_partition'):
        (tmp_path / 'ds' / d).mkdir(parents=True)
    (tmp_path / 'ds' / 'part=0.parquet').write_text('')
    assert partition_columns(str(tmp_path / 'ds')) == ['chromosome', 'source']


def test_partition_columns_of_a_flat_dataset_are_empty(tmp_path: Path) -> None:
    path = _write(tmp_path / 'ds', pa.schema([('id', pa.string())]).empty_table())
    assert partition_columns(path) == []


def test_record_set_fields_from_a_nested_dataset(tmp_path: Path) -> None:
    path = _write(tmp_path / 'ds', NESTED.empty_table())
    record_set = PlatformOutputRecordSets().add_assets_from_paths([path]).get_metadata()[0]
    fields = {f.name: f for f in record_set.fields}
    assert fields['id'].data_types == [mlc.DataType.TEXT]
    assert fields['score'].data_types == [mlc.DataType.FLOAT64]
    assert fields['tags'].repeated is True
    assert fields['tags'].data_types == [mlc.DataType.TEXT]
    assert [s.name for s in fields['loc'].sub_fields] == ['chromosome', 'start']
    assert fields['loc'].sub_fields[1].data_types == [mlc.DataType.INT64]
    assert fields['loc'].sub_fields[0].id == 'ds/loc/chromosome'
    assert [s.name for s in fields['xrefs'].sub_fields] == ['id', 'source']
    assert fields['xrefs'].repeated is True
    assert fields['grid'].data_types == [mlc.DataType.INT32]
    assert fields['counts'].repeated is True
    assert [s.name for s in fields['counts'].sub_fields] == ['key', 'value']
    assert fields['counts'].sub_fields[1].data_types == [mlc.DataType.INT64]


def test_distribution_fileset_for_a_partitioned_dataset(tmp_path: Path) -> None:
    _write(tmp_path / 'ds' / 'chromosome=1', pa.schema([('id', pa.string())]).empty_table())
    fileset = PlatformOutputDistribution().add_assets_from_paths([str(tmp_path / 'ds')]).get_metadata()[0]
    assert fileset.includes == ['ds/**/*.parquet']
    assert fileset.description.endswith('partitioned by chromosome')
