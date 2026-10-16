"""Read dataset schemas from parquet footers, with Spark's type names."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import fs

SPARK_METADATA_KEY = 'org.apache.spark.sql.parquet.row.metadata'
"""Footer key under which Spark stores its schema, including per-field metadata."""


@dataclass(frozen=True)
class SchemaType:
    """A column type: its Spark type name and, for nested types, its children."""

    name: str
    element: SchemaType | None = None
    fields: list[SchemaField] = field(default_factory=list)
    key: SchemaType | None = None
    value: SchemaType | None = None


@dataclass(frozen=True)
class SchemaField:
    """A named column with its type and its Spark field metadata."""

    name: str
    type: SchemaType
    metadata: dict[str, str] = field(default_factory=dict)


def type_name(arrow_type: pa.DataType) -> str:
    """The name Spark gives a parquet column of this Arrow type."""
    match arrow_type:
        case _ if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
            return 'string'
        case _ if pa.types.is_boolean(arrow_type):
            return 'boolean'
        case _ if pa.types.is_int8(arrow_type):
            return 'byte'
        case _ if pa.types.is_int16(arrow_type) or pa.types.is_uint8(arrow_type):
            return 'short'
        case _ if pa.types.is_int32(arrow_type) or pa.types.is_uint16(arrow_type):
            return 'integer'
        case _ if pa.types.is_int64(arrow_type) or pa.types.is_uint32(arrow_type):
            return 'long'
        case _ if pa.types.is_uint64(arrow_type) or pa.types.is_decimal(arrow_type):
            return 'decimal'
        case _ if pa.types.is_float32(arrow_type):
            return 'float'
        case _ if pa.types.is_float64(arrow_type):
            return 'double'
        case _ if pa.types.is_date(arrow_type):
            return 'date'
        case _ if pa.types.is_timestamp(arrow_type):
            return 'timestamp' if arrow_type.tz else 'timestamp_ntz'
        case _ if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
            return 'array'
        case _ if pa.types.is_struct(arrow_type):
            return 'struct'
        case _ if pa.types.is_map(arrow_type):
            return 'map'
        case _ if pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
            return 'binary'
        case _:
            return str(arrow_type)


def _spark_metadata(spark_fields: list[dict]) -> dict[str, dict]:
    """Spark field metadata by field name, nested fields under their parent's name."""
    out: dict[str, dict] = {}
    for spark_field in spark_fields:
        name = spark_field['name']
        out[name] = {'metadata': spark_field.get('metadata') or {}, 'children': {}}
        spark_type = spark_field['type']
        while isinstance(spark_type, dict) and spark_type.get('type') == 'array':
            spark_type = spark_type['elementType']
        if isinstance(spark_type, dict) and spark_type.get('type') == 'struct':
            out[name]['children'] = _spark_metadata(spark_type['fields'])
    return out


def _convert_type(arrow_type: pa.DataType, spark: dict[str, dict]) -> SchemaType:
    name = type_name(arrow_type)
    if name == 'array':
        return SchemaType(name, element=_convert_type(arrow_type.value_type, spark))
    if name == 'struct':
        fields = [_convert_field(arrow_type.field(i), spark) for i in range(arrow_type.num_fields)]
        return SchemaType(name, fields=fields)
    if name == 'map':
        key, value = _convert_type(arrow_type.key_type, {}), _convert_type(arrow_type.item_type, {})
        return SchemaType(name, key=key, value=value)
    return SchemaType(name)


def _convert_field(arrow_field: pa.Field, spark: dict[str, dict]) -> SchemaField:
    entry = spark.get(arrow_field.name, {})
    return SchemaField(
        name=arrow_field.name,
        type=_convert_type(arrow_field.type, entry.get('children', {})),
        metadata=dict(entry.get('metadata', {})),
    )


def _filesystem(path: str) -> tuple[fs.FileSystem, str]:
    return fs.FileSystem.from_uri(path if '://' in path else f'file://{path}')


def _parquet_files(path: str) -> list[str]:
    filesystem, root = _filesystem(path)
    infos = filesystem.get_file_info(fs.FileSelector(root, recursive=True))
    return sorted(
        info.path
        for info in infos
        if info.type == fs.FileType.File
        and info.base_name.endswith('.parquet')
        and not info.base_name.startswith(('_', '.'))
    )


def read_schema(path: str) -> list[SchemaField]:
    """The columns of the dataset at `path`, from the footer of its first parquet file.

    Raises:
        ValueError: If the dataset has no readable parquet file.
    """
    try:
        filesystem, _ = _filesystem(path)
        schema = pq.read_schema(_parquet_files(path)[0], filesystem=filesystem)
    except (IndexError, OSError, pa.ArrowException) as error:
        raise ValueError(f'Could not read dataset: {path}') from error
    spark_blob = (schema.metadata or {}).get(SPARK_METADATA_KEY.encode())
    spark = _spark_metadata(json.loads(spark_blob)['fields']) if spark_blob else {}
    return [_convert_field(arrow_field, spark) for arrow_field in schema]


def partition_columns(path: str) -> list[str]:
    """The hive partition columns of the dataset at `path`: its `<column>=<value>` folders."""
    filesystem, root = _filesystem(path)
    infos = filesystem.get_file_info(fs.FileSelector(root))
    return sorted({
        info.base_name.split('=')[0]
        for info in infos
        if info.type == fs.FileType.Directory and '=' in info.base_name
    })
