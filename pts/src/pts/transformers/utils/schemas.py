"""Schema conversion utilities for transformers."""

from __future__ import annotations

import importlib.resources as pkg_resources
import json
from functools import cache
from typing import Any

import polars as pl

from pts import schemas

_SPARK_TO_POLARS_TYPE: dict[str, Any] = {
    'boolean': pl.Boolean,
    'byte': pl.Int8,
    'short': pl.Int16,
    'integer': pl.Int32,
    'long': pl.Int64,
    'float': pl.Float32,
    'double': pl.Float64,
    'string': pl.String,
    'date': pl.Date,
    'timestamp': pl.Datetime,
}


def spark_type_to_polars(type_spec: str | dict[str, Any]) -> Any:
    """Convert a Spark SQL type specification into a Polars dtype."""
    if isinstance(type_spec, str):
        mapped = _SPARK_TO_POLARS_TYPE.get(type_spec)
        if mapped is not None:
            return mapped
        if type_spec.startswith('decimal'):
            return pl.Decimal
        msg = f'Unsupported Spark type: {type_spec}'
        raise ValueError(msg)

    type_name = type_spec.get('type')
    if type_name == 'array':
        return pl.List(spark_type_to_polars(type_spec['elementType']))

    if type_name == 'struct':
        return pl.Struct([
            pl.Field(field['name'], spark_type_to_polars(field['type']))
            for field in type_spec['fields']
        ])

    msg = f'Unsupported Spark complex type: {type_name}'
    raise ValueError(msg)


@cache
def load_spark_schema_as_polars(schema_json: str) -> dict[str, Any]:
    """Load a Spark JSON schema from `pts.schemas` and map it to Polars dtypes."""
    schema_path = pkg_resources.files(schemas).joinpath(schema_json)
    with schema_path.open(encoding='utf-8') as f:
        spark_schema = json.load(f)

    return {
        field['name']: spark_type_to_polars(field['type'])
        for field in spark_schema['fields']
    }
