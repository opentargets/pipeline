"""Conversion of the polars schemas to pyspark schemas."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from pyspark.sql import types as t

if TYPE_CHECKING:
    from collections.abc import Sequence


def _to_spark_type(dtype: pl.DataType | pl.datatypes.DataTypeClass) -> t.DataType:
    """Map a polars dtype to the matching pyspark type."""
    if isinstance(dtype, pl.datatypes.DataTypeClass):
        dtype = dtype()
    match dtype:
        case pl.String() | pl.Enum() | pl.Categorical():
            return t.StringType()
        case pl.Int8():
            return t.ByteType()
        case pl.Int16():
            return t.ShortType()
        case pl.Int32():
            return t.IntegerType()
        case pl.Int64():
            return t.LongType()
        case pl.Float32():
            return t.FloatType()
        case pl.Float64():
            return t.DoubleType()
        case pl.Boolean():
            return t.BooleanType()
        case pl.List():
            return t.ArrayType(_to_spark_type(dtype.inner))
        case pl.Struct():
            return t.StructType([t.StructField(f.name, _to_spark_type(f.dtype)) for f in dtype.fields])
        case _:
            raise TypeError(f'unsupported polars dtype: {dtype}')


def to_spark_schema(schema: pl.Schema, columns: Sequence[str] | None = None) -> t.StructType:
    """Build a pyspark `StructType` from a polars schema.

    Args:
        schema: the polars schema.
        columns: columns to keep, in the order they are wanted. Defaults to every column.

    Returns:
        A `StructType` of nullable fields.

    Raises:
        KeyError: if a column is not in the schema.
        TypeError: if a column has an unsupported dtype.
    """
    names = list(schema.names()) if columns is None else list(columns)
    return t.StructType([t.StructField(name, _to_spark_type(schema[name])) for name in names])
