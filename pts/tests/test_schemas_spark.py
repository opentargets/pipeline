"""Tests for the polars to spark schema conversion."""

import polars as pl
import pytest
from pyspark.sql.types import (
    ArrayType,
    ByteType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
    StructField,
    StructType,
)

from pts.schemas.credible_set import (
    credible_set_schema,
    credible_set_variant_schema,
    validated_credible_set_schema,
)
from pts.schemas.spark import to_spark_schema


def test_maps_credible_set_variant_columns():
    s = to_spark_schema(
        credible_set_variant_schema,
        ['credibleSetId', 'posteriorProbability', 'reportedBeta', 'qualityControls'],
    )
    assert s == StructType([
        StructField('credibleSetId', StringType()),
        StructField('posteriorProbability', FloatType()),
        StructField('reportedBeta', DoubleType()),
        StructField('qualityControls', ArrayType(StringType())),
    ])


def test_follows_requested_column_order():
    s = to_spark_schema(credible_set_variant_schema, ['reportedBeta', 'credibleSetId'])
    assert s.fieldNames() == ['reportedBeta', 'credibleSetId']


def test_whole_schema_keeps_schema_order():
    assert to_spark_schema(credible_set_schema).fieldNames() == list(credible_set_schema.names())


def test_integer_and_boolean_types():
    s = to_spark_schema(
        pl.Schema({'a': pl.Int8(), 'b': pl.Int16(), 'c': pl.Int32(), 'd': pl.Int64(), 'e': pl.Boolean()})
    )
    assert [f.dataType.simpleString() for f in s] == ['tinyint', 'smallint', 'int', 'bigint', 'boolean']
    assert [type(f.dataType) for f in s][:2] == [ByteType, ShortType]
    assert isinstance(s['c'].dataType, IntegerType)
    assert isinstance(s['d'].dataType, LongType)
    assert all(f.nullable for f in s)


def test_struct_is_recursive():
    s = to_spark_schema(pl.Schema({'x': pl.Struct({'a': pl.Int64(), 'b': pl.List(pl.Float32())})}))
    assert s['x'].dataType == StructType([
        StructField('a', LongType()),
        StructField('b', ArrayType(FloatType())),
    ])


def test_enum_maps_to_string():
    assert to_spark_schema(validated_credible_set_schema, ['studyType'])['studyType'].dataType == StringType()


def test_categorical_maps_to_string():
    assert to_spark_schema(pl.Schema({'c': pl.Categorical()}))['c'].dataType == StringType()


def test_unknown_column_raises():
    with pytest.raises(KeyError):
        to_spark_schema(credible_set_schema, ['nope'])


def test_unsupported_dtype_raises():
    with pytest.raises(TypeError):
        to_spark_schema(pl.Schema({'d': pl.Date()}))


def test_spark_reads_polars_parquet_with_derived_schema(spark, tmp_path):
    cs_path = tmp_path / 'cs.parquet'
    var_path = tmp_path / 'var.parquet'
    cs = pl.DataFrame(
        {
            'id': ['cs1'],
            'chromosome': ['1'],
            'studyType': ['gwas'],
            'fineMappingMethod': ['SuSiE'],
            'qualityControls': [['A', 'B']],
        },
        schema={
            c: validated_credible_set_schema[c]
            for c in ['id', 'chromosome', 'studyType', 'fineMappingMethod', 'qualityControls']
        },
    )
    var = pl.DataFrame(
        {'credibleSetId': ['cs1'], 'posteriorProbability': [0.25], 'reportedBeta': [-1.5], 'rescaledDirection': [-1]},
        schema={
            c: credible_set_variant_schema[c]
            for c in ['credibleSetId', 'posteriorProbability', 'reportedBeta', 'rescaledDirection']
        },
    )
    cs.write_parquet(cs_path)
    var.write_parquet(var_path)

    cs_cols = ['id', 'chromosome', 'studyType', 'fineMappingMethod', 'qualityControls']
    var_cols = ['credibleSetId', 'posteriorProbability', 'reportedBeta', 'rescaledDirection']
    cs_rows = spark.read.schema(to_spark_schema(validated_credible_set_schema, cs_cols)).parquet(str(cs_path)).collect()
    var_rows = (
        spark.read.schema(to_spark_schema(credible_set_variant_schema, var_cols)).parquet(str(var_path)).collect()
    )

    assert len(cs_rows) == 1
    assert cs_rows[0].asDict() == {
        'id': 'cs1',
        'chromosome': '1',
        'studyType': 'gwas',
        'fineMappingMethod': 'SuSiE',
        'qualityControls': ['A', 'B'],
    }
    assert len(var_rows) == 1
    assert var_rows[0].asDict() == {
        'credibleSetId': 'cs1',
        'posteriorProbability': 0.25,
        'reportedBeta': -1.5,
        'rescaledDirection': -1,
    }
