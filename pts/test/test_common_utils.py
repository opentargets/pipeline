"""Tests for pts.pyspark.common.utils camelCase conversion utilities."""

import pytest
from pyspark.sql import Row
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from pts.pyspark.common.utils import rename_columns_to_camel_case, snake_to_lower_camel

# ---------------------------------------------------------------------------
# snake_to_lower_camel
# ---------------------------------------------------------------------------


def test_snake_to_lower_camel_single_word():
    assert snake_to_lower_camel('score') == 'score'


def test_snake_to_lower_camel_two_words():
    assert snake_to_lower_camel('evidence_score') == 'evidenceScore'


def test_snake_to_lower_camel_three_words():
    assert snake_to_lower_camel('interaction_mi_identifier') == 'interactionMiIdentifier'


def test_snake_to_lower_camel_already_camel():
    assert snake_to_lower_camel('evidenceScore') == 'evidenceScore'


# ---------------------------------------------------------------------------
# rename_columns_to_camel_case — top-level columns
# ---------------------------------------------------------------------------


def test_rename_top_level_columns(spark):
    df = spark.createDataFrame([Row(evidence_score=1.0, source_database='intact')])
    result = rename_columns_to_camel_case(df)
    assert 'evidenceScore' in result.columns
    assert 'sourceDatabase' in result.columns
    assert result.collect()[0]['evidenceScore'] == 1


def test_rename_preserves_values(spark):
    df = spark.createDataFrame([Row(target_a='ENSG1', target_b='ENSG2', score=0.5)])
    result = rename_columns_to_camel_case(df)
    row = result.collect()[0]
    assert row['targetA'] == 'ENSG1'
    assert row['targetB'] == 'ENSG2'
    assert row['score'] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# rename_columns_to_camel_case — nested structs
# ---------------------------------------------------------------------------


def test_rename_nested_struct_fields(spark):
    schema = StructType([
        StructField(
            'info',
            StructType([
                StructField('database_version', StringType()),
                StructField('source_database', StringType()),
            ]),
        ),
    ])
    df = spark.createDataFrame([Row(info=Row(database_version='v1', source_database='intact'))], schema)
    result = rename_columns_to_camel_case(df)
    row = result.collect()[0]
    assert row['info']['databaseVersion'] == 'v1'
    assert row['info']['sourceDatabase'] == 'intact'


def test_rename_deeply_nested_struct(spark):
    schema = StructType([
        StructField(
            'outer_field',
            StructType([
                StructField(
                    'inner_field',
                    StructType([
                        StructField('deep_value', StringType()),
                    ]),
                ),
            ]),
        ),
    ])
    df = spark.createDataFrame([Row(outer_field=Row(inner_field=Row(deep_value='hello')))], schema)
    result = rename_columns_to_camel_case(df)
    row = result.collect()[0]
    assert row['outerField']['innerField']['deepValue'] == 'hello'


# ---------------------------------------------------------------------------
# rename_columns_to_camel_case — array<struct>
# ---------------------------------------------------------------------------


def test_rename_array_struct_fields(spark):
    schema = StructType([
        StructField(
            'methods',
            ArrayType(
                StructType([
                    StructField('mi_identifier', StringType()),
                    StructField('short_name', StringType()),
                ])
            ),
        ),
    ])
    df = spark.createDataFrame([Row(methods=[Row(mi_identifier='MI:0001', short_name='test')])], schema)
    result = rename_columns_to_camel_case(df)
    row = result.collect()[0]
    assert row['methods'][0]['miIdentifier'] == 'MI:0001'
    assert row['methods'][0]['shortName'] == 'test'


# ---------------------------------------------------------------------------
# rename_columns_to_camel_case — mixed schema
# ---------------------------------------------------------------------------


def test_rename_mixed_schema_preserves_all_data(spark):
    schema = StructType([
        StructField('target_id', StringType()),
        StructField('interaction_score', IntegerType()),
        StructField(
            'source_info',
            StructType([
                StructField('database_version', StringType()),
            ]),
        ),
        StructField(
            'detection_methods',
            ArrayType(
                StructType([
                    StructField('method_name', StringType()),
                    StructField('is_valid', BooleanType()),
                ])
            ),
        ),
    ])
    df = spark.createDataFrame(
        [
            Row(
                target_id='ENSG1',
                interaction_score=42,
                source_info=Row(database_version='v2'),
                detection_methods=[Row(method_name='pull-down', is_valid=True)],
            )
        ],
        schema,
    )
    result = rename_columns_to_camel_case(df)
    row = result.collect()[0]
    assert row['targetId'] == 'ENSG1'
    assert row['interactionScore'] == 42
    assert row['sourceInfo']['databaseVersion'] == 'v2'
    assert row['detectionMethods'][0]['methodName'] == 'pull-down'
    assert row['detectionMethods'][0]['isValid'] is True
