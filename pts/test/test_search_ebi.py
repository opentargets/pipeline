"""Tests for the search_ebi pyspark module.

Ported from platform-etl-backend SearchEBI step.
"""

from pyspark.sql import Row
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
)

from pts.pyspark.search_ebi import _generate_datasets

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

DISEASE_SCHEMA = StructType([
    StructField('diseaseId', StringType()),
    StructField('name', StringType()),
])

TARGET_SCHEMA = StructType([
    StructField('targetId', StringType()),
    StructField('approvedSymbol', StringType()),
])

ASSOC_SCHEMA = StructType([
    StructField('targetId', StringType()),
    StructField('diseaseId', StringType()),
    StructField('associationScore', DoubleType()),
])

EVIDENCE_SCHEMA = StructType([
    StructField('targetId', StringType()),
    StructField('diseaseId', StringType()),
    StructField('score', DoubleType()),
])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_generate_datasets_associations_output_columns(spark):
    """_generate_datasets associations output has expected columns."""
    diseases = spark.createDataFrame([Row(diseaseId='D1', name='Cancer')], DISEASE_SCHEMA)
    targets = spark.createDataFrame([Row(targetId='T1', approvedSymbol='BRCA1')], TARGET_SCHEMA)
    associations = spark.createDataFrame(
        [Row(targetId='T1', diseaseId='D1', associationScore=0.9)], ASSOC_SCHEMA
    )
    evidence = spark.createDataFrame([], EVIDENCE_SCHEMA)

    result = _generate_datasets(diseases, targets, associations, evidence)
    cols = set(result['associations'].columns)
    assert 'targetId' in cols
    assert 'diseaseId' in cols
    assert 'approvedSymbol' in cols
    assert 'name' in cols
    assert 'score' in cols


def test_generate_datasets_associations_inner_join(spark):
    """_generate_datasets associations only keeps rows matching both target and disease."""
    diseases = spark.createDataFrame([Row(diseaseId='D1', name='Cancer')], DISEASE_SCHEMA)
    targets = spark.createDataFrame([Row(targetId='T1', approvedSymbol='BRCA1')], TARGET_SCHEMA)
    associations = spark.createDataFrame([
        Row(targetId='T1', diseaseId='D1', associationScore=0.9),
        Row(targetId='T2', diseaseId='D1', associationScore=0.5),  # T2 not in targets
    ], ASSOC_SCHEMA)
    evidence = spark.createDataFrame([], EVIDENCE_SCHEMA)

    result = _generate_datasets(diseases, targets, associations, evidence)
    rows = result['associations'].collect()
    assert len(rows) == 1
    assert rows[0].targetId == 'T1'


def test_generate_datasets_associations_score_alias(spark):
    """_generate_datasets renames associationScore to score."""
    diseases = spark.createDataFrame([Row(diseaseId='D1', name='Cancer')], DISEASE_SCHEMA)
    targets = spark.createDataFrame([Row(targetId='T1', approvedSymbol='BRCA1')], TARGET_SCHEMA)
    associations = spark.createDataFrame(
        [Row(targetId='T1', diseaseId='D1', associationScore=0.8)], ASSOC_SCHEMA
    )
    evidence = spark.createDataFrame([], EVIDENCE_SCHEMA)

    result = _generate_datasets(diseases, targets, associations, evidence)
    row = result['associations'].collect()[0]
    assert abs(row.score - 0.8) < 1e-6


def test_generate_datasets_evidence_output_columns(spark):
    """_generate_datasets evidence output has expected columns."""
    diseases = spark.createDataFrame([Row(diseaseId='D1', name='Cancer')], DISEASE_SCHEMA)
    targets = spark.createDataFrame([Row(targetId='T1', approvedSymbol='BRCA1')], TARGET_SCHEMA)
    associations = spark.createDataFrame([], ASSOC_SCHEMA)
    evidence = spark.createDataFrame(
        [Row(targetId='T1', diseaseId='D1', score=0.7)], EVIDENCE_SCHEMA
    )

    result = _generate_datasets(diseases, targets, associations, evidence)
    cols = set(result['evidence'].columns)
    assert 'targetId' in cols
    assert 'diseaseId' in cols
    assert 'approvedSymbol' in cols
    assert 'name' in cols
    assert 'score' in cols


def test_generate_datasets_evidence_inner_join(spark):
    """_generate_datasets evidence only keeps rows matching both target and disease."""
    diseases = spark.createDataFrame([Row(diseaseId='D1', name='Cancer')], DISEASE_SCHEMA)
    targets = spark.createDataFrame([Row(targetId='T1', approvedSymbol='BRCA1')], TARGET_SCHEMA)
    associations = spark.createDataFrame([], ASSOC_SCHEMA)
    evidence = spark.createDataFrame([
        Row(targetId='T1', diseaseId='D1', score=0.7),
        Row(targetId='T1', diseaseId='D2', score=0.5),  # D2 not in diseases
    ], EVIDENCE_SCHEMA)

    result = _generate_datasets(diseases, targets, associations, evidence)
    rows = result['evidence'].collect()
    assert len(rows) == 1
    assert rows[0].diseaseId == 'D1'
