"""Tests for the clinical_precedence PySpark module."""

import pyspark.sql.functions as f

from pts.pyspark.clinical_precedence import _result_literature

TRIAL_LITERATURE_SCHEMA = 'id STRING, trialLiterature ARRAY<STRUCT<id: STRING, type: STRING>>'


def _literature(spark, rows) -> dict[str, list[str] | None]:
    """Run `_result_literature` over `rows` and return {id: literature}."""
    df = spark.createDataFrame(rows, TRIAL_LITERATURE_SCHEMA).withColumn(
        'literature', _result_literature(f.col('trialLiterature'))
    )
    return {r['id']: r['literature'] for r in df.select('id', 'literature').collect()}


def test_result_literature_keeps_outcome_references_and_drops_background(spark) -> None:
    """RESULT and DERIVED report the trial's outcome; BACKGROUND is what its authors cited."""
    result = _literature(spark, [
        ('t1', [('111', 'RESULT'), ('222', 'BACKGROUND'), ('333', 'DERIVED')]),
    ])
    assert sorted(result['t1']) == ['111', '333']


def test_result_literature_is_null_when_only_background(spark) -> None:
    """A trial citing background literature only yields null, not an empty array."""
    result = _literature(spark, [('t1', [('222', 'BACKGROUND')])])
    assert result['t1'] is None


def test_result_literature_is_null_when_trial_literature_is_null(spark) -> None:
    """Non-trial reports (ChEMBL, EMA, PMDA, TTD) have no trialLiterature at all."""
    result = _literature(spark, [('t1', None)])
    assert result['t1'] is None


def test_result_literature_drops_null_pmids(spark) -> None:
    """An outcome reference with no PMID must not put a null in the array."""
    result = _literature(spark, [
        ('t1', [(None, 'RESULT'), ('111', 'RESULT')]),
        ('t2', [(None, 'RESULT')]),
    ])
    assert result['t1'] == ['111']
    assert result['t2'] is None


def test_result_literature_matches_reference_type_case_insensitively(spark) -> None:
    """Reference types match regardless of case, so a case change upstream is not silent."""
    result = _literature(spark, [
        ('t1', [('111', 'result'), ('222', 'background'), ('333', 'derived')]),
    ])
    assert sorted(result['t1']) == ['111', '333']
