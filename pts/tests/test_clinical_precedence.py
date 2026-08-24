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
    """66.5k trials in the 2026-06 AACT dump cite background literature only.

    They must yield null rather than an empty array, so the field stays absent for them
    exactly as it does for reports that carry no trial literature at all.
    """
    result = _literature(spark, [('t1', [('222', 'BACKGROUND')])])
    assert result['t1'] is None


def test_result_literature_is_null_when_trial_literature_is_null(spark) -> None:
    """Non-trial reports (ChEMBL, EMA, PMDA, TTD) have no trialLiterature at all."""
    result = _literature(spark, [('t1', None)])
    assert result['t1'] is None


def test_result_literature_drops_null_pmids(spark) -> None:
    """~3.8k RESULT rows in AACT carry a null pmid; a null must not reach the array."""
    result = _literature(spark, [
        ('t1', [(None, 'RESULT'), ('111', 'RESULT')]),
        ('t2', [(None, 'RESULT')]),
    ])
    assert result['t1'] == ['111']
    assert result['t2'] is None


def test_result_literature_matches_reference_type_case_insensitively(spark) -> None:
    """AACT spells the type uppercase, but a case change upstream must not empty the column.

    `clinical_mining`'s own test fixtures use the lowercase spelling, so the two cases
    are demonstrably both in circulation.
    """
    result = _literature(spark, [
        ('t1', [('111', 'result'), ('222', 'background'), ('333', 'derived')]),
    ])
    assert sorted(result['t1']) == ['111', '333']
