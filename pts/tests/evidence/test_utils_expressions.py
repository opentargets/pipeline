"""Tests for the evidence_postprocess score/direction expression registry.

The registry's parity with real spark is verified separately (see
`.superpowers/sdd/plan/registry-parity-output.txt` and `task-registry-report.md`), and the
handful of expected values hard-coded below were measured against that spark run, not guessed.
The crispr and encore scores are also checked bit for bit against spark at the end of this file.

`pts/config.yaml` no longer carries `score_expression` / `direction_on_trait_expression` /
`direction_on_target_expression` -- those were stripped once the registry below became the sole
source of truth (see `evidence_expressions.py`'s module docstring for where the original SQL a
given entry was translated from still lives in git history). So config.yaml can no longer serve as
this test's oracle for *which* fields a datasource has; the checks below are what config.yaml can
still ground: that every step's `datasource_id` maps to a registry entry and back, and that every
entry is actually usable.
"""

from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass
from math import isnan
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml
from pyspark.sql import SparkSession
from pyspark.sql import functions as f

from pts.transformers.evidence.encore import ENCORE
from pts.transformers.evidence.utils.expressions import EXPRESSIONS, log10

nan = float('nan')

CONFIG = Path(__file__).parents[2] / 'config.yaml'


def _config_datasource_ids() -> set[str]:
    """`datasource_id` for every step in config.yaml run by the `evidence_postprocess` transformer."""
    steps = yaml.safe_load(CONFIG.read_text())['steps']
    return {
        tasks[0]['settings']['datasource_id']
        for tasks in steps.values()
        if tasks and tasks[0].get('transformer') == 'evidence_postprocess'
    }


CONFIG_DATASOURCE_IDS = _config_datasource_ids()


def test_every_config_datasource_has_a_registry_entry() -> None:
    missing = CONFIG_DATASOURCE_IDS - set(EXPRESSIONS)
    assert not missing


def test_registry_has_no_unexpected_datasources() -> None:
    extra = set(EXPRESSIONS) - CONFIG_DATASOURCE_IDS
    assert not extra


@pytest.mark.parametrize('datasource_id', sorted(EXPRESSIONS))
def test_every_registry_expression_is_a_polars_expr(datasource_id: str) -> None:
    """Every entry's `score` is a usable, non-None polars expression.

    The direction fields are each either absent or a polars expression -- never a leftover string,
    a spark Column, or some other stand-in.
    """
    entry = EXPRESSIONS[datasource_id]
    assert entry.score is not None
    assert isinstance(entry.score, pl.Expr)
    for field in ('direction_on_trait', 'direction_on_target'):
        value = getattr(entry, field)
        assert value is None or isinstance(value, pl.Expr)


# ------------------------------------------------------------------------------------------------
# Hand-picked edge cases. Expected values measured against real spark evaluating the literal SQL
# from config.yaml -- see registry-parity-output.txt for the full run these are drawn from.
# ------------------------------------------------------------------------------------------------


def test_log10_rescaled_score_saturates_at_zero_and_null_propagates() -> None:
    # crispr_screen: out_min=0.0, weak_ref=0.5, strong_ref=0.005 -- measured against spark.
    df = pl.DataFrame({'resourceScore': [0.5, 0.005, 0.0, -1.0, None, float('nan')]})
    got = df.with_columns(EXPRESSIONS['crispr_screen'].score.alias('s'))['s'].to_list()
    assert got[0] == pytest.approx(0.0)
    assert got[1] == pytest.approx(1.0)
    assert got[2] == pytest.approx(1.0)  # non-positive input saturates to 1.0
    assert got[3] == pytest.approx(1.0)
    assert got[4] is None
    # NaN pins the '| value.is_nan()' guard separately from is_null(): without it a NaN resourceScore
    # scores `nan` instead of None, and 0.005 lands so close to 1.0 (0.9999999999999998) that
    # pytest.approx would let a broken NaN guard hide behind that row instead.
    assert got[5] is None


def test_europepmc_scores_one_for_null_or_nan_resource_score() -> None:
    # The score routes null/NaN resourceScore through a min_horizontal(..., 1.0) that, like spark's
    # array_min, ignores a null/NaN element rather than propagating it -- so the score saturates to
    # 1.0 instead of going null.
    df = pl.DataFrame({'resourceScore': [None, float('nan')]})
    got = df.with_columns(EXPRESSIONS['europepmc'].score.alias('s'))['s'].to_list()
    assert got == pytest.approx([1.0, 1.0])


def test_log10_is_numpy_log10_and_within_one_ulp_of_libm() -> None:
    rng = random.Random(0)
    values = [10 ** rng.uniform(-12, 3) for _ in range(5000)]
    got = pl.DataFrame({'x': values}).select(log10(pl.col('x')).alias('y'))['y'].to_list()
    assert got == np.log10(np.array(values)).tolist()
    assert all(abs(g - math.log10(v)) <= math.ulp(math.log10(v)) for g, v in zip(got, values, strict=True))


def test_log10_keeps_null_nan_zero_and_one() -> None:
    edge = pl.DataFrame({'x': [None, float('nan'), 0.0, 1.0]}).select(log10(pl.col('x')).alias('y'))['y'].to_list()
    assert edge[0] is None
    assert isnan(edge[1])
    assert edge[2] == float('-inf')
    assert edge[3] == 0.0


def test_log10_accepts_integer_and_float32_input() -> None:
    ints = pl.DataFrame({'x': [1, 10, 1000, None]}, schema={'x': pl.Int64}).select(log10(pl.col('x')).alias('y'))['y']
    assert ints.dtype == pl.Float64
    assert ints.to_list() == [0.0, 1.0, 3.0, None]
    floats = pl.DataFrame({'x': [10.0, 100.0]}, schema={'x': pl.Float32}).select(log10(pl.col('x')).alias('y'))['y']
    assert floats.dtype == pl.Float64
    assert floats.to_list() == [1.0, 2.0]


def test_log10_of_a_negative_is_nan_without_a_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        got = pl.DataFrame({'x': [-1.0, 0.0]}).select(log10(pl.col('x')).alias('y'))['y'].to_list()
    assert isnan(got[0])
    assert got[1] == float('-inf')


def test_log10_of_an_all_null_column_is_all_null() -> None:
    got = pl.DataFrame({'x': [None, None]}, schema={'x': pl.Float64}).select(log10(pl.col('x')).alias('y'))['y']
    assert got.to_list() == [None, None]


def test_linear_rescale_clamps_project_score() -> None:
    # crispr: linear_rescale(resourceScore, 41.5, 100, 0.415, 1.0) -- measured against spark.
    df = pl.DataFrame({'resourceScore': [0.0, 41.5, 100.0, 150.0]})
    got = df.with_columns(EXPRESSIONS['crispr'].score.alias('s'))['s'].to_list()
    assert got == pytest.approx([0.415, 0.415, 1.0, 1.0])


def test_clinical_precedence_score_multiplies_stage_by_stop_reason_minimum() -> None:
    df = pl.DataFrame({
        'clinicalStage': ['PHASE_2_3', 'PRECLINICAL', 'PHASE_1', 'PHASE_2_3'],
        'trialStopReasonCategories': [['Success', 'Negative'], [], ['unknown_reason'], None],
    })
    # measured against spark: 0.5*0.5, 0.01*null, 0.1*null, 0.5*1.0
    got = df.with_columns(EXPRESSIONS['clinical_precedence'].score.alias('s'))['s'].to_list()
    assert got[0] == pytest.approx(0.25)
    assert got[1] is None
    assert got[2] is None
    # a NULL trialStopReasonCategories (as opposed to an empty, non-null list) takes the
    # `IS NULL -> 1.0` branch and scores the stage score alone -- deleting that branch collapses
    # this case into the empty-list case above and it would score None instead of 0.5.
    assert got[3] == pytest.approx(0.5)


def test_direction_when_disease_present_is_null_safe() -> None:
    direction_on_trait = EXPRESSIONS['clinical_precedence'].direction_on_trait
    assert direction_on_trait is not None
    df = pl.DataFrame({'diseaseId': ['MONDO:1', None]})
    got = df.with_columns(direction_on_trait.alias('d'))['d'].to_list()
    assert got == ['protect', None]


# ------------------------------------------------------------------------------------------------
# Exhaustive one-case-per-field table -- all 25 registry fields (15 `score`, 5
# `direction_on_trait`, 5 `direction_on_target`). Each `expected` was derived by running the
# ORIGINAL SQL (git `834f1db7` `pts/config.yaml`, the last commit before the registry replaced it)
# against real spark, not by hand-computing it or reading it off this module -- that would just
# make the test a mirror of the code it exists to pin. `score` fields were run through
# `CAST(... AS DOUBLE)` and, for `crispr`'s `linear_rescale` UDF specifically, with the UDF
# registered exactly as production did (no explicit `returnType`, which pyspark 3.5.7 defaults to
# StringType rather than inferring `DoubleType` from the function's `-> float` hint) -- both match
# `calculate_evidence_score`'s `.withColumn('score', f.expr(score_expression).cast(DoubleType()))`
# (`evidence.py:240`, before the pyspark module was deleted). `direction_on_trait` /
# `direction_on_target` are asserted uncast, matching `assign_direction_on_trait` /
# `assign_direction_on_target`, which apply `f.expr(...)` directly with no cast. Every value here
# was cross-checked against the current registry too (0 mismatches) -- spark and the registry agree
# on all 25. encore, gwas_credible_sets and intogen moved to their own modules.
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _ExpressionCase:
    """One registry field, a small input frame, and its spark-derived expected output."""

    datasource_id: str
    field: str
    data: dict[str, list]
    expected: list


CASES = [
    _ExpressionCase('cancer_biomarkers', 'score', {'dummy': ['x', None]}, [1.0, 1.0]),
    _ExpressionCase('ot_crispr', 'score', {'dummy': ['x', None]}, [1.0, 1.0]),
    _ExpressionCase(
        'clinical_precedence',
        'score',
        {
            'clinicalStage': ['PHASE_2_3', 'PRECLINICAL', 'PHASE_1', 'PHASE_2_3'],
            'trialStopReasonCategories': [['Success', 'Negative'], [], ['unknown_reason'], None],
        },
        [0.25, None, None, 0.5],
    ),
    _ExpressionCase(
        'clinical_precedence', 'direction_on_trait', {'diseaseId': ['MONDO:1', None]}, ['protect', None]
    ),
    _ExpressionCase(
        'clinical_precedence',
        'direction_on_target',
        {'actionType': ['INHIBITOR', 'AGONIST', 'UNKNOWN', None]},
        ['LoF', 'GoF', None, None],
    ),
    _ExpressionCase(
        'crispr_screen',
        'score',
        {'resourceScore': [0.5, 0.005, 0.0, -1.0, None, nan]},
        [0.0, 1.0, 1.0, 1.0, None, None],
    ),
    _ExpressionCase(
        'europepmc', 'score', {'resourceScore': [50.0, 150.0, 0.0, None, nan]}, [0.5, 1.0, 0.0, 1.0, 1.0]
    ),
    _ExpressionCase(
        'crispr',
        'score',
        {'resourceScore': [0.0, 41.5, 100.0, 150.0, None, nan]},
        [0.415, 0.415, 1.0, 1.0, None, None],
    ),
]


def _assert_matches(got: list, expected: list) -> None:
    """Element-wise compare, tolerating float rounding and null."""
    assert len(got) == len(expected)
    for value, exp in zip(got, expected, strict=True):
        if exp is None:
            assert value is None
        elif isinstance(exp, float):
            assert value == pytest.approx(exp)
        else:
            assert value == exp


@pytest.mark.parametrize('case', CASES, ids=lambda c: f'{c.datasource_id}.{c.field}')
def test_registry_expression_matches_spark(case: _ExpressionCase) -> None:
    entry = EXPRESSIONS[case.datasource_id]
    expr = getattr(entry, case.field)
    assert expr is not None
    df = pl.DataFrame(case.data)
    got = df.with_columns(expr.alias('out'))['out'].to_list()
    _assert_matches(got, case.expected)


def test_case_table_covers_every_registry_field() -> None:
    """The table above is exhaustive, not just large.

    Every `(datasource_id, field)` in the registry has exactly one case, and there is nothing in
    the table that isn't in the registry.
    """
    registry_fields = {
        (datasource_id, field)
        for datasource_id, entry in EXPRESSIONS.items()
        for field in ('score', 'direction_on_trait', 'direction_on_target')
        if getattr(entry, field) is not None
    }
    table_fields = {(case.datasource_id, case.field) for case in CASES}
    assert len(CASES) == len(table_fields), 'duplicate (datasource_id, field) in CASES'
    assert table_fields == registry_fields


# ------------------------------------------------------------------------------------------------
# Exact parity with the SQL on main for the scores rewritten since the case table was measured.
# ------------------------------------------------------------------------------------------------

CRISPR_SQL = """
CASE
  WHEN resourceScore IS NULL OR isnan(resourceScore) THEN NULL
  ELSE LEAST(1.0, GREATEST(0.415,
    (1.0 - 0.415) * (CAST(resourceScore AS DOUBLE) - 41.5) / (100 - 41.5) + 0.415
  ))
END
"""

ENCORE_SQL = 'ABS ( geneticInteractionScore  ) / 16.0'


def _spark_scores(spark: SparkSession, column: str, values: list, ddl_type: str, sql: str) -> list:
    df = spark.createDataFrame([(v,) for v in values], f'{column} {ddl_type}')
    return [row.s for row in df.select(f.expr(sql).cast('double').alias('s')).collect()]


def _polars_scores(datasource_id: str, column: str, values: list, dtype: pl.DataType) -> list:
    df = pl.DataFrame({column: values}, schema={column: dtype})
    return df.select(EXPRESSIONS[datasource_id].score.alias('s'))['s'].to_list()


def _same(a: list, b: list) -> bool:
    return len(a) == len(b) and all(
        (x is None and y is None) or (x is not None and y is not None and (x == y or (isnan(x) and isnan(y))))
        for x, y in zip(a, b, strict=True)
    )


def test_crispr_score_matches_spark_bit_for_bit(spark: SparkSession) -> None:
    rng = random.Random(0)
    values: list[float | None] = [0.0, -5.0, 36.0, 41.4, 41.5, 41.6, 44.8, 52.8, 83.8, 99.99, 100.0, 100.1, 1e30]
    values += [-1e30, float('inf'), float('-inf'), float('nan'), None]
    values += [rng.uniform(30.0, 110.0) for _ in range(5000)]
    expected = _spark_scores(spark, 'resourceScore', values, 'DOUBLE', CRISPR_SQL)
    assert _same(_polars_scores('crispr', 'resourceScore', values, pl.Float64), expected)


def test_crispr_integer_score_matches_spark(spark: SparkSession) -> None:
    values = [*range(-10, 151), None]
    expected = _spark_scores(spark, 'resourceScore', values, 'BIGINT', CRISPR_SQL)
    assert _same(_polars_scores('crispr', 'resourceScore', values, pl.Int64), expected)


def test_encore_score_matches_spark_bit_for_bit(spark: SparkSession) -> None:
    rng = random.Random(0)
    values = [0.0, -0.0, 16.0, -32.5, 1e-310, float('inf'), float('nan'), None]
    values += [rng.uniform(-20.0, 20.0) for _ in range(5000)]
    expected = _spark_scores(spark, 'geneticInteractionScore', values, 'DOUBLE', ENCORE_SQL)
    got = pl.DataFrame({'geneticInteractionScore': values}, schema={'geneticInteractionScore': pl.Float64})
    assert _same(got.select(ENCORE.score.alias('s'))['s'].to_list(), expected)


@pytest.mark.parametrize('datasource_id', ['europepmc'])
def test_percent_scores_divide_exactly(datasource_id: str) -> None:
    rng = random.Random(0)
    values = [rng.uniform(0.0, 100.0) for _ in range(5000)]
    got = _polars_scores(datasource_id, 'resourceScore', values, pl.Float64)
    assert got == [min(v / 100, 1.0) if datasource_id == 'europepmc' else v / 100 for v in values]
