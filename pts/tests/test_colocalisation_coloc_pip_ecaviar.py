"""Tests for the coloc_pip_ecaviar colocalisation aggregation."""

from __future__ import annotations

import pytest
from pyspark.sql import types as t

from pts.pyspark.colocalisation_utils.coloc_pip_ecaviar import colocalise
from pts.pyspark.colocalisation_utils.overlap import aligned_tags, overlapping_pairs, tags
from pts.schemas.colocalisation import colocalisation_schema
from pts.schemas.spark import to_spark_schema
from tests.colocalisation_fixtures import credible_set_variants, credible_sets

_ALIGNED = t.StructType([
    t.StructField('leftLocusId', t.StringType()),
    t.StructField('rightLocusId', t.StringType()),
    t.StructField('rightStudyType', t.StringType()),
    t.StructField('chromosome', t.StringType()),
    t.StructField('tagVariantId', t.StringType()),
    t.StructField('left_posteriorProbability', t.DoubleType()),
    t.StructField('right_posteriorProbability', t.DoubleType()),
    t.StructField('left_beta', t.DoubleType()),
    t.StructField('right_beta', t.DoubleType()),
])


def _aligned(spark, rows):
    return spark.createDataFrame([('L', 'R', 'eqtl', '1', *r) for r in rows], _ALIGNED)


def _one_sided(spark):
    return _aligned(spark, [('v1', 0.5, 1.0, 0.2, -0.3), ('v2', 0.5, None, 0.1, None)])


def test_worked_example(spark):
    row = colocalise(_one_sided(spark)).collect()[0]
    assert row.numberColocalisingVariants == 1
    assert row.clpp == pytest.approx(0.5, rel=1e-12)
    assert row.h4 == pytest.approx(5e-6 / (5e-6 + 5e-9), rel=1e-9)
    assert row.h3 == pytest.approx(5e-9 / (5e-6 + 5e-9), rel=1e-9)
    assert row.betaRatioSignAverage == -1.0
    assert row.colocalisationMethod == 'COLOC_PIP_ECAVIAR'


def test_one_sided_tag_is_floored_and_not_counted(spark):
    two_sided = _aligned(spark, [('v1', 0.5, 1.0, 0.2, -0.3)])
    base = colocalise(two_sided).collect()[0]
    row = colocalise(_one_sided(spark)).collect()[0]
    assert row.numberColocalisingVariants == 1
    assert row.clpp == base.clpp
    s1, s2, b = 1.0, 1.0 + 1e-16, 0.5 + 0.5 * 1e-16
    h3u, h4u = 1e-8 * max(s1 * s2 - b, 0.0), 1e-5 * b
    assert row.h4 == pytest.approx(h4u / (h3u + h4u), rel=1e-12)
    assert base.h4 != row.h4


def test_beta_ratio_is_null_without_usable_betas(spark):
    df = _aligned(spark, [('v1', 0.5, 0.5, 0.2, 0.0), ('v2', 0.5, 0.5, 0.1, None), ('v3', 0.0, 0.0, 0.0, 1.0)])
    assert colocalise(df).collect()[0].betaRatioSignAverage is None


def test_beta_ratio_averages_usable_rows_only(spark):
    df = _aligned(spark, [('v1', 0.5, 0.5, 0.2, 0.1), ('v2', 0.5, 0.5, 0.2, -0.1), ('v3', 0.5, 0.5, 0.2, 0.0)])
    assert colocalise(df).collect()[0].betaRatioSignAverage == 0.0


def test_h3_is_zero_when_all_mass_is_shared(spark):
    row = colocalise(_aligned(spark, [('v1', 1.0, 1.0, 0.2, 0.3)])).collect()[0]
    assert row.h3 == 0.0
    assert row.h4 == 1.0


def test_output_schema(spark):
    schema = colocalise(_one_sided(spark)).schema
    expected = to_spark_schema(colocalisation_schema)
    assert [(f.name, f.dataType) for f in schema] == [(f.name, f.dataType) for f in expected]


def test_left_locus_in_two_pairs_counts_independently(spark):
    sets = credible_sets(spark, [('g', 'gwas', '1'), ('q1', 'eqtl', '1'), ('q2', 'eqtl', '1')])
    variants = credible_set_variants(
        spark,
        [
            ('g', 'v1', 0.5, None, 0.2),
            ('g', 'v2', 0.5, None, 0.1),
            ('q1', 'v1', 1.0, None, 0.3),
            ('q2', 'v1', 0.5, None, 0.3),
            ('q2', 'v2', 0.5, None, 0.3),
        ],
    )
    all_tags = tags(sets, variants)
    out = colocalise(aligned_tags(all_tags, overlapping_pairs(all_tags)))
    rows = {r.rightLocusId: r for r in out.collect()}
    assert set(rows) == {'q1', 'q2'}
    assert rows['q1'].numberColocalisingVariants == 1
    assert rows['q2'].numberColocalisingVariants == 2
    assert rows['q1'].clpp == pytest.approx(0.5)
    assert rows['q2'].clpp == pytest.approx(0.5)
