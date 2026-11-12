"""Tests for the colocalisation tags and overlapping credible set pairs."""

from __future__ import annotations

import pytest
from pyspark.sql import types as t

from pts.pyspark.colocalisation_utils.overlap import aligned_tags, overlapping_pairs, tags
from tests.colocalisation_fixtures import credible_set_variants, credible_sets


def _pairs(df):
    return {tuple(r) for r in df.select('leftLocusId', 'rightLocusId', 'rightStudyType', 'chromosome').collect()}


def test_tags_prefer_conditional_beta(spark):
    sets = credible_sets(spark, [('a', 'gwas', '1'), ('b', 'gwas', '1')])
    variants = credible_set_variants(spark, [('a', 'v', 0.5, 0.25, 0.5), ('b', 'v', 0.5, None, 0.5)])
    betas = {r.locusId: r.beta for r in tags(sets, variants).collect()}
    assert betas == {'a': 0.25, 'b': 0.5}


def test_tags_cast_float32_inputs_to_double(spark):
    sets = credible_sets(spark, [('a', 'gwas', '1')])
    variants = credible_set_variants(spark, [('a', 'v', 0.5, 0.25, 0.5)])
    df = tags(sets, variants)
    assert df.columns == ['locusId', 'studyType', 'chromosome', 'tagVariantId', 'posteriorProbability', 'beta']
    assert df.schema['posteriorProbability'].dataType == t.DoubleType()
    assert df.schema['beta'].dataType == t.DoubleType()


def test_overlap_pairs_rules(spark):
    sets = credible_sets(
        spark,
        [('g1', 'gwas', '1'), ('g2', 'gwas', '1'), ('q1', 'eqtl', '1'), ('q2', 'eqtl', '1')],
    )
    variants = credible_set_variants(spark, [(i, '1_10_A_T', 0.5, None, 0.1) for i in ('g1', 'g2', 'q1', 'q2')])
    pairs = overlapping_pairs(tags(sets, variants))
    assert _pairs(pairs) == {
        ('g2', 'g1', 'gwas', '1'),
        ('g1', 'q1', 'eqtl', '1'),
        ('g1', 'q2', 'eqtl', '1'),
        ('g2', 'q1', 'eqtl', '1'),
        ('g2', 'q2', 'eqtl', '1'),
    }
    assert pairs.count() == 5


def test_pairs_do_not_cross_chromosomes(spark):
    sets = credible_sets(spark, [('g1', 'gwas', '1'), ('q1', 'eqtl', '2')])
    variants = credible_set_variants(spark, [('g1', 'v', 0.5, None, 0.1), ('q1', 'v', 0.5, None, 0.1)])
    assert overlapping_pairs(tags(sets, variants)).count() == 0


def test_aligned_tags_keep_one_sided_tags(spark):
    sets = credible_sets(spark, [('g1', 'gwas', '1'), ('q1', 'eqtl', '1')])
    variants = credible_set_variants(
        spark,
        [('g1', 'v1', 0.6, None, 0.1), ('g1', 'v2', 0.4, None, -0.2), ('q1', 'v1', 0.9, 0.3, 0.1)],
    )
    t_ = tags(sets, variants)
    df = aligned_tags(t_, overlapping_pairs(t_))
    assert df.columns == [
        'leftLocusId',
        'rightLocusId',
        'rightStudyType',
        'chromosome',
        'tagVariantId',
        'left_posteriorProbability',
        'right_posteriorProbability',
        'left_beta',
        'right_beta',
    ]
    rows = {r.tagVariantId: r for r in df.collect()}
    assert set(rows) == {'v1', 'v2'}
    assert rows['v1'].left_posteriorProbability == pytest.approx(0.6)
    assert rows['v1'].right_posteriorProbability == pytest.approx(0.9)
    assert rows['v1'].right_beta == pytest.approx(0.3)
    assert rows['v2'].right_posteriorProbability is None
    assert rows['v2'].right_beta is None
    assert rows['v2'].left_beta == pytest.approx(-0.2)


def test_aligned_tags_keep_tags_only_in_the_right_locus(spark):
    sets = credible_sets(spark, [('g1', 'gwas', '1'), ('q1', 'eqtl', '1')])
    variants = credible_set_variants(
        spark,
        [('g1', 'v1', 0.6, None, 0.1), ('q1', 'v1', 0.9, None, 0.1), ('q1', 'v3', 0.1, None, 0.1)],
    )
    t_ = tags(sets, variants)
    rows = {r.tagVariantId: r for r in aligned_tags(t_, overlapping_pairs(t_)).collect()}
    assert rows['v3'].left_posteriorProbability is None
    assert rows['v3'].right_posteriorProbability == pytest.approx(0.1)
