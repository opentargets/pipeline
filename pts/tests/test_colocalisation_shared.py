"""The shared-tag colocalisation equals the aligned-tag colocalisation."""

from __future__ import annotations

import pytest

from pts.pyspark.colocalisation_utils.coloc_pip_ecaviar import colocalise, colocalise_tags
from pts.pyspark.colocalisation_utils.overlap import aligned_tags, overlapping_pairs, tags
from tests.colocalisation_fixtures import credible_set_variants, credible_sets

_KEY = ('leftLocusId', 'rightLocusId')


def _both(spark, sets, variants):
    all_tags = tags(credible_sets(spark, sets), credible_set_variants(spark, variants))
    aligned = colocalise(aligned_tags(all_tags, overlapping_pairs(all_tags)))
    shared = colocalise_tags(all_tags)
    return (
        {tuple(r[k] for k in _KEY): r.asDict() for r in aligned.collect()},
        {tuple(r[k] for k in _KEY): r.asDict() for r in shared.collect()},
        aligned.schema,
        shared.schema,
    )


def _assert_equal(expected, actual):
    assert expected.keys() == actual.keys()
    for key, row in expected.items():
        other = actual[key]
        for column, value in row.items():
            if isinstance(value, float):
                assert other[column] == pytest.approx(value, rel=1e-12, abs=1e-300), (key, column)
            else:
                assert other[column] == value, (key, column)


SETS = [
    ('g1', 'gwas', '1'),
    ('g2', 'gwas', '1'),
    ('g3', 'gwas', '2'),
    ('q1', 'eqtl', '1'),
    ('q2', 'pqtl', '1'),
    ('q3', 'eqtl', '2'),
]
VARIANTS = [
    # g1: shared with g2, q1 and q2, plus one-sided tags
    ('g1', 'v1', 0.6, 0.2, None),
    ('g1', 'v2', 0.3, None, -0.4),
    ('g1', 'v3', 0.1, 0.0, 0.1),
    ('g1', 'v9', 1e-15, 0.3, None),
    # g2: GWAS-GWAS with g1 (g2 > g1, so g2 is on the left)
    ('g2', 'v1', 0.5, -0.2, None),
    ('g2', 'v4', 0.5, 0.1, None),
    # q1: shares v1, v2; one tag with null PP, one with only right-side tags
    ('q1', 'v1', 0.4, 0.5, None),
    ('q1', 'v2', None, -0.1, None),
    ('q1', 'v5', 0.6, None, None),
    # q2: shares only the tiny-PP tag with g1
    ('q2', 'v9', 1e-15, 0.2, None),
    ('q2', 'v6', 0.99, 0.1, None),
    # chromosome 2: GWAS vs QTL, many one-sided tags
    ('g3', 'w1', 0.25, 0.1, None),
    ('g3', 'w2', 0.25, 0.2, None),
    ('g3', 'w3', 0.25, None, None),
    ('g3', 'w4', 0.25, -0.3, None),
    ('q3', 'w4', 0.9, 0.4, None),
    ('q3', 'w5', 0.1, 0.4, None),
    # QTL-QTL overlap is not a pair
    ('q2', 'v5', 0.01, 0.1, None),
]


def test_shared_equals_aligned(spark):
    aligned, shared, aligned_schema, shared_schema = _both(spark, SETS, VARIANTS)
    assert set(aligned) == {('g2', 'g1'), ('g1', 'q1'), ('g2', 'q1'), ('g1', 'q2'), ('g3', 'q3')}
    assert shared_schema == aligned_schema
    _assert_equal(aligned, shared)


def test_shared_floor_is_visible(spark):
    aligned, shared, _, _ = _both(spark, SETS, VARIANTS)
    row = shared[('g1', 'q2')]
    assert row['numberColocalisingVariants'] == 1
    assert row['h3'] == pytest.approx(aligned[('g1', 'q2')]['h3'], rel=1e-12)
    assert row['h3'] > 0.5
