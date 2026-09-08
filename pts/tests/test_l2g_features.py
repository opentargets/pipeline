"""Tests for the shared L2G feature contract."""

import polars as pl
import pytest

from pts.transformers.l2g.features import FEATURES, impute_and_cast


def test_features_has_31_names_in_fitted_order() -> None:
    assert len(FEATURES) == 31
    assert len(set(FEATURES)) == 31
    assert FEATURES[0] == 'eQtlColocClppMaximum'
    assert FEATURES[-1] == 'transPQtlColocH4MaximumNeighbourhood'


def test_impute_and_cast_fills_gene_counts_with_the_locus_mean() -> None:
    frame = pl.LazyFrame({
        'studyLocusId': ['a', 'a', 'a'],
        'geneId': ['g1', 'g2', 'g3'],
        'geneCount500kb': [2.0, 4.0, None],
        'proteinGeneCount500kb': [1.0, 1.0, 1.0],
    })
    out = impute_and_cast(frame, ['geneCount500kb', 'proteinGeneCount500kb']).collect()
    assert out['geneCount500kb'].to_list() == [2.0, 4.0, 3.0]


def test_impute_and_cast_does_not_borrow_the_mean_across_loci() -> None:
    frame = pl.LazyFrame({
        'studyLocusId': ['a', 'b'],
        'geneId': ['g1', 'g2'],
        'geneCount500kb': [8.0, None],
        'proteinGeneCount500kb': [1.0, 1.0],
    })
    out = impute_and_cast(frame, ['geneCount500kb', 'proteinGeneCount500kb']).collect()
    # locus 'b' has no non-null value of its own, so the mean is null and the 0.0 fill applies
    assert out['geneCount500kb'].to_list() == [8.0, 0.0]


def test_impute_and_cast_fills_every_other_feature_with_zero() -> None:
    frame = pl.LazyFrame({
        'studyLocusId': ['a'],
        'geneId': ['g1'],
        'vepMaximum': [None],
    })
    out = impute_and_cast(frame, ['vepMaximum']).collect()
    assert out['vepMaximum'].to_list() == [0.0]


def test_impute_and_cast_returns_features_in_the_requested_order_as_float32() -> None:
    frame = pl.LazyFrame({
        'studyLocusId': ['a'],
        'geneId': ['g1'],
        'vepMean': [1.0],
        'vepMaximum': [2.0],
    })
    out = impute_and_cast(frame, ['vepMaximum', 'vepMean']).collect()
    assert out.columns == ['studyLocusId', 'geneId', 'vepMaximum', 'vepMean']
    assert out.schema['vepMaximum'] == pl.Float32
    assert out.schema['vepMean'] == pl.Float32


def test_impute_and_cast_keeps_extra_columns_after_the_fixed_ones() -> None:
    frame = pl.LazyFrame({
        'studyLocusId': ['a'],
        'geneId': ['g1'],
        'goldStandardSet': ['positive'],
        'vepMaximum': [2.0],
    })
    out = impute_and_cast(frame, ['vepMaximum'], keep=['goldStandardSet']).collect()
    assert out.columns == ['studyLocusId', 'geneId', 'goldStandardSet', 'vepMaximum']


def test_impute_and_cast_raises_when_a_feature_is_absent() -> None:
    frame = pl.LazyFrame({'studyLocusId': ['a'], 'geneId': ['g1']})
    with pytest.raises(ValueError, match='vepMaximum'):
        impute_and_cast(frame, ['vepMaximum']).collect()
