"""Tests for the shared L2G feature contract."""

import polars as pl
import pytest

from pts.transformers.l2g.features import FEATURES, impute_and_cast


def test_features_has_31_names_in_fitted_order() -> None:
    """The whole tuple, spelled out, because the order IS the contract.

    Asserting the count, the uniqueness and the two ends leaves the 29 names between them free to
    be transposed, and a transposition is exactly the failure that matters: the classifier would
    be handed one feature's values under another's name, silently, with no schema to catch it.
    """
    assert FEATURES == (
        'eQtlColocClppMaximum',
        'pQtlColocClppMaximum',
        'sQtlColocClppMaximum',
        'eQtlColocH4Maximum',
        'pQtlColocH4Maximum',
        'sQtlColocH4Maximum',
        'eQtlColocClppMaximumNeighbourhood',
        'pQtlColocClppMaximumNeighbourhood',
        'sQtlColocClppMaximumNeighbourhood',
        'eQtlColocH4MaximumNeighbourhood',
        'pQtlColocH4MaximumNeighbourhood',
        'sQtlColocH4MaximumNeighbourhood',
        'distanceSentinelFootprint',
        'distanceSentinelFootprintNeighbourhood',
        'distanceFootprintMean',
        'distanceFootprintMeanNeighbourhood',
        'distanceTssMean',
        'distanceTssMeanNeighbourhood',
        'distanceSentinelTss',
        'distanceSentinelTssNeighbourhood',
        'vepMaximum',
        'vepMaximumNeighbourhood',
        'vepMean',
        'vepMeanNeighbourhood',
        'e2gMean',
        'e2gMeanNeighbourhood',
        'geneCount500kb',
        'proteinGeneCount500kb',
        'credibleSetConfidence',
        'transPQtlColocH4Maximum',
        'transPQtlColocH4MaximumNeighbourhood',
    )
    assert len(FEATURES) == 31
    assert len(set(FEATURES)) == 31


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
