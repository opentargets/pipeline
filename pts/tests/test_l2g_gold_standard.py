"""Tests for gold-standard parsing and the annotated gold-standard feature matrix."""

import polars as pl
import pytest

from pts.transformers.l2g.gold_standard import annotate, parse_gold_standard

CURATED = {
    'studyLocusId': ['sl1'],
    'geneId': ['g1'],
    'diseaseIds': [['EFO_1']],
    'variantId': ['1_1_A_G'],
    'studyId': ['GCST1'],
    'goldStandardSet': ['positive'],
}


def test_parse_drops_diseaseids_and_keeps_the_curated_columns() -> None:
    out = parse_gold_standard(pl.LazyFrame(CURATED)).collect()
    assert out.columns == ['studyLocusId', 'variantId', 'studyId', 'geneId', 'goldStandardSet']


def test_parse_rejects_the_unported_otg_curation_format() -> None:
    frame = pl.LazyFrame({
        'association_info': [None],
        'gold_standard_info': [None],
        'metadata': [None],
        'sentinel_variant': [None],
        'trait_info': [None],
    })
    with pytest.raises(ValueError, match='OTG curation format'):
        parse_gold_standard(frame)


def test_parse_rejects_a_frame_missing_a_curated_column() -> None:
    frame = pl.LazyFrame({k: v for k, v in CURATED.items() if k != 'goldStandardSet'})
    with pytest.raises(ValueError, match='goldStandardSet'):
        parse_gold_standard(frame)


def _feature_matrix() -> pl.LazyFrame:
    return pl.LazyFrame({
        'studyLocusId': ['sl1', 'sl1', 'sl2'],
        'geneId': ['g1', 'g2', 'g1'],
        'isProteinCoding': [1, 1, 0],
        'vepMaximum': [0.5, 0.25, 0.75],
    })


def _credible_set() -> pl.LazyFrame:
    return pl.LazyFrame({
        'studyLocusId': ['sl1', 'sl2'],
        'variantId': ['1_1_A_G', '2_2_C_T'],
        'studyId': ['GCST1', 'GCST2'],
        'studyType': ['gwas', 'gwas'],
    })


def test_annotate_keeps_only_gold_standard_pairs() -> None:
    gold = pl.LazyFrame({
        'studyLocusId': ['sl1'],
        'variantId': ['1_1_A_G'],
        'studyId': ['GCST1'],
        'geneId': ['g1'],
        'goldStandardSet': ['positive'],
    })
    out = annotate(_feature_matrix(), _credible_set(), gold, ['vepMaximum']).collect()
    assert out.select('studyLocusId', 'geneId').rows() == [('sl1', 'g1')]
    assert out.columns == ['studyLocusId', 'geneId', 'goldStandardSet', 'vepMaximum']


def test_annotate_drops_non_protein_coding_rows() -> None:
    gold = pl.LazyFrame({
        'studyLocusId': ['sl2'],
        'variantId': ['2_2_C_T'],
        'studyId': ['GCST2'],
        'geneId': ['g1'],
        'goldStandardSet': ['positive'],
    })
    out = annotate(_feature_matrix(), _credible_set(), gold, ['vepMaximum']).collect()
    assert out.height == 0


def test_annotate_deduplicates_rows_differing_only_in_the_gold_standards_own_study_locus() -> None:
    """The gold standard's `studyLocusId` must not reach the join, or the dedup cannot see past it.

    The two curated rows below agree on everything the annotated matrix carries -- study, variant,
    gene and label -- and differ only in which credible set the curator assigned them to. That
    column is not part of the output, so keeping it on the gold-standard side of the join emits
    `studyLocusId_right`, the `unique` treats the two rows as distinct, and the fit sees the same
    training row twice. Giving both rows the SAME `studyLocusId` makes the divergence
    unobservable, which is why this fixture does not.
    """
    gold = pl.LazyFrame({
        'studyLocusId': ['sl1', 'sl9'],
        'variantId': ['1_1_A_G', '1_1_A_G'],
        'studyId': ['GCST1', 'GCST1'],
        'geneId': ['g1', 'g1'],
        'goldStandardSet': ['positive', 'positive'],
    })
    out = annotate(_feature_matrix(), _credible_set(), gold, ['vepMaximum']).collect()
    assert out.height == 1
