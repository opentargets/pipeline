"""Tests for the study search index."""

import polars as pl

from pts.transformers.search.helpers import LIST_STR
from pts.transformers.search.study import build_study_index, credible_set_counts

STUDY_SCHEMA = {
    'studyId': pl.String,
    'traitFromSource': pl.String,
    'pubmedId': pl.String,
    'publicationFirstAuthor': pl.String,
    'diseaseIds': LIST_STR,
    'nSamples': pl.Int32,
    'geneId': pl.String,
}


def _study(**overrides):
    row = {
        'studyId': 'S1',
        'traitFromSource': 'asthma trait',
        'pubmedId': 'PM1',
        'publicationFirstAuthor': 'Smith',
        'diseaseIds': ['D1'],
        'nSamples': 100,
        'geneId': 'T1',
    }
    row.update(overrides)
    return row


def _targets():
    return pl.LazyFrame({'targetId': ['T1'], 'approvedSymbol': ['EGFR']})


def _credible(rows):
    return pl.LazyFrame(rows, schema={'studyId': pl.String})


def test_study_index_carries_the_release_identity_columns() -> None:
    studies = pl.LazyFrame([_study()], schema=STUDY_SCHEMA)
    result = build_study_index(studies, _targets(), _credible({'studyId': ['S1']})).collect()

    assert result['id'].to_list() == ['S1']
    assert result['name'].to_list() == ['S1']
    assert result['description'].to_list() == [None]
    assert result['entity'].to_list() == ['study']
    assert result['category'][0].to_list() == ['study']


def test_study_keywords_hold_the_id_pubmed_and_author() -> None:
    studies = pl.LazyFrame([_study()], schema=STUDY_SCHEMA)
    result = build_study_index(studies, _targets(), _credible({'studyId': ['S1']})).collect()

    assert set(result['keywords'][0].to_list()) == {'S1', 'PM1', 'Smith'}
    assert result['ngrams'][0].to_list() == ['S1']


def test_study_terms_hold_the_trait_diseases_and_resolved_target() -> None:
    studies = pl.LazyFrame([_study()], schema=STUDY_SCHEMA)
    result = build_study_index(studies, _targets(), _credible({'studyId': ['S1']})).collect()

    for column in ('terms', 'terms25', 'terms5'):
        assert set(result[column][0].to_list()) == {'asthma trait', 'D1', 'EGFR', 'T1'}


def test_credible_set_counts_are_a_float_count_per_study() -> None:
    result = credible_set_counts(_credible({'studyId': ['S1', 'S1', 'S2']})).collect().sort('studyId')

    assert result['credibleSetCount'].to_list() == [2.0, 1.0]
    assert result.schema['credibleSetCount'] == pl.Float64


def test_a_single_study_gets_a_multiplier_of_one() -> None:
    """`max_rank` is 1, so the `(max_rank - 1)` denominator would divide by zero."""
    studies = pl.LazyFrame([_study()], schema=STUDY_SCHEMA)
    result = build_study_index(studies, _targets(), _credible({'studyId': ['S1']})).collect()

    assert result['multiplier'][0] == 1.0


def test_the_best_ranked_study_scores_two_and_the_worst_scores_one() -> None:
    studies = pl.LazyFrame([_study(studyId='S1'), _study(studyId='S2')], schema=STUDY_SCHEMA)
    credible = _credible({'studyId': ['S1', 'S1', 'S2']})

    result = build_study_index(studies, _targets(), credible).collect().sort('id')

    assert dict(zip(result['id'], result['multiplier'], strict=True)) == {'S1': 2.0, 'S2': 1.0}


def test_spark_a_study_with_no_credible_sets_ranks_last_not_first() -> None:
    """`credibleSetCount` is null there, and spark's DESC NULLS LAST ranks it worst. polars
    sorts nulls FIRST by default, which would hand the emptiest study the best multiplier."""
    studies = pl.LazyFrame([_study(studyId='S1'), _study(studyId='S2')], schema=STUDY_SCHEMA)
    credible = _credible({'studyId': ['S2']})

    result = build_study_index(studies, _targets(), credible).collect().sort('id')

    assert dict(zip(result['id'], result['multiplier'], strict=True)) == {'S1': 1.0, 'S2': 2.0}


def test_study_multiplier_is_never_null() -> None:
    studies = pl.LazyFrame([_study(studyId='S1'), _study(studyId='S2')], schema=STUDY_SCHEMA)

    result = build_study_index(studies, _targets(), _credible({'studyId': []})).collect()

    assert result['multiplier'].null_count() == 0


def test_study_index_emits_exactly_one_document_per_study() -> None:
    studies = pl.LazyFrame([_study(studyId='S1'), _study(studyId='S2')], schema=STUDY_SCHEMA)

    result = build_study_index(studies, _targets(), _credible({'studyId': ['S1', 'S2']})).collect()

    assert result.height == 2
    assert sorted(result['id'].to_list()) == ['S1', 'S2']
