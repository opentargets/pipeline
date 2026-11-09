"""Tests for the train/test split on the genes of the pinned test set."""

import polars as pl

from pts.transformers.l2g.split import count_test_genes, derive_splits, encode_labels, split_stats


def _annotated() -> pl.LazyFrame:
    # cs1: test positive g1 only           -> test
    # cs2: train positive g2, negative g1  -> neither (holds a test gene but no test positive)
    # cs3: train positive g3 only          -> train
    # cs4: test positive g1 and train positive g2 -> dropped
    return pl.LazyFrame({
        'credibleSetId': ['cs1', 'cs1', 'cs2', 'cs2', 'cs3', 'cs3', 'cs4', 'cs4'],
        'geneId': ['g1', 'g4', 'g2', 'g1', 'g3', 'g4', 'g1', 'g2'],
        'goldStandardSet': [
            'positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'positive'
        ],
        'vepMaximum': [0.5, 0.25, 0.75, 0.125, 0.5, 0.5, 0.5, 0.5],
    })


def _predefined(label: object = 'positive') -> pl.LazyFrame:
    # the pinned set is keyed by an older release; only its positive genes matter
    return pl.LazyFrame({
        'studyLocusId': ['old1', 'old2'],
        'geneId': ['g1', 'g3'],
        'goldStandardSet': [label, 'negative'],
    })


def test_encode_labels_maps_strings_to_integers() -> None:
    out = encode_labels(pl.LazyFrame({'goldStandardSet': ['negative', 'positive']})).collect()
    assert out['goldStandardSet'].to_list() == [0, 1]


def test_encode_labels_passes_already_encoded_integers_through() -> None:
    out = encode_labels(pl.LazyFrame({'goldStandardSet': ['0', '1']})).collect()
    assert out['goldStandardSet'].to_list() == [0, 1]


def test_derive_splits_matches_gentropy_split_by_test_genes() -> None:
    train, test = derive_splits(_annotated(), _predefined())
    assert sorted(train['credibleSetId'].unique().to_list()) == ['cs3']
    assert sorted(test['credibleSetId'].unique().to_list()) == ['cs1']


def test_derive_splits_keeps_every_row_of_a_selected_credible_set() -> None:
    _, test = derive_splits(_annotated(), _predefined())
    assert sorted(test.select('credibleSetId', 'geneId').rows()) == [('cs1', 'g1'), ('cs1', 'g4')]


def test_derive_splits_uses_only_the_positive_genes_of_the_pinned_set() -> None:
    # g3 is negative in the pinned set, so it is not a test gene and cs3 stays in training
    train, _ = derive_splits(_annotated(), _predefined())
    assert 'cs3' in train['credibleSetId'].to_list()


def test_derive_splits_re_derives_test_features_from_the_current_matrix() -> None:
    _, test = derive_splits(_annotated(), _predefined())
    assert test.sort('geneId')['vepMaximum'].to_list() == [0.5, 0.25]


def test_derive_splits_accepts_integer_labels_in_the_predefined_test_set() -> None:
    train, test = derive_splits(_annotated(), _predefined(label='1'))
    assert sorted(test['credibleSetId'].unique().to_list()) == ['cs1']
    assert sorted(train['credibleSetId'].unique().to_list()) == ['cs3']


def test_derive_splits_returns_integer_labels() -> None:
    train, test = derive_splits(_annotated(), _predefined())
    assert train['goldStandardSet'].dtype == pl.Int32
    assert sorted(test['goldStandardSet'].to_list()) == [0, 1]


def test_count_test_genes_counts_distinct_positive_genes() -> None:
    assert count_test_genes(_predefined()) == 1


def test_split_stats_reports_the_gentropy_v4_key_set() -> None:
    train = pl.DataFrame({'credibleSetId': ['cs3'], 'geneId': ['g3'], 'goldStandardSet': [1]})
    test = pl.DataFrame({'credibleSetId': ['cs1'], 'geneId': ['g1'], 'goldStandardSet': [0]})
    stats = split_stats(4, 1, train, test)
    assert stats == {
        'n_original_total': 4,
        'n_predefined_test_genes': 1,
        'n_train': 1,
        'n_test': 1,
        'n_lost_total': 2,
        'train': {'n_positive': 1, 'n_negative': 0, 'n_unique_loci': 1, 'n_unique_positive_genes': 1},
        'test': {'n_positive': 0, 'n_negative': 1, 'n_unique_loci': 1, 'n_unique_positive_genes': 0},
    }
