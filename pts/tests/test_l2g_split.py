"""Tests for the pinned train/test split derivation."""

import polars as pl

from pts.transformers.l2g.split import derive_splits, encode_labels, split_stats


def _annotated() -> pl.LazyFrame:
    return pl.LazyFrame({
        'studyLocusId': ['sl1', 'sl1', 'sl2', 'sl3'],
        'geneId': ['g1', 'g2', 'g1', 'g3'],
        'goldStandardSet': ['positive', 'negative', 'negative', 'positive'],
        'vepMaximum': [0.5, 0.25, 0.75, 0.125],
    })


def test_encode_labels_maps_strings_to_integers() -> None:
    out = encode_labels(pl.LazyFrame({'goldStandardSet': ['negative', 'positive']})).collect()
    assert out['goldStandardSet'].to_list() == [0, 1]


def test_encode_labels_passes_already_encoded_integers_through() -> None:
    out = encode_labels(pl.LazyFrame({'goldStandardSet': ['0', '1']})).collect()
    assert out['goldStandardSet'].to_list() == [0, 1]


def test_derive_splits_excludes_every_locus_touching_a_test_positive_gene() -> None:
    # g1 is positive in the test set, so sl1 AND sl2 leave training even though sl2's
    # own row for g1 is labelled negative.
    predefined = pl.LazyFrame({
        'studyLocusId': ['sl1'],
        'geneId': ['g1'],
        'goldStandardSet': ['positive'],
    })
    train, test = derive_splits(_annotated(), predefined)
    assert sorted(train['studyLocusId'].unique().to_list()) == ['sl3']
    assert test.select('studyLocusId', 'geneId').rows() == [('sl1', 'g1')]


def test_derive_splits_ignores_the_current_label_when_finding_contamination() -> None:
    # sl2/g1 flipped to negative in this run, but the pair is still in the test set and
    # must not reach training.
    predefined = pl.LazyFrame({
        'studyLocusId': ['sl2'],
        'geneId': ['g1'],
        'goldStandardSet': ['positive'],
    })
    train, test = derive_splits(_annotated(), predefined)
    assert 'sl1' not in train['studyLocusId'].to_list()
    assert test.select('studyLocusId', 'geneId').rows() == [('sl2', 'g1')]


def test_derive_splits_re_derives_test_features_from_the_current_matrix() -> None:
    predefined = pl.LazyFrame({
        'studyLocusId': ['sl2'],
        'geneId': ['g1'],
        'goldStandardSet': ['positive'],
    })
    _, test = derive_splits(_annotated(), predefined)
    assert test['vepMaximum'].to_list() == [0.75]


def test_derive_splits_accepts_integer_labels_in_the_predefined_test_set() -> None:
    predefined = pl.LazyFrame({
        'studyLocusId': ['sl1'],
        'geneId': ['g1'],
        'goldStandardSet': [1],
    })
    train, test = derive_splits(_annotated(), predefined)
    assert test.height == 1
    assert 'sl1' not in train['studyLocusId'].to_list()


def test_derive_splits_returns_integer_labels() -> None:
    predefined = pl.LazyFrame({
        'studyLocusId': ['sl1'],
        'geneId': ['g1'],
        'goldStandardSet': ['positive'],
    })
    train, test = derive_splits(_annotated(), predefined)
    assert train['goldStandardSet'].dtype == pl.Int32
    assert test['goldStandardSet'].to_list() == [1]


def test_split_stats_reports_the_gentropy_key_set() -> None:
    train = pl.DataFrame({
        'studyLocusId': ['sl3'],
        'geneId': ['g3'],
        'goldStandardSet': [1],
    })
    test = pl.DataFrame({
        'studyLocusId': ['sl1'],
        'geneId': ['g1'],
        'goldStandardSet': [0],
    })
    stats = split_stats(4, 1, train, test)
    assert stats == {
        'n_original_total': 4,
        'n_original_test': 1,
        'n_test_new': 1,
        'n_lost_test': 0,
        'n_train': 1,
        'n_lost_total': 2,
        'train': {
            'n_positive': 1,
            'n_negative': 0,
            'n_unique_loci': 1,
            'n_unique_positive_genes': 1,
        },
        'test': {
            'n_positive': 0,
            'n_negative': 1,
            'n_unique_loci': 1,
            'n_unique_positive_genes': 0,
        },
    }
