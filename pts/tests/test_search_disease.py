"""Tests for the disease search index."""

import polars as pl

from pts.transformers.search.disease import build_disease_index
from pts.transformers.search.helpers import LIST_STR

SYNONYM_STRUCT = pl.Struct(
    {
        'hasExactSynonym': LIST_STR,
        'hasRelatedSynonym': LIST_STR,
        'hasNarrowSynonym': LIST_STR,
        'hasBroadSynonym': LIST_STR,
    }
)


def _inputs(**overrides):
    base = {
        'diseases': pl.LazyFrame(
            [
                {
                    'diseaseId': 'D1',
                    'name': 'asthma',
                    'description': 'a disease',
                    'synonyms': {
                        'hasExactSynonym': ['exact'],
                        'hasRelatedSynonym': None,
                        'hasNarrowSynonym': None,
                        'hasBroadSynonym': None,
                    },
                    'therapeutic_labels': ['respiratory'],
                }
            ],
            schema={
                'diseaseId': pl.String,
                'name': pl.String,
                'description': pl.String,
                'synonyms': SYNONYM_STRUCT,
                'therapeutic_labels': LIST_STR,
            },
        ),
        'phenotypes': pl.LazyFrame(
            {'diseaseId': ['D1'], 'phenotype_labels': [['wheeze']]},
            schema={'diseaseId': pl.String, 'phenotype_labels': LIST_STR},
        ),
        'associations': pl.LazyFrame(
            {'associationId': ['D1-T1'], 'targetId': ['T1'], 'diseaseId': ['D1'], 'score': [0.8]}
        ),
        'scored_drugs': pl.LazyFrame(
            {'associationId': ['D1-T1'], 'drugId': ['CH1'], 'targetId': ['T1'], 'diseaseId': ['D1'], 'score': [0.8]}
        ),
        't_lut': pl.LazyFrame(
            {'targetId': ['T1'], 'target_labels': [['EGFR']]},
            schema={'targetId': pl.String, 'target_labels': LIST_STR},
        ),
        'dr_lut': pl.LazyFrame(
            {'drugId': ['CH1'], 'drug_labels': [['aspirin']]},
            schema={'drugId': pl.String, 'drug_labels': LIST_STR},
        ),
        'studies': pl.LazyFrame(
            {'studyId': ['S1'], 'diseaseIds': [['D1']]},
            schema={'studyId': pl.String, 'diseaseIds': LIST_STR},
        ),
        'nct_by_disease': pl.LazyFrame(
            {'diseaseId': ['D1'], 'nctIds': [['nct01']]},
            schema={'diseaseId': pl.String, 'nctIds': LIST_STR},
        ),
    }
    base.update(overrides)
    return base


def test_disease_index_carries_the_release_identity_columns() -> None:
    result = build_disease_index(**_inputs()).collect()

    assert result['id'].to_list() == ['D1']
    assert result['name'].to_list() == ['asthma']
    assert result['description'].to_list() == ['a disease']
    assert result['entity'].to_list() == ['disease']
    assert result['category'][0].to_list() == ['respiratory']


def test_disease_keywords_hold_the_id_name_synonyms_and_trials() -> None:
    keywords = build_disease_index(**_inputs()).collect()['keywords'][0].to_list()

    assert set(keywords) == {'asthma', 'D1', 'exact', 'nct01'}


def test_disease_prefixes_exclude_the_id_and_trials() -> None:
    prefixes = build_disease_index(**_inputs()).collect()['prefixes'][0].to_list()

    assert set(prefixes) == {'asthma', 'exact'}


def test_disease_ngrams_add_phenotype_labels() -> None:
    ngrams = build_disease_index(**_inputs()).collect()['ngrams'][0].to_list()

    assert set(ngrams) == {'asthma', 'exact', 'wheeze'}


def test_disease_terms_merge_targets_drugs_and_studies() -> None:
    result = build_disease_index(**_inputs()).collect()

    assert set(result['terms'][0].to_list()) == {'EGFR', 'aspirin', 'S1'}
    assert set(result['terms5'][0].to_list()) == {'EGFR', 'aspirin', 'S1'}


def test_disease_multiplier_is_log1p_of_the_mean_score_plus_one() -> None:
    import math

    result = build_disease_index(**_inputs()).collect()

    assert result['multiplier'][0] == math.log1p(0.8) + 1.0


def test_disease_with_no_associations_falls_back_to_the_default_multiplier() -> None:
    inputs = _inputs(
        associations=pl.LazyFrame(
            {'associationId': [], 'targetId': [], 'diseaseId': [], 'score': []},
            schema={'associationId': pl.String, 'targetId': pl.String, 'diseaseId': pl.String, 'score': pl.Float64},
        ),
        scored_drugs=pl.LazyFrame(
            {'associationId': [], 'drugId': [], 'targetId': [], 'diseaseId': [], 'score': []},
            schema={
                'associationId': pl.String,
                'drugId': pl.String,
                'targetId': pl.String,
                'diseaseId': pl.String,
                'score': pl.Float64,
            },
        ),
    )

    result = build_disease_index(**inputs).collect()

    assert result['multiplier'][0] == 0.01
    assert result['terms'][0].to_list() == ['S1']


def test_disease_category_stays_null_when_no_therapeutic_area_resolves() -> None:
    """Pins the null-vs-empty distinction the equivalence bar depends on."""
    diseases = _inputs()['diseases'].with_columns(pl.lit(None, dtype=LIST_STR).alias('therapeutic_labels'))

    result = build_disease_index(**_inputs(diseases=diseases)).collect()

    assert result['category'][0] is None


def test_disease_index_emits_exactly_one_document_per_disease() -> None:
    """The one-document-per-entity invariant, at unit scale: two associations for one disease
    must not fan the disease out into two rows."""
    inputs = _inputs(
        associations=pl.LazyFrame(
            {
                'associationId': ['D1-T1', 'D1-T2'],
                'targetId': ['T1', 'T2'],
                'diseaseId': ['D1', 'D1'],
                'score': [0.8, 0.2],
            }
        ),
        t_lut=pl.LazyFrame(
            {'targetId': ['T1', 'T2'], 'target_labels': [['EGFR'], ['BRAF']]},
            schema={'targetId': pl.String, 'target_labels': LIST_STR},
        ),
    )

    result = build_disease_index(**inputs).collect()

    assert result.height == 1
    assert set(result['terms'][0].to_list()) >= {'EGFR', 'BRAF'}


def test_disease_tiers_respect_the_rank_cutoffs() -> None:
    """30 distinct-scored associations for one disease, enough to separate all three rank
    cutoffs: `terms5` must hold only the top 5 labels, `terms25` only the top 25, and `terms`
    (top 50) all 30. Fails if any of the 5/25/50 cutoffs moves."""
    count = 30
    target_ids = [f'T{i}' for i in range(1, count + 1)]
    labels = [f'L{i:02d}' for i in range(1, count + 1)]
    scores = [float(count - i) for i in range(count)]

    inputs = _inputs(
        associations=pl.LazyFrame(
            {
                'associationId': [f'D1-{target_id}' for target_id in target_ids],
                'targetId': target_ids,
                'diseaseId': ['D1'] * count,
                'score': scores,
            }
        ),
        scored_drugs=pl.LazyFrame(
            {'associationId': [], 'drugId': [], 'targetId': [], 'diseaseId': [], 'score': []},
            schema={
                'associationId': pl.String,
                'drugId': pl.String,
                'targetId': pl.String,
                'diseaseId': pl.String,
                'score': pl.Float64,
            },
        ),
        t_lut=pl.LazyFrame(
            {'targetId': target_ids, 'target_labels': [[label] for label in labels]},
            schema={'targetId': pl.String, 'target_labels': LIST_STR},
        ),
        studies=pl.LazyFrame(
            {'studyId': [], 'diseaseIds': []},
            schema={'studyId': pl.String, 'diseaseIds': LIST_STR},
        ),
    )

    result = build_disease_index(**inputs).collect()

    assert set(result['terms5'][0].to_list()) == set(labels[:5])
    assert set(result['terms25'][0].to_list()) == set(labels[:25])
    assert set(result['terms'][0].to_list()) == set(labels)


def test_disease_terms_include_every_distinct_study_for_a_disease() -> None:
    """Two distinct studies against one disease must both reach `terms`.

    `flatten_cat` deduplicates its output, so a disease appearing twice under the SAME study id
    is not distinguishable at this boundary from appearing once -- the studies aggregate uses
    `collect_list` semantics (duplicates preserved) rather than `collect_set`, matching spark, but
    that choice has no observable effect here because the final `terms`/`terms5`/`terms25`
    columns always pass through `flatten_cat`'s `list.unique`. This test instead pins what IS
    observable: that every DISTINCT study reaches the index, i.e. the aggregate is not silently
    collapsed to a single study per disease (e.g. by an accidental `.first()`).
    """
    inputs = _inputs(
        studies=pl.LazyFrame(
            {'studyId': ['S1', 'S2'], 'diseaseIds': [['D1'], ['D1']]},
            schema={'studyId': pl.String, 'diseaseIds': LIST_STR},
        ),
    )

    result = build_disease_index(**inputs).collect()

    assert {'S1', 'S2'} <= set(result['terms'][0].to_list())
