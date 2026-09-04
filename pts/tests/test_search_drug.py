"""Tests for the drug search index."""

import polars as pl

from pts.transformers.search.drug import build_drug_index
from pts.transformers.search.helpers import LIST_STR

LABELLED = pl.List(pl.Struct({'label': pl.String, 'source': pl.String}))
XREFS = pl.List(pl.Struct({'source': pl.String, 'ids': LIST_STR}))
ROWS = pl.List(pl.Struct({'mechanismOfAction': pl.String}))

DRUG_SCHEMA = {
    'drugId': pl.String,
    'name': pl.String,
    'description': pl.String,
    'drugType': pl.String,
    'synonyms': LABELLED,
    'tradeNames': LABELLED,
    'crossReferences': XREFS,
    'childChemblIds': LIST_STR,
    'rows': ROWS,
    'indications': LIST_STR,
}


def _drug(**overrides):
    row = {
        'drugId': 'CH1',
        'name': 'aspirin',
        'description': 'a drug',
        'drugType': 'Small molecule',
        'synonyms': [{'label': 'ASA', 'source': 'x'}],
        'tradeNames': [{'label': 'Bayer', 'source': 'x'}],
        'crossReferences': [{'source': 'drugbank', 'ids': ['DB01']}],
        'childChemblIds': ['CH2'],
        'rows': [{'mechanismOfAction': 'COX inhibitor'}],
        'indications': ['D1'],
    }
    row.update(overrides)
    return row


def _inputs(**overrides):
    base = {
        'drugs': pl.LazyFrame([_drug()], schema=DRUG_SCHEMA),
        'drug_assocs': pl.LazyFrame(
            {
                'drugId': ['CH1'],
                'targetIds': [['T1']],
                'diseaseIds': [['D1']],
                'meanScore': [0.5],
                'drug_relevance': [0.25],
            },
            schema={
                'drugId': pl.String,
                'targetIds': LIST_STR,
                'diseaseIds': LIST_STR,
                'meanScore': pl.Float64,
                'drug_relevance': pl.Float64,
            },
        ),
        't_lut': pl.LazyFrame(
            {'targetId': ['T1'], 'target_labels': [['EGFR']]},
            schema={'targetId': pl.String, 'target_labels': LIST_STR},
        ),
        'd_lut': pl.LazyFrame(
            {
                'diseaseId': ['D1'],
                'disease_labels': [['asthma']],
                'disease_name': ['asthma'],
                'therapeutic_labels': [['respiratory']],
            },
            schema={
                'diseaseId': pl.String,
                'disease_labels': LIST_STR,
                'disease_name': pl.String,
                'therapeutic_labels': LIST_STR,
            },
        ),
        'nct_by_drug': pl.LazyFrame(
            {'drugId': ['CH1'], 'nctIds': [['nct01']]},
            schema={'drugId': pl.String, 'nctIds': LIST_STR},
        ),
    }
    base.update(overrides)
    return base


def test_drug_index_carries_the_release_identity_columns() -> None:
    result = build_drug_index(**_inputs()).collect()

    assert result['id'].to_list() == ['CH1']
    assert result['name'].to_list() == ['aspirin']
    assert result['description'].to_list() == ['a drug']
    assert result['entity'].to_list() == ['drug']
    assert result['category'][0].to_list() == ['Small molecule']


def test_drug_keywords_hold_names_xrefs_and_trials() -> None:
    keywords = build_drug_index(**_inputs()).collect()['keywords'][0].to_list()

    assert set(keywords) == {'ASA', 'Bayer', 'aspirin', 'CH1', 'DB01', 'nct01'}


def test_drug_prefixes_swap_trials_for_the_mechanism_of_action() -> None:
    prefixes = build_drug_index(**_inputs()).collect()['prefixes'][0].to_list()

    assert set(prefixes) == {'ASA', 'Bayer', 'aspirin', 'COX inhibitor'}


def test_drug_terms_merge_diseases_targets_indications_areas_and_children() -> None:
    drugs = pl.LazyFrame([_drug(indications=['D2'])], schema=DRUG_SCHEMA)
    d_lut = pl.LazyFrame(
        {
            'diseaseId': ['D1', 'D2'],
            'disease_labels': [['asthma'], ['eczema']],
            'disease_name': ['asthma', 'eczema'],
            'therapeutic_labels': [['respiratory'], ['skin']],
        },
        schema={
            'diseaseId': pl.String,
            'disease_labels': LIST_STR,
            'disease_name': pl.String,
            'therapeutic_labels': LIST_STR,
        },
    )

    terms = build_drug_index(**_inputs(drugs=drugs, d_lut=d_lut)).collect()['terms'][0].to_list()

    # `drug_assocs.diseaseIds` (D1) drives disease_labels/therapeutic_labels; `drugs.indications`
    # (D2) is a separate join and must resolve to its own, distinct label.
    assert set(terms) == {'asthma', 'EGFR', 'respiratory', 'eczema', 'CH2'}


def test_drug_terms25_and_terms5_are_empty_matching_the_release() -> None:
    result = build_drug_index(**_inputs()).collect()

    assert result['terms25'][0].to_list() == []
    assert result['terms5'][0].to_list() == []


def test_drug_multiplier_is_log1p_of_relevance_plus_one() -> None:
    import math

    result = build_drug_index(**_inputs()).collect()

    assert result['multiplier'][0] == math.log1p(0.25) + 1.0


def test_drug_with_no_associations_falls_back_to_the_default_multiplier() -> None:
    drug_assocs = pl.LazyFrame(
        {'drugId': [], 'targetIds': [], 'diseaseIds': [], 'meanScore': [], 'drug_relevance': []},
        schema={
            'drugId': pl.String,
            'targetIds': LIST_STR,
            'diseaseIds': LIST_STR,
            'meanScore': pl.Float64,
            'drug_relevance': pl.Float64,
        },
    )

    result = build_drug_index(**_inputs(drug_assocs=drug_assocs)).collect()

    assert result['multiplier'][0] == 0.01
    assert result.height == 1


def test_cross_reference_ids_are_flattened_deduplicated_and_sorted() -> None:
    drugs = pl.LazyFrame(
        [_drug(crossReferences=[{'source': 'a', 'ids': ['z', 'y']}, {'source': 'b', 'ids': ['y', 'x']}])],
        schema=DRUG_SCHEMA,
    )

    keywords = build_drug_index(**_inputs(drugs=drugs)).collect()['keywords'][0].to_list()

    assert [k for k in keywords if k in {'x', 'y', 'z'}] == ['x', 'y', 'z']


def test_drug_index_emits_exactly_one_document_per_drug() -> None:
    drugs = pl.LazyFrame([_drug(indications=['D1', 'D1'])], schema=DRUG_SCHEMA)

    assert build_drug_index(**_inputs(drugs=drugs)).collect().height == 1
