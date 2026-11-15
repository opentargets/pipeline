import json

from pts.transformers.clinical_mira.schemas import TrialExtractionV2, TrialLink
from pts.transformers.clinical_mira.trial_links import (
    drug_synonym_entries,
    link_drug_labels,
    report_entities_frame,
    single_drug_trials,
)
from tests.clinical_mira.extraction_v2_fixtures import v2_frame, v2_link, v2_role


def test_schema_has_no_numeric_bounds() -> None:
    schema = json.dumps(TrialExtractionV2.model_json_schema())
    assert 'minimum' not in schema
    assert 'maximum' not in schema


def test_the_drug_field_asks_for_the_generic_name() -> None:
    assert 'generic name' in TrialLink.model_json_schema()['properties']['drug']['description']


def test_a_link_carries_a_broader_term() -> None:
    assert 'broader' in TrialLink.model_json_schema()['required']


def test_report_entities_drop_links_below_threshold():
    df = v2_frame({
        'N1': {
            'links': [
                v2_link('b', 'asthma', 0.8, broader='lung disease'),
                v2_link('a', 'copd', 0.79),
                v2_link('b', 'asthma', 0.95, broader='other'),
                v2_link('c', 'gout', None),
            ]
        }
    })
    out = report_entities_frame(df, 0.8)
    assert out['nct_id'].to_list() == ['N1']
    assert out['drugFromSource'].to_list() == [['b']]
    assert out['diseaseFromSource'].to_list() == [['asthma']]
    assert out['diseaseBroader'].to_list() == [[{'indication': 'asthma', 'broader': 'lung disease'}]]


def test_trial_with_no_link_above_threshold_has_empty_lists():
    df = v2_frame({'N1': {'links': [v2_link('a', 'x', 0.1)]}, 'N2': {}, 'N3': {'links': [v2_link('a', 'x')]}})
    out = report_entities_frame(df, 0.8)
    assert out['nct_id'].to_list() == ['N1', 'N2', 'N3']
    assert out['drugFromSource'].to_list() == [[], [], ['a']]
    assert out['diseaseFromSource'].to_list() == [[], [], ['x']]
    assert out['diseaseBroader'].to_list()[:2] == [[], []]


def test_link_drug_labels_ignore_confidence_and_dedupe():
    df = v2_frame({'N1': {'links': [v2_link('a', 'x', 0.1), v2_link('a', 'y'), v2_link('b', 'x')]}, 'N2': {}})
    out = link_drug_labels(df)
    assert out.columns == ['id', 'label']
    assert sorted(out.rows()) == [('N1', 'a'), ('N1', 'b')]


def test_synonym_entries_cover_tested_and_role_drugs():
    df = v2_frame({
        'N1': {
            'links': [v2_link('a', 'x', synonyms=['A1']), v2_link('a', 'y', synonyms=['A1', 'A2'])],
            'background_drugs': [v2_role('bg', 'backbone', ['B1'])],
            'other_drugs': [v2_role('ot', 'rescue')],
        },
        'N2': {},
    })
    out = drug_synonym_entries(df)
    assert out.columns == ['id', 'drug', 'synonyms', 'role']
    rows = {(i, d, r): sorted(s) for i, d, s, r in out.rows()}
    assert rows == {('N1', 'a', 'tested'): ['A1', 'A2'], ('N1', 'bg', 'backbone'): ['B1'], ('N1', 'ot', 'rescue'): []}


def test_single_drug_trials_count_distinct_link_drugs():
    df = v2_frame({
        'N1': {'links': [v2_link('Aspirin', 'x'), v2_link(' aspirin ', 'y')]},
        'N2': {'links': [v2_link('a', 'x'), v2_link('b', 'x')]},
        'N3': {},
    })
    out = single_drug_trials(df)
    assert out.columns == ['id', 'drug']
    assert out.rows() == [('N1', 'Aspirin')]


def test_report_entities_come_from_links_only():
    df = v2_frame({
        'N1': {
            'links': [
                v2_link('drug y', 'lung cancer'),
                v2_link('drug x', 'mesothelioma'),
                v2_link('drug x', 'lung cancer'),
            ],
            'background_drugs': [v2_role('cisplatin', 'backbone')],
            'other_drugs': [v2_role('dexamethasone', 'premedication')],
            'population_conditions': ['hypertension'],
        }
    })
    out = report_entities_frame(df, 0.8)
    assert out['drugFromSource'].to_list() == [['drug x', 'drug y']]
    assert out['diseaseFromSource'].to_list() == [['lung cancer', 'mesothelioma']]
