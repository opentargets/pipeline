import json

import polars as pl

from pts.transformers.clinical_mira.model_frames import polars_schema
from pts.transformers.clinical_mira.schemas import RoleDrug, TrialExtractionV2, TrialLink
from pts.transformers.clinical_mira.trial_links import (
    drug_synonym_entries,
    link_drug_labels,
    link_rows,
    report_entities,
    report_entities_frame,
    single_drug_trials,
)


def _link(drug: str, indication: str, confidence: float = 0.9) -> TrialLink:
    return TrialLink(
        drug=drug,
        synonyms=[],
        indication=indication,
        broader=indication,
        use='treat',
        arms=['A'],
        confidence=confidence,
        evidence_quote='q',
    )


def _extraction(**changes) -> TrialExtractionV2:
    fields = {
        'id': 'NCT1',
        'drug_intent': 'therapeutic',
        'drug_intent_confidence': 0.9,
        'links': [_link('drug x', 'lung cancer'), _link('drug x', 'mesothelioma'), _link('drug y', 'lung cancer')],
        'background_drugs': [RoleDrug(drug='cisplatin', role='backbone', synonyms=[], evidence_quote='all arms')],
        'other_drugs': [
            RoleDrug(drug='dexamethasone', role='premedication', synonyms=[], evidence_quote='premedication')
        ],
        'population_conditions': ['hypertension'],
    }
    return TrialExtractionV2(**(fields | changes))


def test_report_entities_come_from_links_only() -> None:
    drugs, diseases = report_entities(_extraction())
    assert drugs == ['drug x', 'drug y']
    assert diseases == ['lung cancer', 'mesothelioma']


def test_report_entities_of_a_trial_without_links_are_empty() -> None:
    assert report_entities(_extraction(links=[])) == ([], [])


def test_schema_has_no_numeric_bounds() -> None:
    schema = json.dumps(TrialExtractionV2.model_json_schema())
    assert 'minimum' not in schema
    assert 'maximum' not in schema


def test_link_rows_one_row_per_link() -> None:
    rows = link_rows({'nct1': _extraction(), 'nct2': _extraction(links=[_link('drug z', 'asthma', 0.4)])})
    assert rows.columns == ['id', 'drug', 'indication', 'broader', 'use', 'arms', 'confidence']
    assert rows.height == 4
    assert rows.filter(rows['id'] == 'nct2').row(0) == ('nct2', 'drug z', 'asthma', 'asthma', 'treat', ['A'], 0.4)


def test_the_drug_field_asks_for_the_generic_name() -> None:
    assert 'generic name' in TrialLink.model_json_schema()['properties']['drug']['description']


def test_a_link_carries_a_broader_term() -> None:
    assert 'broader' in TrialLink.model_json_schema()['required']


SCHEMA = pl.Schema({'id': pl.String, 'prompt_sha256': pl.String, **polars_schema(TrialExtractionV2)})


def v2_link(drug, indication, confidence=0.9, broader=None, synonyms=()):
    return {
        'drug': drug,
        'synonyms': list(synonyms),
        'indication': indication,
        'broader': broader or indication,
        'use': 'treat',
        'arms': [],
        'confidence': confidence,
        'evidence_quote': 'q',
    }


def v2_role(drug, kind, synonyms=()):
    return {'drug': drug, 'synonyms': list(synonyms), 'role': kind, 'evidence_quote': 'q'}


def v2_frame(rows: dict[str, dict]) -> pl.DataFrame:
    full = [
        {
            'id': trial_id,
            'prompt_sha256': 'x',
            'drug_intent': 'therapeutic',
            'drug_intent_confidence': 0.9,
            'links': [],
            'background_drugs': [],
            'other_drugs': [],
            'population_conditions': [],
            **row,
        }
        for trial_id, row in rows.items()
    ]
    return pl.DataFrame(full, schema=SCHEMA)


def test_report_entities_drop_links_below_threshold():
    df = v2_frame({
        'N1': {
            'links': [
                v2_link('b', 'asthma', 0.8, broader='lung disease'),
                v2_link('a', 'copd', 0.79),
                v2_link('b', 'asthma', 0.95, broader='other'),
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
