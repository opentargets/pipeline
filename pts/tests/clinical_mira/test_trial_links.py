import json

from pts.transformers.clinical_mira.schemas import RoleDrug, TrialExtractionV2, TrialLink
from pts.transformers.clinical_mira.trial_links import link_rows, report_entities


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
