import hashlib

import pytest
from pydantic import ValidationError

from pts.transformers.clinical_mira.moa import schemas
from pts.transformers.clinical_mira.moa.schemas import (
    ActionType,
    Evidence,
    JudgeVerdict,
    MoaExtraction,
    TargetType,
    load_prompt,
    ruleset_sha256,
)

MECHANISM = {
    'targetFromSource': 'KRAS G12C',
    'targetSymbolSuggestion': ['KRAS'],
    'isHumanTarget': True,
    'actsOnReceptor': False,
    'relation': 'direct',
    'statementType': 'drug_description',
    'targetType': 'single protein',
    'actionType': 'INHIBITOR',
    'mechanismOfAction': 'KRAS G12C inhibitor',
    'evidence': [{'nctId': 'nct1', 'quote': 'a KRAS G12C inhibitor'}],
    'confidence': 0.9,
}


def test_moa_extraction_round_trips():
    extraction = MoaExtraction.model_validate({'drugName': 'x', 'notStated': False, 'mechanisms': [MECHANISM]})

    assert extraction.mechanisms[0].evidence[0].nct_id == 'nct1'
    assert extraction.mechanisms[0].action_type is ActionType.INHIBITOR
    assert extraction.mechanisms[0].target_type is TargetType.SINGLE_PROTEIN
    assert extraction.mechanisms[0].acts_on_receptor is False


def test_response_schema_uses_camel_case_names():
    properties = MoaExtraction.model_json_schema(by_alias=True)['$defs']['Mechanism']['properties']

    assert {'targetFromSource', 'targetSymbolSuggestion', 'isHumanTarget', 'actionType'} <= set(properties)


def test_extra_fields_rejected():
    with pytest.raises(ValidationError):
        Evidence.model_validate({'nctId': 'nct1', 'quote': 'q', 'extra': 1})


def test_action_types_cover_benchmark_values():
    assert {'INHIBITOR', 'ANTAGONIST', 'AGONIST', 'BINDING AGENT', 'OTHER'} <= {a.value for a in ActionType}
    assert {'single protein', 'protein family', 'OTHER'} <= {t.value for t in TargetType}


def test_judge_verdict_values_are_closed():
    with pytest.raises(ValidationError):
        JudgeVerdict.model_validate({
            'targetSupported': 'partly',
            'actionSupported': 'yes',
            'primaryMechanism': 'yes',
            'rationale': '',
        })


def test_prompts_load_and_state_text_only_rule():
    assert 'outside knowledge' in load_prompt('extract').lower()
    assert 'pharmacologist' in load_prompt('judge').lower()


def test_extract_prompt_states_the_pilot_lessons():
    prompt = load_prompt('extract').lower()

    assert 'the receptor, not the ligand' in prompt
    assert 'primary mechanism' in prompt
    assert 'states that this drug acts on the target' in prompt


def test_ruleset_changes_with_prompt_text(monkeypatch):
    before = ruleset_sha256('extract')
    monkeypatch.setattr(schemas, 'load_prompt', lambda name: 'something else')

    assert ruleset_sha256('extract') != before


def test_relation_and_statement_type_are_closed():
    extraction = MoaExtraction.model_validate({'drugName': 'x', 'notStated': False, 'mechanisms': [MECHANISM]})

    assert extraction.mechanisms[0].relation == 'direct'
    assert extraction.mechanisms[0].statement_type == 'drug_description'
    with pytest.raises(ValidationError):
        MoaExtraction.model_validate({
            'drugName': 'x',
            'notStated': False,
            'mechanisms': [MECHANISM | {'relation': 'indirect'}],
        })


def test_extract_prompt_states_the_full_run_lessons():
    prompt = load_prompt('extract').lower()

    assert 'changes_amount' in prompt
    assert 'hypothesis' in prompt
    assert 'decoy' in prompt
    assert 'releasing agent' in prompt
    assert 'list every member' in prompt
    assert 'class of drugs' in prompt


def test_literature_system_prompt_matches_experiment() -> None:
    prompt = schemas.literature_system_prompt()
    assert prompt.startswith(load_prompt('extract'))
    assert (
        hashlib.sha256(prompt.encode()).hexdigest()
        == 'b104cd4b33d7517e63643da4ba67d51688f8e3f60e9768dc19faac0e6a835d6d'
    )
