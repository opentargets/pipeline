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
        JudgeVerdict.model_validate({'quoteSupports': 'maybe', 'primaryMechanism': 'yes', 'rationale': ''})


def test_prompts_load_and_state_text_only_rule():
    assert 'outside knowledge' in load_prompt('extract').lower()
    assert 'pharmacologist' in load_prompt('judge').lower()


def test_ruleset_changes_with_prompt_text(monkeypatch):
    before = ruleset_sha256('extract')
    monkeypatch.setattr(schemas, 'load_prompt', lambda name: 'something else')

    assert ruleset_sha256('extract') != before
