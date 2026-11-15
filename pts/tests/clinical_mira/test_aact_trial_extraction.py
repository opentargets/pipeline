from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, cast

import polars as pl
import pytest
from pydantic import BaseModel, Field

from pts.transformers.clinical_mira import aact_trial_extraction as extraction
from pts.transformers.clinical_mira import extract_aact_clinical_report
from pts.transformers.clinical_mira.model_frames import polars_schema
from pts.transformers.clinical_mira.providers import aact as aact_provider
from pts.transformers.clinical_mira.record_store import text_sha256


def test_prompts_without_publications_match_mira() -> None:
    report = pl.DataFrame({
        'id': ['nct00000001', 'nct00000002', 'nct00000003'],
        'trialOfficialTitle': ['A Phase 2 Study of Aspirin', None, 'Étude: Sézary syndrome & K&L'],
        'trialDescription': ['Tests aspirin\nfor headache.', 'Brief only', None],
        'trialDetailedDescription': [None, 'Detailed protocol.', 'Multi\nline'],
        'trialPhase': ['PHASE2', None, 'PHASE1'],
        'drugs': [
            [{'drugFromSource': 'aspirin', 'drugId': None}, {'drugFromSource': 'ibuprofen', 'drugId': None}],
            [{'drugFromSource': None, 'drugId': 'CHEMBL1'}],
            None,
        ],
    })

    prompts = extraction.prompts_for(report)

    # mira 1.0.5 build_prompts with publications disabled, via prompts_for
    assert prompts.to_dicts() == [
        {
            'id': 'nct00000001',
            'prompt': (
                'Trial ID: nct00000001\nOfficial Title: A Phase 2 Study of Aspirin\nDescription: Tests aspirin\n'
                'for headache.\nDetailed Description: null\nInterventions: aspirin; ibuprofen'
            ),
            'prompt_sha256': 'ad34534226e9e4bb74e1da9ef5f65fdd9f08d19c5094131f18ca2da120524e6e',
        },
        {
            'id': 'nct00000002',
            'prompt': (
                'Trial ID: nct00000002\nOfficial Title: null\nDescription: Brief only\n'
                'Detailed Description: Detailed protocol.'
            ),
            'prompt_sha256': '0f484a99130e21f5ccc612919f6f43c6f02ea56fce20755c5a31397c352af305',
        },
        {
            'id': 'nct00000003',
            'prompt': (
                'Trial ID: nct00000003\nOfficial Title: Étude: Sézary syndrome & K&L\nDescription: null\n'
                'Detailed Description: Multi\nline'
            ),
            'prompt_sha256': '7ea74f612091f04407ad77ecd516b3f9bcf2ab17652951b586c8d12dc9fb4533',
        },
    ]


def test_trial_report_passes_detailed_description_to_the_aact_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tables = {
        'studies': pl.DataFrame({
            'nct_id': ['nct1'],
            'study_type': ['interventional'],
            'phase': ['phase1'],
            'official_title': ['Trial'],
        }),
        'interventions': pl.DataFrame({'nct_id': ['nct1']}),
        'conditions': pl.DataFrame({'nct_id': ['nct1']}),
        'study_references': pl.DataFrame({'nct_id': ['nct1'], 'pmid': ['1']}),
        'brief_summaries': pl.DataFrame({'nct_id': ['nct1'], 'description': ['Brief']}),
        'detailed_descriptions': pl.DataFrame({'nct_id': ['nct1'], 'description': ['Detail']}),
    }
    captured: dict[str, Any] = {}

    def fake_extract_clinical_report(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(df=pl.DataFrame({'id': ['nct1']}))

    monkeypatch.setattr(aact_provider, 'read_dump_tables', lambda *args, **kwargs: tables)
    monkeypatch.setattr(extraction, 'extract_aact_clinical_report', fake_extract_clinical_report)
    monkeypatch.setattr(extraction, 'sample_report', lambda report, sample_size: report)

    extraction.trial_report(str(tmp_path / 'aact.zip'), str(tmp_path), None)

    detailed = cast(list[pl.DataFrame], captured['additional_metadata'])[2]
    assert detailed.columns == ['nct_id', 'detailed_description']
    assert detailed['detailed_description'].to_list() == ['Detail']


def _report(*ids: str, title: str = 'title') -> pl.DataFrame:
    return pl.DataFrame({
        'id': list(ids),
        'trialOfficialTitle': [f'{title} {i}' for i in ids],
        'trialDescription': ['d'] * len(ids),
        'trialDetailedDescription': ['dd'] * len(ids),
    })


def test_prompts_are_independent_of_row_order() -> None:
    studies = pl.DataFrame({
        'nct_id': ['n1', 'n2'],
        'study_type': ['INTERVENTIONAL'] * 2,
        'phase': ['PHASE2'] * 2,
        'official_title': ['t1', 't2'],
    })
    drugs = [f'drug{i}' for i in range(12)]
    interventions = pl.DataFrame({
        'nct_id': ['n1'] * 12 + ['n2'] * 3,
        'intervention_type': ['DRUG'] * 15,
        'name': [*drugs, 'x', 'y', 'z'],
    })
    conditions = pl.DataFrame({'nct_id': ['n1', 'n2'] * 2, 'downcase_name': ['a', 'b', 'c', 'd']})
    summaries = pl.DataFrame({'nct_id': ['n1', 'n2'], 'description': ['s1', 's2']})

    def prompts(seed: int) -> pl.DataFrame:
        report = extract_aact_clinical_report(
            studies=studies.sample(fraction=1, shuffle=True, seed=seed),
            interventions=interventions.sample(fraction=1, shuffle=True, seed=seed),
            conditions=conditions.sample(fraction=1, shuffle=True, seed=seed),
            additional_metadata=[summaries.sample(fraction=1, shuffle=True, seed=seed)],
        )
        return extraction.prompts_for(report.df).sort('id')

    first = prompts(1)
    for seed in (2, 3, 4):
        again = prompts(seed)
        assert again['prompt'].to_list() == first['prompt'].to_list()
        assert again['prompt_sha256'].to_list() == first['prompt_sha256'].to_list()


def test_prompts_for_hashes_each_prompt() -> None:
    prompts = extraction.prompts_for(_report('nct1', 'nct2'))

    assert prompts.columns == ['id', 'prompt', 'prompt_sha256']
    assert prompts['prompt_sha256'].to_list() == [text_sha256(p) for p in prompts['prompt']]


class Colour(StrEnum):
    RED = 'red'


class EnumField(BaseModel):
    id: str
    payload: Colour


class AnnotatedInList(BaseModel):
    id: str
    payload: list[Annotated[str, Field(min_length=1)]]


class ExcludedField(BaseModel):
    id: str
    payload: str = Field(default='', exclude=True)


class DictField(BaseModel):
    id: str
    payload: dict[str, str]


@pytest.mark.parametrize('model_class', [EnumField, AnnotatedInList, ExcludedField, DictField])
def test_an_unsupported_field_raises_naming_it(model_class: type[BaseModel]) -> None:
    with pytest.raises(TypeError, match=rf'^{model_class.__name__}\.payload[: ]'):
        polars_schema(model_class)
