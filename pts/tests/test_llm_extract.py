from pathlib import Path

import mira.workflows.llm as llm_workflow
import polars as pl
import pytest
from mira.provider.aact import extract_clinical_report

from pts.result_cache import cache_key
from pts.tasks.llm_extract import LlmExtract, LlmExtractSpec, _additional_metadata, _run_extraction_in_thread


class _FakeClient:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_run_extraction_closes_async_client_before_loop_shutdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _FakeClient()

    def create_client(**kwargs: object) -> _FakeClient:
        return client

    async def fake_run_async(*args: object, **kwargs: object) -> tuple[list[object], list[object]]:
        return [], []

    monkeypatch.setattr(llm_workflow, 'AsyncOpenAI', create_client)
    monkeypatch.setattr(llm_workflow, '_run_async', fake_run_async)
    prompt_path = tmp_path / 'system_prompt.txt'
    prompt_path.write_text('system prompt')

    result = _run_extraction_in_thread(
        prompts=[{'id': 'a', 'prompt': 'prompt'}, {'id': 'b', 'prompt': 'prompt'}],
        model_class='mira.schemas.ClinicalReportExtractionSchema',
        system_prompt_path=str(prompt_path),
        model='test-model',
        openai_key='test-key',
        service_tier='auto',
        concurrency=1,
        max_retries=0,
    )

    assert result is not None and result.is_empty()
    assert client.closed


def test_detailed_description_reaches_prompt() -> None:
    tables = {
        'study_references': pl.DataFrame({'nct_id': ['nct1'], 'pmid': ['123']}),
        'brief_summaries': pl.DataFrame({'nct_id': ['nct1'], 'description': ['BRIEF SUMMARY TEXT']}),
        'detailed_descriptions': pl.DataFrame({
            'nct_id': ['nct1'],
            'description': ['DETAILED DESCRIPTION TEXT'],
        }),
    }
    report = extract_clinical_report(
        studies=pl.DataFrame({
            'nct_id': ['nct1'],
            'study_type': ['INTERVENTIONAL'],
            'phase': ['PHASE1'],
            'official_title': ['A title'],
        }),
        interventions=pl.DataFrame({
            'nct_id': ['nct1'],
            'intervention_type': ['DRUG'],
            'name': ['Aspirin'],
        }),
        conditions=pl.DataFrame({'nct_id': ['nct1'], 'downcase_name': ['headache']}),
        additional_metadata=_additional_metadata(tables),
        aggregation_specs={'pmid': {'group_by': 'nct_id', 'alias': 'literature'}},
    )
    task = object.__new__(LlmExtract)
    task.spec = LlmExtractSpec(
        name='llm_extract aact trials', source={}, destination={}, cache_uri='cache', snapshot='snapshot'
    )

    prompts = task._build_prompts(report.df, 'schema-digest')

    assert 'Detailed Description: DETAILED DESCRIPTION TEXT' in prompts.item(0, 'prompt')
    assert prompts.item(0, 'cache_key') == cache_key('nct1', 'schema-digest')
