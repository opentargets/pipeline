from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import mira.workflows.llm as llm_workflow
import polars as pl
import pytest

from pts.tasks import llm_extract
from pts.tasks.llm_extract import LlmExtract, LlmExtractSpec, _run_extraction_in_thread, _schema_cache_uri


def test_schema_cache_uri_namespaces_the_cache_by_schema() -> None:
    assert _schema_cache_uri('gs://bucket/cache/', 'schema-digest') == 'gs://bucket/cache/schema-digest'


def test_trial_report_passes_detailed_description_to_mira(
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

    class FakeStorageHandle:
        def __init__(self, path: str, **_: object) -> None:
            self.absolute = path

    def fake_extract_clinical_report(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(df=pl.DataFrame({'id': ['nct1']}))

    monkeypatch.setattr(llm_extract, 'StorageHandle', FakeStorageHandle)
    monkeypatch.setattr(llm_extract, 'read_dump_tables', lambda *args, **kwargs: tables)
    monkeypatch.setattr(llm_extract, 'extract_clinical_report', fake_extract_clinical_report)
    monkeypatch.setattr(llm_extract, 'sample_report', lambda report, sample_size: report)

    task = LlmExtract(
        LlmExtractSpec(
            name='llm_extract test',
            source={'aact': str(tmp_path / 'aact.zip')},
            destination={'prompts': 'prompts.parquet', 'extraction': 'extractions.parquet'},
            cache_uri='cache',
            snapshot='snapshot',
        ),
        SimpleNamespace(config=SimpleNamespace(work_path=str(tmp_path))),
    )

    task._trial_report()

    detailed = cast(list[pl.DataFrame], captured['additional_metadata'])[2]
    assert detailed.columns == ['nct_id', 'detailed_description']
    assert detailed['detailed_description'].to_list() == ['Detail']


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
