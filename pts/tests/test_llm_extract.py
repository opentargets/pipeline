import asyncio
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from time import sleep
from types import SimpleNamespace
from typing import Any, cast

import polars as pl
import pytest

from pts import aact_trial_extraction
from pts.aact_trial_extraction import _run_extraction_in_thread
from pts.tasks.llm_extract import DEFAULT_SYSTEM_PROMPT, LlmExtract, LlmExtractSpec, _import_class, _schema_cache_uri
from pts.transformers.clinical import ClinicalReportExtractionSchema, llm_workflow, parse_batch_results
from tests.clinical.test_llm_extraction import TestParseSingleRecord


def test_registering_tasks_does_not_import_heavy_deps() -> None:
    script = (
        'import sys\n'
        'from otter.task.task_registry import TaskRegistry\n'
        "TaskRegistry(None, None).register('pts.tasks')\n"
        "heavy = {'openai', 'torch', 'transformers', 'mira', 'ontoma'}\n"
        "print(sorted({m.split('.')[0] for m in sys.modules} & heavy))\n"
    )
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '[]'


def test_default_system_prompt_is_packaged() -> None:
    task = LlmExtract(
        LlmExtractSpec(
            name='llm_extract test',
            source={'aact': 'aact.zip'},
            destination={},
            cache_uri='cache',
            snapshot='snapshot',
        ),
        SimpleNamespace(config=None),
    )

    assert DEFAULT_SYSTEM_PROMPT == ('pts.transformers.clinical.prompts', 'aact_llm.txt')
    assert task._system_prompt().startswith('You extract drug-indication evidence for Open Targets')


def test_default_model_class_is_the_extraction_schema() -> None:
    spec = LlmExtractSpec(
        name='llm_extract test', source={'aact': 'aact.zip'}, destination={}, cache_uri='cache', snapshot='snapshot'
    )

    assert spec.model_class == 'pts.transformers.clinical.schemas.ClinicalReportExtractionSchema'
    assert _import_class(spec.model_class) is ClinicalReportExtractionSchema


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

    prompts = aact_trial_extraction.prompts_for(report, 'schema-digest')

    # mira 1.0.5 build_prompts with publications disabled, via prompts_for
    assert prompts.to_dicts() == [
        {
            'id': 'nct00000001',
            'prompt': (
                'Trial ID: nct00000001\nOfficial Title: A Phase 2 Study of Aspirin\nDescription: Tests aspirin\n'
                'for headache.\nDetailed Description: null\nInterventions: aspirin; ibuprofen'
            ),
            'prompt_sha256': 'ad34534226e9e4bb74e1da9ef5f65fdd9f08d19c5094131f18ca2da120524e6e',
            'cache_key': '70aa829644d9d4ecc7c0321e741bb39395c02927bba67d48cc2504d1a59f5a63',
        },
        {
            'id': 'nct00000002',
            'prompt': (
                'Trial ID: nct00000002\nOfficial Title: null\nDescription: Brief only\n'
                'Detailed Description: Detailed protocol.'
            ),
            'prompt_sha256': '0f484a99130e21f5ccc612919f6f43c6f02ea56fce20755c5a31397c352af305',
            'cache_key': 'fed716ad4c120a80fb03717e3efd1421b2269739e7b0295361bcb30fb883a498',
        },
        {
            'id': 'nct00000003',
            'prompt': (
                'Trial ID: nct00000003\nOfficial Title: Étude: Sézary syndrome & K&L\nDescription: null\n'
                'Detailed Description: Multi\nline'
            ),
            'prompt_sha256': '7ea74f612091f04407ad77ecd516b3f9bcf2ab17652951b586c8d12dc9fb4533',
            'cache_key': '036c247a2c2191d29218239672b8731d579d212b471063002cca004c80a0ad16',
        },
    ]


def test_schema_cache_uri_namespaces_the_cache_by_schema() -> None:
    assert _schema_cache_uri('gs://bucket/cache/', 'schema-digest') == 'gs://bucket/cache/schema-digest'


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

    monkeypatch.setattr(aact_trial_extraction, 'read_dump_tables', lambda *args, **kwargs: tables)
    monkeypatch.setattr(aact_trial_extraction, 'extract_aact_clinical_report', fake_extract_clinical_report)
    monkeypatch.setattr(aact_trial_extraction, 'sample_report', lambda report, sample_size: report)

    aact_trial_extraction.trial_report(str(tmp_path / 'aact.zip'), str(tmp_path), None)

    detailed = cast(list[pl.DataFrame], captured['additional_metadata'])[2]
    assert detailed.columns == ['nct_id', 'detailed_description']
    assert detailed['detailed_description'].to_list() == ['Detail']


class _FakeClient:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_asyncio_proxy_delegates_and_closes_clients_on_its_event_loop() -> None:
    client = _FakeClient()
    proxy = aact_trial_extraction._AsyncioWithClientCleanup(asyncio, [client])

    async def return_result() -> str:
        return 'result'

    assert proxy.Semaphore is asyncio.Semaphore
    assert proxy.gather is asyncio.gather
    assert proxy.run(return_result()) == 'result'
    assert client.closed


def test_workflow_patches_are_restored_when_extraction_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    original_to_df = llm_workflow._extractions_to_df
    original_client_factory = llm_workflow.AsyncOpenAI

    def fail(**kwargs: object) -> None:
        assert llm_workflow.asyncio is not asyncio
        raise RuntimeError('extraction failed')

    monkeypatch.setattr(aact_trial_extraction, 'run_extraction', fail)

    with pytest.raises(RuntimeError, match='extraction failed'):
        _run_extraction_in_thread()

    assert llm_workflow._extractions_to_df is original_to_df
    assert llm_workflow.AsyncOpenAI is original_client_factory
    assert llm_workflow.asyncio is asyncio


def test_workflow_patches_are_serialised(monkeypatch: pytest.MonkeyPatch) -> None:
    state_lock = Lock()
    active = 0
    most_active = 0

    def observe(**kwargs: object) -> pl.DataFrame:
        nonlocal active, most_active
        with state_lock:
            active += 1
            most_active = max(most_active, active)
        sleep(0.05)
        with state_lock:
            active -= 1
        return pl.DataFrame()

    monkeypatch.setattr(aact_trial_extraction, 'run_extraction', observe)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_run_extraction_in_thread) for _ in range(2)]
        for future in futures:
            future.result()

    assert most_active == 1


def test_run_extraction_closes_async_client_before_loop_shutdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert aact_trial_extraction.run_extraction is llm_workflow.run_extraction
    client = _FakeClient()
    global_asyncio_run = asyncio.run

    def create_client(**kwargs: object) -> _FakeClient:
        return client

    async def fake_run_async(*args: object, **kwargs: object) -> tuple[list[object], list[object]]:
        assert asyncio.run is global_asyncio_run
        return [], []

    monkeypatch.setattr(llm_workflow, 'AsyncOpenAI', create_client)
    monkeypatch.setattr(llm_workflow, '_run_async', fake_run_async)
    prompt_path = tmp_path / 'system_prompt.txt'
    prompt_path.write_text('system prompt')

    result = _run_extraction_in_thread(
        prompts=[{'id': 'a', 'prompt': 'prompt'}, {'id': 'b', 'prompt': 'prompt'}],
        model_class='pts.transformers.clinical.schemas.ClinicalReportExtractionSchema',
        system_prompt_path=str(prompt_path),
        model='test-model',
        openai_key='test-key',
        service_tier='auto',
        concurrency=1,
        max_retries=0,
    )

    assert result is not None and result.is_empty()
    assert client.closed
    assert llm_workflow.asyncio is asyncio


def test_publish_writes_consolidated_errors(tmp_path: Path) -> None:
    errors_path = tmp_path / 'errors.jsonl'
    task = LlmExtract(
        LlmExtractSpec(
            name='llm_extract test',
            source={'aact': str(tmp_path / 'aact.zip')},
            destination={
                'prompts': str(tmp_path / 'prompts.parquet'),
                'extraction': str(tmp_path / 'extractions.parquet'),
                'errors': str(errors_path),
            },
            cache_uri=str(tmp_path / 'cache'),
            snapshot='snapshot',
        ),
        SimpleNamespace(config=None),
    )
    task.errors = ['{"id":"nct1"}', '{"id":"nct2"}']

    task._publish(pl.DataFrame({'id': ['nct1']}), pl.DataFrame({'id': ['nct1']}), str(tmp_path / 'cache'))

    assert errors_path.read_text() == '{"id":"nct1"}\n{"id":"nct2"}\n'


def _legacy_line(record_id: str) -> str:
    return TestParseSingleRecord.SAMPLE_LINE.replace('nct00031889', record_id)


def test_parse_batch_results_reads_nested_output_files(tmp_path: Path) -> None:
    (tmp_path / 'batch_2').mkdir()
    (tmp_path / 'batch_1_output.jsonl').write_text(_legacy_line('nct1') + '\n')
    (tmp_path / 'batch_2' / 'batch_2_output.jsonl').write_text(
        _legacy_line('nct2') + '\n\nnot json\n' + _legacy_line('nct3') + '\n'
    )
    (tmp_path / 'batch_1_input.jsonl').write_text(_legacy_line('ignored') + '\n')

    result = parse_batch_results(str(tmp_path), cast(Any, None))

    assert sorted(result.df['id'].to_list()) == ['nct1', 'nct2', 'nct3']


def test_parse_batch_results_raises_when_no_output_files(tmp_path: Path) -> None:
    (tmp_path / 'batch_1_input.jsonl').write_text('ignored')

    with pytest.raises(ValueError, match=r'No \*_output\.jsonl files found'):
        parse_batch_results(str(tmp_path), cast(Any, None))


def test_seed_legacy_cache_names_the_configured_path_when_nothing_is_staged(tmp_path: Path) -> None:
    legacy = tmp_path / 'legacy'
    legacy.mkdir()
    prompts = pl.DataFrame(schema={'id': pl.String, 'cache_key': pl.String, 'prompt_sha256': pl.String})

    with pytest.raises(ValueError, match=str(legacy)):
        aact_trial_extraction.seed_legacy_cache(prompts, str(legacy), str(tmp_path / 'cache'), cast(Any, None), 'snap')
