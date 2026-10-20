import asyncio
import hashlib
import json
import subprocess
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock
from time import sleep
from types import SimpleNamespace
from typing import Any, cast

import polars as pl
import pytest
from loguru import logger
from otter.config.model import Config
from otter.scratchpad.model import Scratchpad
from otter.task.model import TaskContext

from pts import aact_trial_extraction
from pts.aact_trial_extraction import (
    EXTRACTION_MODEL,
    _models_to_dataframe,
    _run_extraction_in_thread,
    import_extraction_cache,
)
from pts.record_store import read_records, text_sha256
from pts.tasks.llm_extract import DEFAULT_SYSTEM_PROMPT, LlmExtract, LlmExtractSpec, _import_class
from pts.transformers.clinical import ClinicalReportExtractionSchema, extract_aact_clinical_report, llm_workflow
from tests.clinical.batch_fixtures import SAMPLE_LINE


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
            store_uri='store',
            snapshot='snapshot',
        ),
        SimpleNamespace(config=None),
    )

    assert DEFAULT_SYSTEM_PROMPT == ('pts.transformers.clinical.prompts', 'aact_llm.txt')
    assert task._system_prompt().startswith('You extract drug-indication evidence for Open Targets')


def test_default_model_class_is_the_extraction_schema() -> None:
    spec = LlmExtractSpec(
        name='llm_extract test', source={'aact': 'aact.zip'}, destination={}, store_uri='store', snapshot='snapshot'
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


def test_run_extraction_logs_its_counts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, logged: list[str]) -> None:
    errors: list[object] = [{'id': f'nct{i}', 'error_type': 'Boom', 'error_message': 'bad'} for i in range(7)]

    async def fake_run_async(*args: object, **kwargs: object) -> tuple[list[object], list[object]]:
        return [], errors

    monkeypatch.setattr(llm_workflow, 'AsyncOpenAI', lambda **kwargs: _FakeClient())
    monkeypatch.setattr(llm_workflow, '_run_async', fake_run_async)
    prompt_path = tmp_path / 'system_prompt.txt'
    prompt_path.write_text('system prompt')

    _run_extraction_in_thread(
        prompts=[{'id': 'a', 'prompt': 'prompt'}, {'id': 'b', 'prompt': 'prompt'}],
        model_class='pts.transformers.clinical.schemas.ClinicalReportExtractionSchema',
        system_prompt_path=str(prompt_path),
        model='test-model',
        openai_key='test-key',
        service_tier='auto',
        concurrency=3,
        max_retries=0,
        errors_dir=tmp_path,
    )

    assert 'Running extraction: model=test-model  concurrency=3  service_tier=auto' in logged
    assert 'Extraction complete: 0 succeeded, 7 failed.' in logged
    assert any(m.startswith('Failed  id=nct0') and 'type=Boom  message=bad' in m for m in logged)
    assert any(m.startswith('... plus 2 more errors') for m in logged)


@pytest.fixture
def logged() -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m).strip()), level='INFO', format='{message}')
    yield messages
    logger.remove(sink)


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
        return aact_trial_extraction.prompts_for(report.df, 'digest').sort('id')

    first = prompts(1)
    for seed in (2, 3, 4):
        again = prompts(seed)
        assert again['prompt'].to_list() == first['prompt'].to_list()
        assert again['prompt_sha256'].to_list() == first['prompt_sha256'].to_list()


def test_prompts_for_hashes_each_prompt() -> None:
    prompts = aact_trial_extraction.prompts_for(_report('nct1', 'nct2'), 'digest')

    assert prompts.columns == ['id', 'prompt', 'prompt_sha256']
    assert prompts['prompt_sha256'].to_list() == [text_sha256(p) for p in prompts['prompt']]


SAMPLE_EXTRACTION = json.loads(SAMPLE_LINE)['response']['body']['output'][0]['content'][0]['text']


def _model(trial_id: str) -> ClinicalReportExtractionSchema:
    return ClinicalReportExtractionSchema.model_validate_json(SAMPLE_EXTRACTION.replace('nct00031889', trial_id))


class FakeLlm:
    """Stands in for the LLM workflow, remembering which trials it was asked about."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        failing: set[str] | None = None,
        error_type: str | None = 'ValidationError',
    ) -> None:
        """Trials in ``failing`` fail with ``error_type``, or with no reported error when it is None."""
        self.calls: list[list[str]] = []
        self.failing = failing or set()
        self.error_type = error_type
        monkeypatch.setattr(aact_trial_extraction, '_run_extraction_in_thread', self.run)

    def run(self, **kwargs: Any) -> pl.DataFrame:
        ids = sorted({p['id'] for p in kwargs['prompts']})
        self.calls.append(ids)
        errors = [
            {'id': i, 'error_type': self.error_type, 'error_message': f'bad {i}'}
            for i in ids
            if i in self.failing and self.error_type
        ]
        if errors:
            (Path(kwargs['errors_dir']) / 'errors.jsonl').write_text('\n'.join(json.dumps(e) for e in errors) + '\n')
        return _models_to_dataframe([_model(i) for i in ids if i not in self.failing])


def _attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, report: pl.DataFrame, snapshot: str = 'v1', **spec: Any
) -> LlmExtract:
    """Run the task and validate it if the run succeeded; the failure reason is left on the manifest."""
    monkeypatch.setattr(aact_trial_extraction, 'trial_report', lambda *args: report)
    key = tmp_path / 'openai-key'
    key.write_text('test-key')
    spec.setdefault('openai_key_path', str(key))
    context = TaskContext(
        config=Config(step='test', steps=['test'], work_path=tmp_path / 'work'), scratchpad=Scratchpad()
    )
    context.abort = Event()
    task = LlmExtract(
        LlmExtractSpec(
            name='llm_extract test',
            source={'aact': str(tmp_path / 'aact.zip')},
            destination={'extraction': str(tmp_path / snapshot / 'extraction')},
            store_uri=str(tmp_path / 'store'),
            snapshot=snapshot,
            **spec,
        ),
        context,
    )
    asyncio.run(task.run())
    if task.manifest.failure_reason is None:
        asyncio.run(task.validate())
    return task


def _run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, report: pl.DataFrame, snapshot: str = 'v1', **spec: Any
) -> LlmExtract:
    task = _attempt(monkeypatch, tmp_path, report, snapshot, **spec)

    # the `report` decorator swallows exceptions, so surface the failure reason
    assert task.manifest.failure_reason is None, task.manifest.failure_reason
    return task


def _digest() -> str:
    schema = ClinicalReportExtractionSchema.model_json_schema(by_alias=True)
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode('utf-8')).hexdigest()


def _old_way(ids: list[str], prompts: pl.DataFrame) -> pl.DataFrame:
    """Rows as the cache-backed task cached them; it published them without ``computed_at``."""
    extracted = _models_to_dataframe([_model(i) for i in ids])
    keys = prompts.select('id', cache_key=pl.lit('k'), prompt_sha256='prompt_sha256')
    return (
        extracted
        .unique(subset='id')
        .join(keys, on='id', how='inner')
        .with_columns(computed_at=pl.lit(datetime(2026, 9, 4, tzinfo=UTC)).cast(pl.Datetime('us', 'UTC')))
    )


def test_extraction_publishes_only_validated_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeLlm(monkeypatch)
    report = _report('nct1', 'nct2')

    task = _run(monkeypatch, tmp_path, report)

    published = pl.read_parquet(tmp_path / 'v1/extraction/data.parquet')
    prompts = aact_trial_extraction.prompts_for(report, _digest())
    old = _old_way(['nct1', 'nct2'], prompts).drop('cache_key', 'computed_at')
    assert published.schema == old.schema
    assert published.sort('id').equals(old.sort('id'))
    metadata = json.loads((tmp_path / 'v1/extraction/metadata.json').read_text())
    assert metadata == {'model': EXTRACTION_MODEL, 'model_revision': _digest()}
    assert task.stats == {'trials': 2, 'prompts': 2, 'extracted': 2, 'failed': 0, 'published': True}


def _records(prompts: pl.DataFrame, outputs: dict[str, str]) -> pl.DataFrame:
    return prompts.select(text='prompt', text_sha256='prompt_sha256', output=pl.col('id').replace_strict(outputs))


def test_an_invalid_stored_extraction_raises_with_its_trial() -> None:
    prompts = aact_trial_extraction.prompts_for(_report('nct1', 'nct2'), 'digest')
    records = _records(prompts, {'nct1': _model('nct1').model_dump_json(), 'nct2': '{"id": "nct2"}'})

    with pytest.raises(ValueError, match='nct2'):
        aact_trial_extraction.extractions_from_records(records, prompts, ClinicalReportExtractionSchema)


def test_prompt_hash_follows_its_trial_when_a_row_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = aact_trial_extraction.prompts_for(_report('nct1', 'nct2'), 'digest')
    records = _records(prompts, {i: _model(i).model_dump_json() for i in ('nct1', 'nct2')})
    monkeypatch.setattr(aact_trial_extraction, '_models_to_dataframe', lambda models: _models_to_dataframe(models[1:]))

    extractions = aact_trial_extraction.extractions_from_records(records, prompts, ClinicalReportExtractionSchema)

    assert extractions.select('id', 'prompt_sha256').rows() == [('nct2', prompts['prompt_sha256'][1])]


def test_missing_key_aborts_without_recording_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    llm = FakeLlm(monkeypatch)

    task = _attempt(monkeypatch, tmp_path, _report('nct1', 'nct2'), openai_key_path=str(tmp_path / 'no-key'))

    assert task.manifest.failure_reason is not None
    assert 'no-key' in task.manifest.failure_reason
    assert llm.calls == []
    assert not (tmp_path / 'store').exists()
    assert not (tmp_path / 'v1/extraction').exists()


def test_a_call_with_no_extractions_and_no_errors_aborts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ids = [f'nct{i:02d}' for i in range(10)]
    llm = FakeLlm(monkeypatch, failing=set(ids), error_type=None)

    task = _attempt(monkeypatch, tmp_path, _report(*ids))

    assert task.manifest.failure_reason is not None
    assert 'none of 10 trials' in task.manifest.failure_reason
    assert llm.calls == [ids]
    assert not (tmp_path / 'store').exists()
    assert not (tmp_path / 'v1/extraction').exists()


def test_systemic_errors_abort(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ids = [f'nct{i:02d}' for i in range(10)]
    llm = FakeLlm(monkeypatch, failing=set(ids), error_type='APIConnectionError')

    task = _attempt(monkeypatch, tmp_path, _report(*ids))

    assert task.manifest.failure_reason is not None
    assert 'APIConnectionError: bad nct00' in task.manifest.failure_reason
    assert llm.calls == [ids]
    assert not (tmp_path / 'store').exists()
    assert not (tmp_path / 'v1/extraction').exists()


@pytest.mark.parametrize('error_type', ['BadRequestError', 'UnprocessableEntityError'])
def test_a_retry_round_of_only_rejected_requests_does_not_abort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error_type: str
) -> None:
    bad = [f'nct{i:02d}' for i in range(1, 11)]
    llm = FakeLlm(monkeypatch, failing=set(bad), error_type=error_type)

    _run(monkeypatch, tmp_path, _report('nct00', *bad))

    assert llm.calls == [['nct00', *bad], bad, bad]
    assert pl.read_parquet(tmp_path / 'v1/extraction/data.parquet')['id'].to_list() == ['nct00']


def test_many_validation_failures_are_recorded_not_aborted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bad = [f'nct{i:02d}' for i in range(1, 12)]
    llm = FakeLlm(monkeypatch, failing=set(bad))
    report = _report('nct00', *bad)

    _run(monkeypatch, tmp_path, report)

    assert llm.calls == [['nct00', *bad], bad, bad]
    records = read_records(str(tmp_path / 'store'), cast(Any, None), pl.String())
    failed = records.filter(pl.col('status') == 'failed')
    assert failed.height == 33
    assert failed.group_by('text').len()['len'].to_list() == [3] * 11
    assert pl.read_parquet(tmp_path / 'v1/extraction/data.parquet')['id'].to_list() == ['nct00']
    llm.calls.clear()

    _run(monkeypatch, tmp_path, report, snapshot='v1')

    assert llm.calls == []


def test_nothing_extracted_publishes_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    llm = FakeLlm(monkeypatch, failing={'nct1', 'nct2'})

    task = _attempt(monkeypatch, tmp_path, _report('nct1', 'nct2'))

    assert task.manifest.failure_reason is not None
    assert 'no trial' in task.manifest.failure_reason
    assert llm.calls == [['nct1', 'nct2'], ['nct1', 'nct2'], ['nct1', 'nct2']]
    assert not (tmp_path / 'v1/extraction').exists()


def test_failed_trial_is_left_out_and_not_retried_on_rerun(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, logged: list[str]
) -> None:
    llm = FakeLlm(monkeypatch, failing={'nct2'})
    report = _report('nct1', 'nct2')

    task = _run(monkeypatch, tmp_path, report)

    assert llm.calls == [['nct1', 'nct2'], ['nct2'], ['nct2']]
    published = pl.read_parquet(tmp_path / 'v1/extraction/data.parquet')
    assert published['id'].to_list() == ['nct1']
    assert task.stats['failed'] == 1
    assert any(m.startswith('1 of 2 trials (50.0%) have no extraction') for m in logged)
    failed = read_records(str(tmp_path / 'store'), cast(Any, None), pl.String()).filter(pl.col('status') == 'failed')
    assert failed['error'].to_list() == ['ValidationError: bad nct2'] * 3
    snapshot = (tmp_path / 'v1/extraction/data.parquet').stat().st_mtime_ns
    llm.calls.clear()

    again = _run(monkeypatch, tmp_path, report)

    assert llm.calls == []
    assert again.stats['published'] is False
    assert (tmp_path / 'v1/extraction/data.parquet').stat().st_mtime_ns == snapshot


def _write_old_cache(tmp_path: Path, rows: pl.DataFrame, digest: str) -> str:
    uri = tmp_path / 'cache' / digest
    (uri / 'snapshots' / 's1').mkdir(parents=True)
    rows.write_parquet(uri / 'snapshots' / 's1' / 'rows.parquet')
    (uri / 'latest.json').write_text(json.dumps({'snapshot': 's1', 'rows': rows.height}))
    return str(uri)


def test_imported_cache_means_nothing_to_extract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    llm = FakeLlm(monkeypatch)
    report = _report('nct1', 'nct2')
    prompts = aact_trial_extraction.prompts_for(report, _digest())
    cached = _old_way(['nct1', 'nct2'], prompts.with_columns(prompt_sha256=pl.lit('stale')))
    cache_uri = _write_old_cache(tmp_path, cached, _digest())

    _run(monkeypatch, tmp_path, report, import_extraction_cache=cache_uri)

    assert llm.calls == []
    published = pl.read_parquet(tmp_path / 'v1/extraction/data.parquet')
    expected = cached.drop('cache_key', 'computed_at').with_columns(
        prompt_sha256=pl.col('id').replace_strict(prompts['id'], prompts['prompt_sha256'])
    )
    assert published.sort('id').equals(expected.sort('id'))


def test_import_runs_once(tmp_path: Path) -> None:
    prompts = aact_trial_extraction.prompts_for(_report('nct1', 'nct2'), 'digest')
    older = aact_trial_extraction.prompts_for(_report('nct1', 'nct2', 'nct3', title='old'), 'digest')
    cached = _old_way(['nct1', 'nct2', 'nct3'], older).with_columns(
        computed_at=pl.when(pl.col('id') != 'nct2').then(pl.col('computed_at'))
    )
    cache_uri = _write_old_cache(tmp_path, cached, 'digest')
    store = str(tmp_path / 'store')

    assert import_extraction_cache(cache_uri, prompts, store, 'digest', cast(Any, None)) == 2
    assert import_extraction_cache(cache_uri, prompts, store, 'digest', cast(Any, None)) == 0

    records = read_records(store, cast(Any, None), output_dtype=pl.String()).sort('text')
    assert records['text'].to_list() == prompts['prompt'].to_list()
    assert records['text_sha256'].to_list() == prompts['prompt_sha256'].to_list()
    assert records['model'].to_list() == [EXTRACTION_MODEL] * 2
    assert records['model_revision'].to_list() == ['digest'] * 2
    assert records['status'].to_list() == ['ok'] * 2
    assert records['run_id'].to_list() == ['import-extraction-cache-digest'] * 2
    assert records['computed_at'].to_list() == [datetime(2026, 9, 4, tzinfo=UTC), datetime(1970, 1, 1, tzinfo=UTC)]
    outputs = [json.loads(o) for o in records['output']]
    assert [o['id'] for o in outputs] == ['nct1', 'nct2']
    assert not {'cache_key', 'prompt_sha256', 'computed_at'} & set(outputs[0])
    assert ClinicalReportExtractionSchema.model_validate(outputs[0]) == _model('nct1')


def test_each_schema_digest_is_imported_once(tmp_path: Path) -> None:
    digests = ['a' * 64, 'b' * 64]
    store = str(tmp_path / 'store')
    for digest in digests:
        prompts = aact_trial_extraction.prompts_for(_report('nct1'), digest)
        cache_uri = _write_old_cache(tmp_path, _old_way(['nct1'], prompts), digest)

        assert import_extraction_cache(cache_uri, prompts, store, digest, cast(Any, None)) == 1

    records = read_records(store, cast(Any, None), output_dtype=pl.String()).sort('model_revision')
    assert records['run_id'].to_list() == [
        'import-extraction-cache-aaaaaaaaaaaa',
        'import-extraction-cache-bbbbbbbbbbbb',
    ]


def test_import_refuses_a_cache_of_another_schema(tmp_path: Path) -> None:
    prompts = aact_trial_extraction.prompts_for(_report('nct1'), 'digest')
    cache_uri = _write_old_cache(tmp_path, _old_way(['nct1'], prompts), 'other-digest')

    with pytest.raises(ValueError, match='other-digest'):
        import_extraction_cache(cache_uri, prompts, str(tmp_path / 'store'), 'digest', cast(Any, None))

    assert not (tmp_path / 'store').exists()


def test_a_changed_prompt_is_re_extracted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    llm = FakeLlm(monkeypatch)
    _run(monkeypatch, tmp_path, _report('nct1', 'nct2'), snapshot='v1')
    llm.calls.clear()

    _run(monkeypatch, tmp_path, _report('nct1', title='new').vstack(_report('nct2')), snapshot='v2')

    assert llm.calls == [['nct1']]


def test_model_change_does_not_re_extract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    llm = FakeLlm(monkeypatch)
    _run(monkeypatch, tmp_path, _report('nct1', 'nct2'), snapshot='v1')
    llm.calls.clear()
    system_prompt = tmp_path / 'system_prompt.txt'
    system_prompt.write_text('a different system prompt')

    _run(
        monkeypatch,
        tmp_path,
        _report('nct1', 'nct2'),
        snapshot='v2',
        model='another-model',
        system_prompt=str(system_prompt),
    )

    assert llm.calls == []
    assert pl.read_parquet(tmp_path / 'v2/extraction/data.parquet')['id'].to_list() == ['nct1', 'nct2']
