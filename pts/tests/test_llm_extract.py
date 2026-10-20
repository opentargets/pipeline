import asyncio
import hashlib
import subprocess
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from time import sleep
from types import SimpleNamespace
from typing import Any, cast

import polars as pl
import pytest
from loguru import logger

from pts import aact_trial_extraction
from pts.aact_trial_extraction import _run_extraction_in_thread
from pts.result_cache import METADATA_SCHEMA, cache_key, cached_map, read_cache, write_cache
from pts.tasks.llm_extract import DEFAULT_SYSTEM_PROMPT, LlmExtract, LlmExtractSpec, _import_class, _schema_cache_uri
from pts.transformers.clinical import (
    ClinicalReportExtractionSchema,
    extract_aact_clinical_report,
    llm_workflow,
    parse_batch_results,
)
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
            'cache_key': '70fed69e9bb9173a55dde1c3addb9a8c48e02f5f0d9b2701c4c86550687676f2',
        },
        {
            'id': 'nct00000002',
            'prompt': (
                'Trial ID: nct00000002\nOfficial Title: null\nDescription: Brief only\n'
                'Detailed Description: Detailed protocol.'
            ),
            'prompt_sha256': '0f484a99130e21f5ccc612919f6f43c6f02ea56fce20755c5a31397c352af305',
            'cache_key': '91f6cce5820b6f3083e4150e040f40fcdcd1ed05777cc1b02370d21982a13e80',
        },
        {
            'id': 'nct00000003',
            'prompt': (
                'Trial ID: nct00000003\nOfficial Title: Étude: Sézary syndrome & K&L\nDescription: null\n'
                'Detailed Description: Multi\nline'
            ),
            'prompt_sha256': '7ea74f612091f04407ad77ecd516b3f9bcf2ab17652951b586c8d12dc9fb4533',
            'cache_key': 'f34dd6200dbf090edc71974f23a51c4315206fc2e262f413180860a06ae6d321',
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


@pytest.fixture
def logged() -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m).strip()), level='INFO', format='{message}')
    yield messages
    logger.remove(sink)


def _legacy_line(record_id: str) -> str:
    return SAMPLE_LINE.replace('nct00031889', record_id)


def test_parse_batch_results_reads_nested_output_files(tmp_path: Path, logged: list[str]) -> None:
    (tmp_path / 'batch_2').mkdir()
    (tmp_path / 'batch_1_output.jsonl').write_text(_legacy_line('nct1') + '\n')
    (tmp_path / 'batch_2' / 'batch_2_output.jsonl').write_text(
        _legacy_line('nct2') + '\n\nnot json\n' + _legacy_line('nct3') + '\n'
    )
    (tmp_path / 'batch_1_input.jsonl').write_text(_legacy_line('ignored') + '\n')

    result = parse_batch_results(str(tmp_path), cast(Any, None))

    assert sorted(result.df['id'].to_list()) == ['nct1', 'nct2', 'nct3']
    assert any(m.startswith('Dropped 1 malformed rows') for m in logged)


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


def _report(*titles: str) -> pl.DataFrame:
    return pl.DataFrame({
        'id': ['nct1'] * len(titles),
        'trialOfficialTitle': list(titles),
        'trialDescription': ['d'] * len(titles),
        'trialDetailedDescription': ['dd'] * len(titles),
    })


NO_PROMPTS = pl.DataFrame(schema={'id': pl.String, 'prompt_sha256': pl.String, 'cache_key': pl.String})


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _old_cache(tmp_path: Path, digest: str, ids: list[str], prompts: list[str | None] | None = None) -> str:
    uri = str(tmp_path / 'cache')
    data: dict[str, Any] = {
        'cache_key': [cache_key(i, digest) for i in ids],
        'computed_at': [None] * len(ids),
        'id': ids,
        'result': [f'r-{i}' for i in ids],
    }
    if prompts is not None:
        data['prompt_sha256'] = [None if p is None else _sha(p) for p in prompts]
    rows = pl.DataFrame(data, schema_overrides={'computed_at': METADATA_SCHEMA['computed_at']})
    write_cache(rows, uri, cast(Any, None), 'old')
    return uri


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
        assert again['cache_key'].to_list() == first['cache_key'].to_list()


def test_prompts_for_keys_on_trial_schema_and_prompt() -> None:
    prompts = aact_trial_extraction.prompts_for(_report('title one', 'title two'), 'digest')

    assert prompts['id'].to_list() == ['nct1', 'nct1']
    assert prompts['cache_key'].n_unique() == 2
    for row in prompts.to_dicts():
        assert row['cache_key'] == cache_key('nct1', 'digest', _sha(row['prompt']))


def test_rekey_cache_rewrites_keys_from_stored_columns(tmp_path: Path) -> None:
    uri = _old_cache(tmp_path, 'digest', ['a', 'b'], ['pa', 'pb'])

    assert aact_trial_extraction.rekey_cache(uri, 'digest', NO_PROMPTS, cast(Any, None), 'new') == 2

    rows = read_cache(uri, cast(Any, None)).sort('id')
    assert rows['cache_key'].to_list() == [cache_key('a', 'digest', _sha('pa')), cache_key('b', 'digest', _sha('pb'))]
    assert rows['result'].to_list() == ['r-a', 'r-b']


def test_rekey_cache_is_a_noop_the_second_time(tmp_path: Path) -> None:
    uri = _old_cache(tmp_path, 'digest', ['a'], ['pa'])
    aact_trial_extraction.rekey_cache(uri, 'digest', NO_PROMPTS, cast(Any, None), 'new')
    pointer = (Path(uri) / 'latest.json').read_text()

    assert aact_trial_extraction.rekey_cache(uri, 'digest', NO_PROMPTS, cast(Any, None), 'again') == 0

    assert (Path(uri) / 'latest.json').read_text() == pointer
    assert not (Path(uri) / 'snapshots' / 'again').exists()


def test_rekey_leaves_rows_without_prompt_hash(tmp_path: Path) -> None:
    uri = _old_cache(tmp_path, 'digest', ['a', 'b'], [None, 'pb'])

    assert aact_trial_extraction.rekey_cache(uri, 'digest', NO_PROMPTS, cast(Any, None), 'new') == 1

    rows = read_cache(uri, cast(Any, None)).sort('id')
    assert rows['cache_key'][0] == cache_key('a', 'digest')


def test_rekey_leaves_a_cache_without_a_prompt_hash_column(tmp_path: Path) -> None:
    uri = _old_cache(tmp_path, 'digest', ['a'])

    assert aact_trial_extraction.rekey_cache(uri, 'digest', NO_PROMPTS, cast(Any, None), 'new') == 0

    assert read_cache(uri, cast(Any, None))['cache_key'].to_list() == [cache_key('a', 'digest')]


def _computed(prompts: pl.DataFrame, uri: str) -> int:
    computed: list[int] = []

    def compute(shard: pl.DataFrame) -> pl.DataFrame:
        computed.append(shard.height)
        return shard.select('cache_key', 'id', 'prompt_sha256').with_columns(result=pl.lit('x'))

    cached_map(prompts, compute, uri, cast(Any, None), 'run', 'ts')
    return sum(computed)


def test_unchanged_prompt_is_a_hit_after_rekey(tmp_path: Path) -> None:
    prompts = aact_trial_extraction.prompts_for(_report('title'), 'digest')
    uri = _old_cache(tmp_path, 'digest', ['nct1'], [prompts['prompt'][0]])
    aact_trial_extraction.rekey_cache(uri, 'digest', prompts, cast(Any, None), 'new')

    assert _computed(prompts, uri) == 0


def test_pre_migration_rows_adopt_the_current_prompt_hash(tmp_path: Path) -> None:
    prompts = aact_trial_extraction.prompts_for(_report('title'), 'digest')
    uri = _old_cache(tmp_path, 'digest', ['nct1'], ['a stale prompt'])

    assert aact_trial_extraction.rekey_cache(uri, 'digest', prompts, cast(Any, None), 'new') == 1

    row = read_cache(uri, cast(Any, None)).to_dicts()[0]
    assert row['prompt_sha256'] == prompts['prompt_sha256'][0]
    assert row['cache_key'] == prompts['cache_key'][0]
    assert _computed(prompts, uri) == 0


def test_rekeyed_rows_are_never_readopted(tmp_path: Path) -> None:
    old = aact_trial_extraction.prompts_for(_report('title'), 'digest')
    uri = _old_cache(tmp_path, 'digest', ['nct1'], ['a stale prompt'])
    aact_trial_extraction.rekey_cache(uri, 'digest', old, cast(Any, None), 'new')
    new = aact_trial_extraction.prompts_for(_report('other title'), 'digest')

    assert aact_trial_extraction.rekey_cache(uri, 'digest', new, cast(Any, None), 'again') == 0

    assert _computed(new, uri) == 1


def test_pre_migration_row_for_a_trial_no_longer_in_aact_rekeys_from_its_stored_hash(tmp_path: Path) -> None:
    other = aact_trial_extraction.prompts_for(_report('title'), 'digest').with_columns(id=pl.lit('nct2'))
    uri = _old_cache(tmp_path, 'digest', ['nct1'], ['pa'])

    assert aact_trial_extraction.rekey_cache(uri, 'digest', other, cast(Any, None), 'new') == 1

    row = read_cache(uri, cast(Any, None)).to_dicts()[0]
    assert row['cache_key'] == cache_key('nct1', 'digest', _sha('pa'))
    assert row['prompt_sha256'] == _sha('pa')


def test_rekey_keeps_the_newest_of_rows_that_collide(tmp_path: Path) -> None:
    prompts = aact_trial_extraction.prompts_for(_report('title'), 'digest')
    uri = str(tmp_path / 'cache')
    new_key = prompts['cache_key'][0]
    rows = pl.DataFrame(
        {
            'cache_key': [cache_key('nct1', 'digest'), new_key],
            'computed_at': [datetime(2026, 9, 4, tzinfo=UTC), datetime(2026, 10, 1, tzinfo=UTC)],
            'id': ['nct1', 'nct1'],
            'prompt_sha256': ['stale', prompts['prompt_sha256'][0]],
            'result': ['old', 'new'],
        },
        schema_overrides={'computed_at': METADATA_SCHEMA['computed_at']},
    )
    write_cache(rows, uri, cast(Any, None), 'old')

    aact_trial_extraction.rekey_cache(uri, 'digest', prompts, cast(Any, None), 'new')

    assert read_cache(uri, cast(Any, None))['result'].to_list() == ['new']
