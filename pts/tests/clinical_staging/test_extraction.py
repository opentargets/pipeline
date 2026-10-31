import asyncio
import hashlib
import json
import subprocess
import sys
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace
from typing import Annotated, Any, cast

import polars as pl
import pytest
from loguru import logger
from openai import AsyncOpenAI
from otter.config.model import Config
from otter.scratchpad.model import Scratchpad
from otter.task.model import TaskContext
from pydantic import BaseModel, Field, ValidationError

from pts.clinical_staging import extraction
from pts.clinical_staging.extraction import (
    EXTRACTION_MODEL,
    import_class,
    import_extraction_cache,
    models_to_frame,
    polars_schema,
)
from pts.clinical_staging.staged_outputs import EXTRACTION
from pts.record_store import ComputeAborted, read_records, text_sha256
from pts.tasks.llm_extract import DEFAULT_SYSTEM_PROMPT, LlmExtract, LlmExtractSpec
from pts.transformers.clinical import extract_aact_clinical_report
from pts.transformers.clinical.llm_workflow import Extracted, Failed
from pts.transformers.clinical.providers import aact as aact_provider
from pts.transformers.clinical.schemas import ClinicalReportExtractionSchema
from tests.clinical.batch_fixtures import SAMPLE_LINE
from tests.fake_openai import FakeClient, Outcome, bad_request, connection_error, unprocessable

TAG = EXTRACTION.release_tag()


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
    assert import_class(spec.model_class) is ClinicalReportExtractionSchema


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


SAMPLE_EXTRACTION = json.loads(SAMPLE_LINE)['response']['body']['output'][0]['content'][0]['text']


def _model(trial_id: str) -> ClinicalReportExtractionSchema:
    return ClinicalReportExtractionSchema.model_validate_json(SAMPLE_EXTRACTION.replace('nct00031889', trial_id))


def _trial_id(prompt: str) -> str:
    return prompt.split('\n', 1)[0].removeprefix('Trial ID: ')


def _invalid(trial_id: str) -> str:
    return json.dumps({'id': trial_id})


def _validation_error(trial_id: str) -> str:
    try:
        ClinicalReportExtractionSchema.model_validate_json(_invalid(trial_id))
    except ValidationError as exc:
        return f'ValidationError: {exc}'
    raise AssertionError(trial_id)


class FakeLlm:
    """Opens a ``FakeClient`` per extraction call, answering each trial with its sample extraction."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        failing: set[str] | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Trials in ``failing`` raise ``error``, or answer with JSON that does not validate when it is None."""
        self.failing = failing or set()
        self.error = error
        self.clients: list[FakeClient] = []
        self.keys: list[str] = []
        monkeypatch.setattr(extraction, '_open_client', self.open)

    def open(self, spec: LlmExtractSpec, openai_key: str) -> FakeClient:
        client = FakeClient(default=self.answer)
        self.clients.append(client)
        self.keys.append(openai_key)
        return client

    def answer(self, prompt: str) -> Outcome:
        trial_id = _trial_id(prompt)
        if trial_id not in self.failing:
            return SAMPLE_EXTRACTION.replace('nct00031889', trial_id)
        return self.error if self.error is not None else _invalid(trial_id)

    @property
    def calls(self) -> list[list[str]]:
        """The trials of each extraction call."""
        return [sorted(_trial_id(call['input']) for call in client.calls) for client in self.clients]


def _attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, report: pl.DataFrame, snapshot: str = 'v1', **spec: Any
) -> LlmExtract:
    """Run the task and validate it if the run succeeded; the failure reason is left on the manifest."""
    monkeypatch.setattr(extraction, 'trial_report', lambda *args: report)
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
    extracted = models_to_frame([_model(i) for i in ids], ClinicalReportExtractionSchema)
    keys = prompts.select('id', cache_key=pl.lit('k'), prompt_sha256='prompt_sha256')
    return (
        extracted
        .unique(subset='id')
        .join(keys, on='id', how='inner')
        .with_columns(computed_at=pl.lit(datetime(2026, 9, 4, tzinfo=UTC)).cast(pl.Datetime('us', 'UTC')))
    )


def test_extraction_publishes_only_validated_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    llm = FakeLlm(monkeypatch)
    report = _report('nct1', 'nct2')

    task = _run(monkeypatch, tmp_path, report)

    published = pl.read_parquet(tmp_path / f'v1/extraction/{TAG}/data.parquet')
    prompts = extraction.prompts_for(report)
    old = _old_way(['nct1', 'nct2'], prompts).drop('cache_key', 'computed_at')
    assert published.schema == old.schema
    assert published.sort('id').equals(old.sort('id'))
    metadata = json.loads((tmp_path / f'v1/extraction/{TAG}/metadata.json').read_text())
    assert metadata == EXTRACTION.metadata(_digest(), None)
    assert metadata['model_revisions'] == {EXTRACTION_MODEL: _digest()}
    assert task.stats == {'trials': 2, 'prompts': 2, 'extracted': 2, 'computed': 2, 'failed': 0, 'published': True}
    assert llm.keys == ['test-key']
    assert all(client.closed for client in llm.clients)


def test_llm_extract_writes_under_the_extraction_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeLlm(monkeypatch)

    task = _run(monkeypatch, tmp_path, _report('nct1', 'nct2'))

    runs = sorted(path.name for path in (tmp_path / 'store/extraction/runs').iterdir())
    assert len(runs) == 1
    assert runs[0].startswith('v1-')
    assert [artifact.source for artifact in task.artifacts or []] == [str(tmp_path / 'store/extraction')]
    assert not (tmp_path / 'store/runs').exists()


def test_a_rerun_publishes_nothing_and_a_new_digest_publishes_beside_the_old(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    FakeLlm(monkeypatch)
    report = _report('nct1', 'nct2')
    first = _run(monkeypatch, tmp_path, report)
    data = tmp_path / f'v1/extraction/{TAG}/data.parquet'
    written = data.stat().st_mtime_ns

    again = _run(monkeypatch, tmp_path, report)

    assert first.stats['published'] is True
    assert again.stats['published'] is False
    assert data.stat().st_mtime_ns == written

    monkeypatch.setattr('pts.clinical_staging.staged_outputs.schema_digest', lambda model_cls: 'f' * 64)
    monkeypatch.setattr('pts.tasks.llm_extract.new_run_id', lambda snapshot: f'{snapshot}-other-digest')
    other = EXTRACTION.tag('f' * 64, None)
    _run(monkeypatch, tmp_path, report)

    assert other != TAG
    assert sorted(p.name for p in (tmp_path / 'v1/extraction').iterdir()) == sorted([TAG, other])
    assert data.stat().st_mtime_ns == written
    metadata = json.loads((tmp_path / f'v1/extraction/{other}/metadata.json').read_text())
    assert metadata == EXTRACTION.metadata('f' * 64, None)


def test_a_sampled_extraction_publishes_under_its_own_tag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeLlm(monkeypatch)

    _run(monkeypatch, tmp_path, _report('nct1', 'nct2'), sample_size=5)

    sampled = EXTRACTION.tag(_digest(), 5)
    assert sampled != TAG
    assert (tmp_path / f'v1/extraction/{sampled}/data.parquet').exists()
    assert not (tmp_path / f'v1/extraction/{TAG}').exists()


def _records(prompts: pl.DataFrame, outputs: dict[str, str]) -> pl.DataFrame:
    return prompts.select(text='prompt', text_sha256='prompt_sha256', output=pl.col('id').replace_strict(outputs))


def test_an_invalid_stored_extraction_raises_with_its_trial() -> None:
    prompts = extraction.prompts_for(_report('nct1', 'nct2'))
    records = _records(prompts, {'nct1': _model('nct1').model_dump_json(), 'nct2': '{"id": "nct2"}'})

    with pytest.raises(ValueError, match='nct2'):
        extraction.extractions_from_records(records, prompts, ClinicalReportExtractionSchema)


def test_prompt_hash_is_attached_by_trial_id_not_by_position() -> None:
    prompts = extraction.prompts_for(_report('nct1', 'nct2'))
    records = _records(prompts, {i: _model(i).model_dump_json() for i in ('nct1', 'nct2')}).reverse()

    extractions = extraction.extractions_from_records(records, prompts, ClinicalReportExtractionSchema)

    assert extractions.select('id', 'prompt_sha256').rows() == list(
        reversed(prompts.select('id', 'prompt_sha256').rows())
    )


_DISEASE = pl.Struct({
    'name': pl.String,
    'severity': pl.String,
    'stage': pl.String,
    'onset': pl.String,
    'etiology': pl.String,
    'evidence_quote': pl.String,
})
_DRUG = pl.Struct({
    'drug': pl.String,
    'route': pl.String,
    'formulation': pl.String,
    'synonyms': pl.List(pl.String),
    'dosages': pl.List(pl.String),
    'evidence_quote': pl.String,
})
PUBLISHED_EXTRACTION_SCHEMA = pl.Schema({
    'id': pl.String,
    'drug_intent': pl.String,
    'drug_intent_confidence': pl.Float64,
    'primary_indications': pl.List(_DISEASE),
    'background_conditions': pl.List(_DISEASE),
    'investigated_drugs': pl.List(_DRUG),
    'comparator_drugs': pl.List(_DRUG),
    'supportive_drugs': pl.List(_DRUG),
    'conclusion': pl.String,
    'prompt_sha256': pl.String,
})
"""The parquet schema of the published extraction at tag 82af315af627-pfd3cdab9."""


def test_the_extraction_schema_is_the_published_one() -> None:
    derived = pl.Schema({**polars_schema(ClinicalReportExtractionSchema), 'prompt_sha256': pl.String})

    assert derived == PUBLISHED_EXTRACTION_SCHEMA


def test_null_and_empty_fields_keep_their_declared_dtype() -> None:
    data = json.loads(SAMPLE_EXTRACTION.replace('nct00031889', 'nct1'))
    data['background_conditions'] = []
    model = ClinicalReportExtractionSchema.model_validate(data)
    prompts = extraction.prompts_for(_report('nct1'))
    records = _records(prompts, {'nct1': model.model_dump_json()})

    extractions = extraction.extractions_from_records(records, prompts, ClinicalReportExtractionSchema)

    assert extractions.schema == PUBLISHED_EXTRACTION_SCHEMA
    row = extractions.row(0, named=True)
    assert row['background_conditions'] == []
    assert row['comparator_drugs'] is None
    assert row['supportive_drugs'] is None
    assert row['conclusion'] is None
    assert row['investigated_drugs'][0]['synonyms'] is None


def test_zero_extractions_have_the_full_schema() -> None:
    prompts = extraction.prompts_for(_report('nct1'))
    records = _records(prompts, {'nct1': _model('nct1').model_dump_json()}).clear()

    extractions = extraction.extractions_from_records(records, prompts, ClinicalReportExtractionSchema)

    assert extractions.is_empty()
    assert extractions.schema == PUBLISHED_EXTRACTION_SCHEMA


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


class UnsupportedExtraction(ClinicalReportExtractionSchema):
    payload: Colour | None = None


@pytest.mark.parametrize('model_class', [EnumField, AnnotatedInList, ExcludedField, DictField])
def test_an_unsupported_field_raises_naming_it(model_class: type[BaseModel]) -> None:
    with pytest.raises(TypeError, match=rf'^{model_class.__name__}\.payload[: ]'):
        polars_schema(model_class)


def test_an_unsupported_response_model_fails_before_any_llm_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    llm = FakeLlm(monkeypatch)

    task = _attempt(
        monkeypatch,
        tmp_path,
        _report('nct1', 'nct2'),
        model_class='tests.clinical_staging.test_extraction.UnsupportedExtraction',
    )

    assert task.manifest.failure_reason is not None
    assert 'UnsupportedExtraction.payload' in task.manifest.failure_reason
    assert llm.calls == []
    assert not (tmp_path / 'store').exists()


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
    llm = FakeLlm(monkeypatch)
    asked: list[list[str]] = []

    async def report_nothing(
        client: FakeClient, model_cls: type[BaseModel], system_prompt: str, prompts: Sequence[tuple[str, str]], **_: Any
    ) -> dict[str, Extracted | Failed]:
        asked.append(sorted(i for i, _ in prompts))
        return {}

    monkeypatch.setattr(extraction, 'extract_prompts', report_nothing)

    task = _attempt(monkeypatch, tmp_path, _report(*ids))

    assert task.manifest.failure_reason is not None
    assert 'none of 10 trials' in task.manifest.failure_reason
    assert 'no errors reported' in task.manifest.failure_reason
    assert asked == [ids]
    assert [client.closed for client in llm.clients] == [True]
    assert not (tmp_path / 'store').exists()
    assert not (tmp_path / 'v1/extraction').exists()


def test_systemic_errors_abort(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ids = [f'nct{i:02d}' for i in range(10)]
    llm = FakeLlm(monkeypatch, failing=set(ids), error=connection_error())

    task = _attempt(monkeypatch, tmp_path, _report(*ids))

    assert task.manifest.failure_reason is not None
    assert 'none of 10 trials returned an extraction, all for systemic reasons' in task.manifest.failure_reason
    assert 'APIConnectionError: Connection error.' in task.manifest.failure_reason
    assert llm.calls == [ids]
    assert not (tmp_path / 'store').exists()
    assert not (tmp_path / 'v1/extraction').exists()


@pytest.mark.parametrize('error', [bad_request(), unprocessable()], ids=['bad request', 'unprocessable'])
def test_a_retry_round_of_only_rejected_requests_does_not_abort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: BaseException
) -> None:
    bad = [f'nct{i:02d}' for i in range(1, 11)]
    llm = FakeLlm(monkeypatch, failing=set(bad), error=error)

    _run(monkeypatch, tmp_path, _report('nct00', *bad))

    assert llm.calls == [['nct00', *bad], bad, bad]
    assert pl.read_parquet(tmp_path / f'v1/extraction/{TAG}/data.parquet')['id'].to_list() == ['nct00']


def test_many_validation_failures_are_recorded_not_aborted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bad = [f'nct{i:02d}' for i in range(1, 12)]
    llm = FakeLlm(monkeypatch, failing=set(bad))
    report = _report('nct00', *bad)

    _run(monkeypatch, tmp_path, report)

    assert llm.calls == [['nct00', *bad], bad, bad]
    records = read_records(str(tmp_path / 'store/extraction'), cast(Any, None), pl.String())
    failed = records.filter(pl.col('status') == 'failed')
    assert failed.height == 33
    assert failed.group_by('text').len()['len'].to_list() == [3] * 11
    assert pl.read_parquet(tmp_path / f'v1/extraction/{TAG}/data.parquet')['id'].to_list() == ['nct00']
    llm.clients.clear()

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
    published = pl.read_parquet(tmp_path / f'v1/extraction/{TAG}/data.parquet')
    assert published['id'].to_list() == ['nct1']
    assert task.stats['failed'] == 1
    assert any(m.startswith('1 of 2 trials (50.0%) have no extraction') for m in logged)
    failed = read_records(str(tmp_path / 'store/extraction'), cast(Any, None), pl.String()).filter(
        pl.col('status') == 'failed'
    )
    assert failed['error'].to_list() == [_validation_error('nct2')] * 3
    snapshot = (tmp_path / f'v1/extraction/{TAG}/data.parquet').stat().st_mtime_ns
    llm.clients.clear()

    again = _run(monkeypatch, tmp_path, report)

    assert llm.calls == []
    assert again.stats['published'] is False
    assert (tmp_path / f'v1/extraction/{TAG}/data.parquet').stat().st_mtime_ns == snapshot


def _write_old_cache(tmp_path: Path, rows: pl.DataFrame, digest: str) -> str:
    uri = tmp_path / 'cache' / digest
    (uri / 'snapshots' / 's1').mkdir(parents=True)
    rows.write_parquet(uri / 'snapshots' / 's1' / 'rows.parquet')
    (uri / 'latest.json').write_text(json.dumps({'snapshot': 's1', 'rows': rows.height}))
    return str(uri)


def test_imported_cache_means_nothing_to_extract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    llm = FakeLlm(monkeypatch)
    report = _report('nct1', 'nct2')
    prompts = extraction.prompts_for(report)
    cached = _old_way(['nct1', 'nct2'], prompts.with_columns(prompt_sha256=pl.lit('stale')))
    cache_uri = _write_old_cache(tmp_path, cached, _digest())

    _run(monkeypatch, tmp_path, report, import_extraction_cache=cache_uri)

    assert llm.calls == []
    published = pl.read_parquet(tmp_path / f'v1/extraction/{TAG}/data.parquet')
    expected = cached.drop('cache_key', 'computed_at').with_columns(
        prompt_sha256=pl.col('id').replace_strict(prompts['id'], prompts['prompt_sha256'])
    )
    assert published.sort('id').equals(expected.sort('id'))


def test_import_runs_once(tmp_path: Path) -> None:
    prompts = extraction.prompts_for(_report('nct1', 'nct2'))
    older = extraction.prompts_for(_report('nct1', 'nct2', 'nct3', title='old'))
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
        prompts = extraction.prompts_for(_report('nct1'))
        cache_uri = _write_old_cache(tmp_path, _old_way(['nct1'], prompts), digest)

        assert import_extraction_cache(cache_uri, prompts, store, digest, cast(Any, None)) == 1

    records = read_records(store, cast(Any, None), output_dtype=pl.String()).sort('model_revision')
    assert records['run_id'].to_list() == [
        'import-extraction-cache-aaaaaaaaaaaa',
        'import-extraction-cache-bbbbbbbbbbbb',
    ]


def test_import_refuses_a_cache_of_another_schema(tmp_path: Path) -> None:
    prompts = extraction.prompts_for(_report('nct1'))
    cache_uri = _write_old_cache(tmp_path, _old_way(['nct1'], prompts), 'other-digest')

    with pytest.raises(ValueError, match='other-digest'):
        import_extraction_cache(cache_uri, prompts, str(tmp_path / 'store'), 'digest', cast(Any, None))

    assert not (tmp_path / 'store').exists()


def test_a_changed_prompt_is_re_extracted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    llm = FakeLlm(monkeypatch)
    _run(monkeypatch, tmp_path, _report('nct1', 'nct2'), snapshot='v1')
    llm.clients.clear()

    _run(monkeypatch, tmp_path, _report('nct1', title='new').vstack(_report('nct2')), snapshot='v2')

    assert llm.calls == [['nct1']]


def test_model_change_does_not_re_extract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    llm = FakeLlm(monkeypatch)
    _run(monkeypatch, tmp_path, _report('nct1', 'nct2'), snapshot='v1')
    llm.clients.clear()
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
    assert pl.read_parquet(tmp_path / f'v2/extraction/{TAG}/data.parquet')['id'].to_list() == ['nct1', 'nct2']


def _late_populated(trial_id: str) -> ClinicalReportExtractionSchema:
    """An extraction whose nullable fields, null in the sample, all have values."""
    data = json.loads(SAMPLE_EXTRACTION.replace('nct00031889', trial_id))
    data['investigated_drugs'][0]['synonyms'] = ['aromasin']
    data['primary_indications'][0]['severity'] = 'metastatic'
    data['comparator_drugs'] = [data['investigated_drugs'][1]]
    data['supportive_drugs'] = []
    data['conclusion'] = 'Exemestane was well tolerated.'
    return ClinicalReportExtractionSchema.model_validate(data)


def _spec(tmp_path: Path, **spec: Any) -> LlmExtractSpec:
    key = tmp_path / 'openai-key'
    key.write_text('test-key')
    return LlmExtractSpec(
        name='llm_extract test',
        source={'aact': 'aact.zip'},
        destination={},
        store_uri='store',
        snapshot='snapshot',
        openai_key_path=str(key),
        **spec,
    )


def _shard(*ids: str) -> pl.DataFrame:
    return extraction.prompts_for(_report(*ids)).select('id', 'prompt')


def _extract_in_loop(shard: pl.DataFrame, spec: LlmExtractSpec) -> tuple[dict[str, BaseModel], dict[str, str]]:
    """Call ``extract`` from a running event loop, as Otter does."""

    async def call() -> tuple[dict[str, BaseModel], dict[str, str]]:
        return extraction.extract(shard, spec, 'system prompt', ClinicalReportExtractionSchema)

    return asyncio.run(call())


def test_fewer_than_ten_systemic_failures_do_not_abort(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ids = [f'nct{i:02d}' for i in range(9)]
    llm = FakeLlm(monkeypatch, failing=set(ids), error=connection_error())

    task = _attempt(monkeypatch, tmp_path, _report(*ids))

    assert task.manifest.failure_reason is not None
    assert 'no trial of 9 was extracted' in task.manifest.failure_reason
    assert llm.calls == [ids, ids, ids]
    failed = read_records(str(tmp_path / 'store/extraction'), cast(Any, None), pl.String())
    assert failed['status'].to_list() == ['failed'] * 27
    assert set(failed['error']) == {'APIConnectionError: Connection error.'}


def test_a_systemic_retry_round_aborts_and_keeps_the_first_round(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bad = [f'nct{i:02d}' for i in range(1, 11)]
    llm = FakeLlm(monkeypatch, failing=set(bad), error=connection_error())

    task = _attempt(monkeypatch, tmp_path, _report('nct00', *bad))

    assert task.manifest.failure_reason is not None
    assert 'none of 10 trials returned an extraction' in task.manifest.failure_reason
    assert llm.calls == [['nct00', *bad], bad]
    records = read_records(str(tmp_path / 'store/extraction'), cast(Any, None), pl.String()).sort('text')
    assert records['status'].to_list() == ['ok'] + ['failed'] * 10
    assert json.loads(records['output'][0])['id'] == 'nct00'
    assert not (tmp_path / 'v1/extraction').exists()


def test_a_one_trial_shard_makes_one_call(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    llm = FakeLlm(monkeypatch)

    _run(monkeypatch, tmp_path, _report('nct1'))

    assert llm.calls == [['nct1']]
    assert len(llm.clients[0].calls) == 1
    records = read_records(str(tmp_path / 'store/extraction'), cast(Any, None), pl.String())
    assert records['status'].to_list() == ['ok']
    assert ClinicalReportExtractionSchema.model_validate_json(records['output'][0]) == _model('nct1')
    assert pl.read_parquet(tmp_path / f'v1/extraction/{TAG}/data.parquet')['id'].to_list() == ['nct1']


def test_extract_returns_models_and_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    llm = FakeLlm(monkeypatch, failing={'nct2'})

    extractions, errors = _extract_in_loop(_shard('nct1', 'nct2'), _spec(tmp_path))

    assert extractions == {'nct1': _model('nct1')}
    assert errors == {'nct2': _validation_error('nct2')}
    assert llm.keys == ['test-key']


def test_client_is_closed_on_its_loop_after_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    llm = FakeLlm(monkeypatch)

    _extract_in_loop(_shard('nct1', 'nct2'), _spec(tmp_path))

    (client,) = llm.clients
    assert client.closed
    assert client.entered_loop is not None
    assert client.closed_loop is client.entered_loop


def test_client_is_closed_on_its_loop_after_an_abort(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ids = [f'nct{i:02d}' for i in range(10)]
    llm = FakeLlm(monkeypatch, failing=set(ids), error=connection_error())

    with pytest.raises(ComputeAborted, match='none of 10 trials'):
        _extract_in_loop(_shard(*ids), _spec(tmp_path))

    (client,) = llm.clients
    assert client.closed
    assert client.entered_loop is not None
    assert client.closed_loop is client.entered_loop


def test_client_is_closed_on_its_loop_when_extraction_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    llm = FakeLlm(monkeypatch)

    async def fail(*args: object, **kwargs: object) -> dict[str, Extracted | Failed]:
        raise RuntimeError('extraction failed')

    monkeypatch.setattr(extraction, 'extract_prompts', fail)

    with pytest.raises(RuntimeError, match='extraction failed'):
        _extract_in_loop(_shard('nct1'), _spec(tmp_path))

    (client,) = llm.clients
    assert client.closed
    assert client.entered_loop is not None
    assert client.closed_loop is client.entered_loop


def test_an_empty_key_aborts_without_recording_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    llm = FakeLlm(monkeypatch)
    key = tmp_path / 'empty-key'
    key.write_text(' \n')

    task = _attempt(monkeypatch, tmp_path, _report('nct1', 'nct2'), openai_key_path=str(key))

    assert task.manifest.failure_reason is not None
    assert f'the OpenAI key at {key} is empty' in task.manifest.failure_reason
    assert llm.clients == []
    assert not (tmp_path / 'store').exists()
    assert not (tmp_path / 'v1/extraction').exists()


def test_two_extractions_run_at_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    llm = FakeLlm(monkeypatch)
    both_in_flight = Barrier(2, timeout=10)
    answer = llm.answer

    def answer_together(prompt: str) -> Outcome:
        both_in_flight.wait()
        return answer(prompt)

    monkeypatch.setattr(llm, 'answer', answer_together)
    spec = _spec(tmp_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_extract_in_loop, _shard(trial_id), spec) for trial_id in ('nct1', 'nct2')]
        results = [future.result() for future in futures]

    assert results == [({'nct1': _model('nct1')}, {}), ({'nct2': _model('nct2')}, {})]
    assert len(llm.clients) == 2
    assert all(client.closed for client in llm.clients)


def test_open_client_sets_the_timeout_and_retries(tmp_path: Path) -> None:
    for service_tier, timeout in (('flex', 900), ('auto', 600)):
        client = extraction._open_client(_spec(tmp_path, service_tier=service_tier, max_retries=5), 'k')

        assert isinstance(client, AsyncOpenAI)
        assert client.api_key == 'k'
        assert client.timeout == timeout
        assert client.max_retries == 5
        asyncio.run(client.close())


@pytest.mark.parametrize(
    'models',
    [
        [_model('nct000')],
        [_model(f'nct{i:03d}') for i in range(120)] + [_late_populated('nct120')],
    ],
    ids=['sample', 'late-populated'],
)
def test_stored_json_publishes_as_imported_cache_rows(models: list[ClinicalReportExtractionSchema]) -> None:
    ids = [model.id for model in models]
    prompts = extraction.prompts_for(_report(*ids))
    stored = {model.id: model.model_dump_json() for model in models}
    imported = {
        row['id']: json.dumps(row) for row in models_to_frame(models, ClinicalReportExtractionSchema).to_dicts()
    }
    mixed = {trial_id: (stored if n % 2 else imported)[trial_id] for n, trial_id in enumerate(ids)}

    def publish(outputs: dict[str, str]) -> pl.DataFrame:
        records = _records(prompts, outputs)
        return extraction.extractions_from_records(records, prompts, ClinicalReportExtractionSchema)

    expected = publish(imported)
    assert expected.height == len(models)
    assert publish(stored).equals(expected)
    assert publish(mixed).equals(expected)
