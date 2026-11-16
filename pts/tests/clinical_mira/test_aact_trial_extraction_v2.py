import json
from importlib import resources
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest
from otter.config.model import Config
from otter.util.errors import TaskRunError

from pts.transformers.clinical_mira import aact_trial_extraction as extraction
from pts.transformers.clinical_mira import aact_trial_extraction_v2 as v2
from pts.transformers.clinical_mira.aact_trial_extraction_v2 import AactTrialExtractionV2Settings, stage_extraction_v2
from pts.transformers.clinical_mira.moa.claude import custom_id
from pts.transformers.clinical_mira.model_frames import polars_schema
from pts.transformers.clinical_mira.providers.aact_context import RENDER_VERSION, render_trial, trial_context
from pts.transformers.clinical_mira.record_store import ComputeAborted, read_records, text_sha256
from pts.transformers.clinical_mira.schemas import TrialExtractionV2
from pts.transformers.clinical_mira.staged_outputs import (
    EXTRACTION_V2,
    EXTRACTION_V2_MAX_TOKENS,
    EXTRACTION_V2_MODEL,
    EXTRACTION_V2_PROMPT,
    extraction_v2_revision,
)

IDS = ['NCT1', 'NCT2', 'NCT3']
WORKSPACE = 'ANTHROPIC_GOOGLE_CLOUD_WORKSPACE_ID'


def _tables(ids: list[str] = IDS) -> dict[str, pl.DataFrame]:
    n = len(ids)
    return {
        'studies': pl.DataFrame({
            'nct_id': ids,
            'official_title': [f'Drug {i} for disease {i}' for i in ids],
            'phase': ['PHASE2'] * n,
        }),
        'summaries': pl.DataFrame({'nct_id': ids, 'description': [f'Does drug {i} help?' for i in ids]}),
        'details': pl.DataFrame({'nct_id': ids, 'description': ['Randomised.'] * n}),
        'interventions': pl.DataFrame({
            'id': list(range(n)),
            'nct_id': ids,
            'intervention_type': ['DRUG'] * n,
            'name': [f'drug {i}' for i in ids],
            'description': ['10 mg daily'] * n,
        }),
        'design_groups': pl.DataFrame({
            'id': list(range(10, 10 + n)),
            'nct_id': ids,
            'group_type': ['EXPERIMENTAL'] * n,
            'title': [f'arm {i}' for i in ids],
            'description': ['drug arm'] * n,
        }),
        'design_group_interventions': pl.DataFrame({
            'id': list(range(100, 100 + n)),
            'nct_id': ids,
            'design_group_id': list(range(10, 10 + n)),
            'intervention_id': list(range(n)),
        }),
        'design_outcomes': pl.DataFrame({'nct_id': ids, 'outcome_type': ['primary'] * n, 'measure': ['Survival'] * n}),
        'designs': pl.DataFrame({
            'nct_id': ids,
            'primary_purpose': ['TREATMENT'] * n,
            'allocation': ['RANDOMIZED'] * n,
            'intervention_model': ['PARALLEL'] * n,
            'masking': ['NONE'] * n,
        }),
    }


def _answer(text: str) -> dict:
    nct_id = text.split('\n', 1)[0].removeprefix('Trial: ')
    return {
        'id': nct_id.lower(),
        'drug_intent': 'therapeutic',
        'drug_intent_confidence': 0.9,
        'links': [
            {
                'drug': f'drug {nct_id}',
                'synonyms': [],
                'indication': f'disease {nct_id}',
                'broader': f'disease {nct_id}',
                'use': 'treat',
                'arms': [f'arm {nct_id}'],
                'confidence': 0.9,
                'evidence_quote': f'Drug {nct_id} for disease {nct_id}',
            }
        ],
        'background_drugs': [],
        'other_drugs': [],
        'population_conditions': [],
    }


class FakeBatches:
    """Message Batches stand-in answering every trial, refusing those in ``refuse`` up to ``refusals`` times each.

    With ``interrupt`` the first ``retrieve`` raises, as a run killed while polling. ``unanswered`` maps a trial
    to the result it gets instead of an answer.
    """

    def __init__(
        self,
        refuse: frozenset[str] = frozenset(),
        refusals: int | None = None,
        interrupt: bool = False,
        unanswered: dict[str, SimpleNamespace] | None = None,
    ) -> None:
        self.refuse = {custom_id(i): refusals for i in refuse}
        self.interrupt = interrupt
        self.unanswered = {custom_id(i): result for i, result in (unanswered or {}).items()}
        self.created: list[list[dict]] = []

    def create(self, *, requests: list[dict]) -> SimpleNamespace:
        self.created.append(requests)
        return SimpleNamespace(id=f'batch_{len(self.created)}', processing_status='in_progress')

    def retrieve(self, batch_id: str) -> SimpleNamespace:
        if self.interrupt:
            self.interrupt = False
            msg = 'killed while polling'
            raise RuntimeError(msg)
        return SimpleNamespace(id=batch_id, processing_status='ended', request_counts=SimpleNamespace(processing=0))

    def results(self, batch_id: str):
        for request in self.created[int(batch_id.split('_')[1]) - 1]:
            if request['custom_id'] in self.unanswered:
                yield SimpleNamespace(custom_id=request['custom_id'], result=self.unanswered[request['custom_id']])
                continue
            text = request['params']['messages'][0]['content']
            stop = 'end_turn'
            left = self.refuse.get(request['custom_id'], 0)
            if left is None or left > 0:
                stop = 'refusal'
                self.refuse[request['custom_id']] = None if left is None else left - 1
            message = SimpleNamespace(
                stop_reason=stop,
                content=[SimpleNamespace(type='text', text=json.dumps(_answer(text)))],
                usage=SimpleNamespace(
                    input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0
                ),
            )
            yield SimpleNamespace(
                custom_id=request['custom_id'], result=SimpleNamespace(type='succeeded', message=message)
            )

    @property
    def trials(self) -> list[str]:
        """The trial of every request submitted, in order."""
        return [
            r['params']['messages'][0]['content'].split('\n', 1)[0].removeprefix('Trial: ')
            for requests in self.created
            for r in requests
        ]


def _errored(error_type: str) -> SimpleNamespace:
    """An errored batch result, shaped as the API returns it."""
    return SimpleNamespace(type='errored', error=SimpleNamespace(type='error', error=SimpleNamespace(type=error_type)))


def _client(batches: FakeBatches) -> Any:
    return SimpleNamespace(messages=SimpleNamespace(batches=batches))


def _config(tmp_path: Path) -> Config:
    return Config(step='test', steps=['test'], work_path=tmp_path / 'work')


def _stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    batches: FakeBatches | None,
    report_ids: list[str] = IDS,
    run_id: str = 'v1-run1',
    **settings: Any,
) -> dict[str, Any]:
    """Stage ``report_ids`` lowercased, as ``trial_report`` gives them, against context tables with AACT ids."""
    report = pl.DataFrame({'id': [i.lower() for i in report_ids]})
    monkeypatch.setattr(extraction, 'trial_report', lambda *args: report)
    monkeypatch.setattr(v2, 'read_context_tables', lambda *args: _tables())
    monkeypatch.setattr(v2, 'new_run_id', lambda snapshot: run_id)
    return stage_extraction_v2(
        str(tmp_path / 'aact.zip'),
        str(tmp_path / 'extraction_v2'),
        AactTrialExtractionV2Settings(store_uri=str(tmp_path / 'store'), snapshot='v1', **settings),
        _config(tmp_path),
        client=None if batches is None else _client(batches),
    )


def _revision() -> str:
    package, filename = EXTRACTION_V2_PROMPT
    prompt = (resources.files(package) / filename).read_text(encoding='utf-8')
    return extraction_v2_revision('claude-haiku-5-5', prompt, RENDER_VERSION, 'medium', EXTRACTION_V2_MAX_TOKENS)


def _snapshot(tmp_path: Path) -> Path:
    return tmp_path / 'extraction_v2' / EXTRACTION_V2.release_tag()


def _state(tmp_path: Path) -> Path:
    return tmp_path / 'store' / 'batches' / f'extraction_v2-v1-{EXTRACTION_V2.release_tag()}.json'


def test_extracts_and_publishes_a_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    batches = FakeBatches()

    stats = _stage(monkeypatch, tmp_path, batches)

    assert stats == {'trials': 3, 'prompts': 3, 'extracted': 3, 'computed': 3, 'failed': 0, 'published': True}
    params = batches.created[0][0]['params']
    assert params['model'] == 'claude-haiku-5-5'
    assert params['output_config']['effort'] == 'medium'
    published = pl.read_parquet(_snapshot(tmp_path) / 'data.parquet')
    fields = {name: dtype for name, dtype in polars_schema(TrialExtractionV2).items() if name != 'id'}
    assert published.schema == pl.Schema({'id': pl.String, 'prompt_sha256': pl.String, **fields})
    assert published['id'].to_list() == IDS
    rendered = [render_trial(row) for row in trial_context(**_tables()).iter_rows(named=True)]
    assert published['prompt_sha256'].to_list() == [text_sha256(text) for text in rendered]
    assert published['links'].list.eval(pl.element().struct.field('drug')).to_list() == [[f'drug {i}'] for i in IDS]
    metadata = json.loads((_snapshot(tmp_path) / 'metadata.json').read_text())
    assert metadata == EXTRACTION_V2.metadata(_revision(), None)
    records = read_records(str(tmp_path / 'store' / 'extraction_v2'), _config(tmp_path), pl.String())
    assert set(records['model']) == {EXTRACTION_V2_MODEL}
    assert set(records['model_revision']) == {_revision()}
    assert params['max_tokens'] == EXTRACTION_V2_MAX_TOKENS
    assert (tmp_path / 'work' / f'extraction_v2_cache-{EXTRACTION_V2.release_tag()}.jsonl').exists()
    assert json.loads(_state(tmp_path).read_text()) == {'pending': []}


def test_rerun_extracts_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stage(monkeypatch, tmp_path, FakeBatches())
    batches = FakeBatches()

    stats = _stage(monkeypatch, tmp_path, batches, run_id='v1-run2')

    assert batches.created == []
    assert stats['computed'] == 0
    assert stats['extracted'] == 3
    assert stats['published'] is False


def test_resumes_a_submitted_batch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    batches = FakeBatches(interrupt=True)
    with pytest.raises(ComputeAborted):
        _stage(monkeypatch, tmp_path, batches)
    assert json.loads(_state(tmp_path).read_text()) == {'pending': ['batch_1']}

    stats = _stage(monkeypatch, tmp_path, batches, run_id='v1-run2')

    assert len(batches.created) == 1
    assert stats['extracted'] == 3
    assert json.loads(_state(tmp_path).read_text()) == {'pending': []}


def test_failed_trials_are_left_out_and_not_retried_by_a_rerun(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    batches = FakeBatches(refuse=frozenset({'NCT2'}))
    stats = _stage(monkeypatch, tmp_path, batches)

    assert batches.trials.count('NCT2') == 3
    assert stats['extracted'] == 2
    assert stats['failed'] == 1
    published = pl.read_parquet(_snapshot(tmp_path) / 'data.parquet')
    assert published['id'].to_list() == ['NCT1', 'NCT3']
    snapshot = {path.name: path.read_bytes() for path in _snapshot(tmp_path).iterdir()}
    submitted = len(batches.created)

    again = _stage(monkeypatch, tmp_path, batches, run_id='v1-run2')

    assert len(batches.created) == submitted
    assert again['computed'] == 0
    assert again['published'] is False
    assert {path.name: path.read_bytes() for path in _snapshot(tmp_path).iterdir()} == snapshot


def test_a_refusal_is_retried_within_the_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    batches = FakeBatches(refuse=frozenset({'NCT2'}), refusals=1)

    stats = _stage(monkeypatch, tmp_path, batches)

    assert sorted(batches.trials) == ['NCT1', 'NCT2', 'NCT2', 'NCT3']
    assert stats['extracted'] == 3
    assert stats['failed'] == 0


def test_refuses_without_a_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(WORKSPACE, raising=False)

    with pytest.raises(ComputeAborted, match=WORKSPACE):
        _stage(monkeypatch, tmp_path, None)

    assert not _state(tmp_path).exists()
    assert not (tmp_path / 'extraction_v2').exists()


def test_extracts_the_trials_trial_report_gives(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    batches = FakeBatches()

    stats = _stage(monkeypatch, tmp_path, batches, report_ids=['NCT3', 'NCT1'])

    assert sorted(batches.trials) == ['NCT1', 'NCT3']
    assert stats['trials'] == 2
    published = pl.read_parquet(_snapshot(tmp_path) / 'data.parquet')
    assert published['id'].to_list() == ['NCT1', 'NCT3']


def test_a_system_prompt_override_is_sent_and_tags_the_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prompt = tmp_path / 'prompt.txt'
    prompt.write_text('a different system prompt')
    batches = FakeBatches()

    _stage(monkeypatch, tmp_path, batches, system_prompt=str(prompt), sample_size=2)

    assert batches.created[0][0]['params']['system'][0]['text'] == 'a different system prompt'
    revision = extraction_v2_revision(
        'claude-haiku-5-5', 'a different system prompt', RENDER_VERSION, 'medium', EXTRACTION_V2_MAX_TOKENS
    )
    tag = EXTRACTION_V2.tag(revision, 2)
    metadata = json.loads((tmp_path / 'extraction_v2' / tag / 'metadata.json').read_text())
    assert metadata == EXTRACTION_V2.metadata(revision, 2)


def test_report_ids_match_context_ids_whatever_the_case(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    batches = FakeBatches()

    stats = _stage(monkeypatch, tmp_path, batches, report_ids=['NCT1', 'NCT2', 'NCT4'])

    assert sorted(batches.trials) == ['NCT1', 'NCT2']
    assert stats['trials'] == 3
    assert stats['prompts'] == 2
    published = pl.read_parquet(_snapshot(tmp_path) / 'data.parquet')
    assert published['id'].to_list() == ['NCT1', 'NCT2']


def test_systemic_failures_use_no_attempt_and_block_the_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    batches = FakeBatches(
        unanswered={'NCT1': SimpleNamespace(type='expired'), 'NCT2': _errored('overloaded_error')},
        refuse=frozenset({'NCT3'}),
    )

    with pytest.raises(TaskRunError, match='2 trials'):
        _stage(monkeypatch, tmp_path, batches)

    assert sorted(batches.trials) == ['NCT1', 'NCT2', 'NCT3', 'NCT3', 'NCT3']
    records = read_records(str(tmp_path / 'store' / 'extraction_v2'), _config(tmp_path), pl.String())
    assert records['status'].to_list() == ['failed'] * 3
    assert set(records['error']) == {'refusal'}
    assert not _snapshot(tmp_path).exists()

    again = FakeBatches()
    stats = _stage(monkeypatch, tmp_path, again, run_id='v1-run2')

    assert sorted(again.trials) == ['NCT1', 'NCT2']
    assert stats['extracted'] == 2
    assert stats['failed'] == 1
    assert stats['published'] is True


def test_an_invalid_request_uses_an_attempt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    batches = FakeBatches(unanswered={'NCT2': _errored('invalid_request_error')})

    stats = _stage(monkeypatch, tmp_path, batches)

    assert batches.trials.count('NCT2') == 3
    assert stats['failed'] == 1
    assert stats['published'] is True
    records = read_records(str(tmp_path / 'store' / 'extraction_v2'), _config(tmp_path), pl.String())
    assert records.filter(pl.col('status') == 'failed')['error'].to_list() == ['invalid_request_error'] * 3


def test_effort_is_sent_and_tags_the_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    batches = FakeBatches()

    _stage(monkeypatch, tmp_path, batches, effort='high')

    assert batches.created[0][0]['params']['output_config']['effort'] == 'high'
    package, filename = EXTRACTION_V2_PROMPT
    prompt = (resources.files(package) / filename).read_text(encoding='utf-8')
    revision = extraction_v2_revision('claude-haiku-5-5', prompt, RENDER_VERSION, 'high', EXTRACTION_V2_MAX_TOKENS)
    tag = EXTRACTION_V2.tag(revision, None)
    assert tag != EXTRACTION_V2.release_tag()
    metadata = json.loads((tmp_path / 'extraction_v2' / tag / 'metadata.json').read_text())
    assert metadata == EXTRACTION_V2.metadata(revision, None)
