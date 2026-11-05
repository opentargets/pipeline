import json
from types import SimpleNamespace

from pydantic import BaseModel, ConfigDict

from pts.transformers.clinical_mira.moa.claude import custom_id, run_batch_cached


class Echo(BaseModel):
    model_config = ConfigDict(extra='forbid')

    value: str


class FakeBatches:
    """Message Batches stand-in: each request echoes its user message, except ids in ``fail``."""

    def __init__(self, fail: frozenset[str] = frozenset(), refuse: frozenset[str] = frozenset()) -> None:
        self.created: list[list[dict]] = []
        self.retrieved = 0
        self.fail = fail
        self.refuse = refuse

    def create(self, *, requests: list[dict]) -> SimpleNamespace:
        self.created.append(requests)
        return SimpleNamespace(id=f'batch_{len(self.created)}', processing_status='in_progress')

    def retrieve(self, batch_id: str) -> SimpleNamespace:
        self.retrieved += 1
        status = 'ended' if self.retrieved % 2 == 0 else 'in_progress'
        return SimpleNamespace(id=batch_id, processing_status=status, request_counts=SimpleNamespace(processing=1))

    def results(self, batch_id: str):
        requests = self.created[int(batch_id.split('_')[1]) - 1]
        for request in requests:
            cid = request['custom_id']
            if cid in self.fail:
                yield SimpleNamespace(
                    custom_id=cid, result=SimpleNamespace(type='errored', error=SimpleNamespace(type='api_error'))
                )
                continue
            text = request['params']['messages'][0]['content']
            stop = 'refusal' if cid in self.refuse else 'end_turn'
            message = SimpleNamespace(
                stop_reason=stop,
                content=[SimpleNamespace(type='text', text=json.dumps({'value': text}))],
                usage=SimpleNamespace(
                    input_tokens=10, output_tokens=5, cache_read_input_tokens=100, cache_creation_input_tokens=0
                ),
            )
            yield SimpleNamespace(custom_id=cid, result=SimpleNamespace(type='succeeded', message=message))


def _client(batches: FakeBatches) -> SimpleNamespace:
    return SimpleNamespace(messages=SimpleNamespace(batches=batches))


def _run(batches: FakeBatches, tmp_path, requests=(('a', 'x'), ('b:1', 'y')), **kwargs):
    return run_batch_cached(
        list(requests),
        system_prompt='be precise',
        response_model=Echo,
        model='claude-sonnet-5-5',
        cache_path=tmp_path / 'c.jsonl',
        state_path=tmp_path / 'batches.json',
        client=_client(batches),
        sleep=lambda seconds: None,
        **kwargs,
    )


def test_custom_ids_are_api_safe_and_unique():
    assert custom_id('p1:3') != custom_id('p1_3')
    assert all(c.isalnum() or c in '_-' for c in custom_id('pool/with:odd chars'))
    assert len(custom_id('x' * 200)) <= 64


def test_batch_request_carries_schema_cached_system_and_effort(tmp_path):
    batches = FakeBatches()

    results, errors = _run(batches, tmp_path, effort='low')

    params = batches.created[0][0]['params']
    assert params['model'] == 'claude-sonnet-5-5'
    assert params['system'][0]['cache_control'] == {'type': 'ephemeral'}
    assert params['output_config']['format']['type'] == 'json_schema'
    assert params['output_config']['effort'] == 'low'
    assert results['a'] == Echo(value='x') and results['b:1'] == Echo(value='y')
    assert errors == {}


def test_second_run_is_all_cache_hits(tmp_path):
    batches = FakeBatches()
    _run(batches, tmp_path)

    _run(batches, tmp_path)

    assert len(batches.created) == 1


def test_errors_and_refusals_are_reported_not_cached(tmp_path):
    first = FakeBatches(fail=frozenset({custom_id('a')}), refuse=frozenset({custom_id('b:1')}))
    results, errors = _run(first, tmp_path)
    second = FakeBatches()
    again, _ = _run(second, tmp_path)

    assert results == {}
    assert errors == {'a': 'api_error', 'b:1': 'refusal'}
    assert set(again) == {'a', 'b:1'}
    assert len(second.created[0]) == 2


def test_requests_are_split_across_batches(tmp_path):
    batches = FakeBatches()

    results, _ = _run(batches, tmp_path, requests=[(f'r{i}', f't{i}') for i in range(5)], max_batch_requests=2)

    assert [len(b) for b in batches.created] == [2, 2, 1]
    assert len(results) == 5


def test_a_submitted_batch_is_resumed_not_resubmitted(tmp_path):
    batches = FakeBatches()
    batches.create(requests=[{'custom_id': custom_id('a'), 'params': {'messages': [{'content': 'x'}]}}])
    (tmp_path / 'batches.json').write_text(json.dumps({'pending': ['batch_1']}))

    results, _ = _run(batches, tmp_path, requests=[('a', 'x')])

    assert len(batches.created) == 1
    assert results['a'] == Echo(value='x')
    assert json.loads((tmp_path / 'batches.json').read_text())['pending'] == []


def test_token_usage_is_recorded_per_request(tmp_path):
    _run(FakeBatches(), tmp_path)

    rows = [json.loads(line) for line in (tmp_path / 'c.jsonl').read_text().splitlines()]

    assert rows[0]['usage'] == {
        'input_tokens': 10,
        'output_tokens': 5,
        'cache_read_input_tokens': 100,
        'cache_creation_input_tokens': 0,
    }
