import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from pts.transformers.clinical_mira.moa.llm import run_cached


class Echo(BaseModel):
    model_config = ConfigDict(extra='forbid')

    value: str


class FakeClient:
    """Answers each input with ``Echo(value=input)``, failing where ``fail`` says so."""

    def __init__(self, fail: Callable[[str, int], BaseException | None] = lambda text, n: None) -> None:
        self.calls: Counter[str] = Counter()
        self.fail = fail
        self.responses = SimpleNamespace(create=self._create)

    @property
    def total(self) -> int:
        return sum(self.calls.values())

    async def _create(self, *, input: str, **_: object) -> SimpleNamespace:
        self.calls[input] += 1
        error = self.fail(input, self.total)
        if error is not None:
            raise error
        return SimpleNamespace(output_text=json.dumps({'value': input}))

    async def close(self) -> None:
        return None


def _run(client: FakeClient, cache: Path, system_prompt: str = 'i', chunk_size: int = 100):
    return run_cached(
        [('a', 'x'), ('b', 'y')],
        system_prompt=system_prompt,
        response_model=Echo,
        model='m',
        cache_path=cache,
        chunk_size=chunk_size,
        client=client,
    )


def test_second_run_is_all_cache_hits(tmp_path):
    client = FakeClient()
    _run(client, tmp_path / 'c.jsonl')

    results, errors = _run(client, tmp_path / 'c.jsonl')

    assert client.total == 2
    assert results['b'].value == 'y'
    assert errors == {}


def test_changed_system_prompt_misses_the_cache(tmp_path):
    client = FakeClient()
    _run(client, tmp_path / 'c.jsonl')
    _run(client, tmp_path / 'c.jsonl', system_prompt='j')

    assert client.total == 4


def test_failures_are_returned_and_retried_on_rerun(tmp_path):
    attempts: Counter[str] = Counter()

    def fail_y_once(text: str, _: int) -> BaseException | None:
        attempts[text] += 1
        return ValueError('boom') if text == 'y' and attempts[text] == 1 else None

    client = FakeClient(fail_y_once)
    first, first_errors = _run(client, tmp_path / 'c.jsonl')
    second, second_errors = _run(client, tmp_path / 'c.jsonl')

    assert set(first) == {'a'}
    assert first_errors == {'b': 'ValueError'}
    assert set(second) == {'a', 'b'}
    assert second_errors == {}
    assert client.calls == Counter({'x': 1, 'y': 2})


def test_finished_chunks_survive_an_interruption(tmp_path):
    client = FakeClient(lambda text, n: KeyboardInterrupt() if n == 2 else None)

    with pytest.raises(KeyboardInterrupt):
        _run(client, tmp_path / 'c.jsonl', chunk_size=1)

    assert len((tmp_path / 'c.jsonl').read_text().splitlines()) == 1


def test_torn_last_line_does_not_block_a_rerun(tmp_path):
    client = FakeClient()
    _run(client, tmp_path / 'c.jsonl')
    with (tmp_path / 'c.jsonl').open('a') as cache:
        cache.write('{"key": "abc", "resp')

    results, _ = _run(client, tmp_path / 'c.jsonl')

    assert set(results) == {'a', 'b'}
    assert client.total == 2


def test_default_chunk_is_the_concurrency(tmp_path):
    client = FakeClient(lambda text, n: KeyboardInterrupt() if n == 2 else None)

    with pytest.raises(KeyboardInterrupt):
        run_cached(
            [('a', 'x'), ('b', 'y')],
            system_prompt='i',
            response_model=Echo,
            model='m',
            cache_path=tmp_path / 'c.jsonl',
            concurrency=1,
            client=client,
        )

    assert len((tmp_path / 'c.jsonl').read_text().splitlines()) == 1


def test_cached_only_never_calls_the_model(tmp_path):
    client = FakeClient()
    _run(client, tmp_path / 'c.jsonl')

    results, errors = run_cached(
        [('a', 'x'), ('c', 'new')],
        system_prompt='i',
        response_model=Echo,
        model='m',
        cache_path=tmp_path / 'c.jsonl',
        client=client,
        cached_only=True,
    )

    assert set(results) == {'a'}
    assert errors == {}
    assert client.total == 2


class FakeChatClient:
    """Chat-completions client answering ``Echo(value=<last user message>)``."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[dict] = []
        self.fail_on = fail_on
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        text = kwargs['messages'][-1]['content']
        if text == self.fail_on:
            raise ValueError('boom')
        message = SimpleNamespace(content=json.dumps({'value': text}))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    async def close(self) -> None:
        return None


def test_chat_api_sends_system_prompt_and_strict_schema(tmp_path):
    client = FakeChatClient()

    results, errors = run_cached(
        [('a', 'x')],
        system_prompt='be precise',
        response_model=Echo,
        model='openai/gpt-oss-120b-maas',
        cache_path=tmp_path / 'c.jsonl',
        api='chat',
        client=client,
    )

    assert results['a'] == Echo(value='x')
    assert errors == {}
    call = client.calls[0]
    assert call['messages'][0] == {'role': 'system', 'content': 'be precise'}
    assert call['response_format']['type'] == 'json_schema'
    assert call['response_format']['json_schema']['strict'] is True
    assert call['response_format']['json_schema']['schema']['additionalProperties'] is False


def test_chat_api_failures_are_reported_and_retried(tmp_path):
    first, first_errors = run_cached(
        [('a', 'x'), ('b', 'y')],
        system_prompt='i',
        response_model=Echo,
        model='m',
        cache_path=tmp_path / 'c.jsonl',
        api='chat',
        client=FakeChatClient(fail_on='y'),
    )
    second_client = FakeChatClient()
    second, _ = run_cached(
        [('a', 'x'), ('b', 'y')],
        system_prompt='i',
        response_model=Echo,
        model='m',
        cache_path=tmp_path / 'c.jsonl',
        api='chat',
        client=second_client,
    )

    assert set(first) == {'a'} and first_errors == {'b': 'ValueError'}
    assert set(second) == {'a', 'b'} and len(second_client.calls) == 1


def test_client_factory_is_called_once_per_chunk(tmp_path):
    made: list[FakeChatClient] = []

    def factory() -> FakeChatClient:
        made.append(FakeChatClient())
        return made[-1]

    run_cached(
        [('a', 'x'), ('b', 'y'), ('c', 'z')],
        system_prompt='i',
        response_model=Echo,
        model='m',
        cache_path=tmp_path / 'c.jsonl',
        api='chat',
        chunk_size=2,
        client_factory=factory,
    )

    assert len(made) == 2
