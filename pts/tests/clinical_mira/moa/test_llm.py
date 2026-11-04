import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

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
