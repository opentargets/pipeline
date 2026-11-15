import json

from pydantic import BaseModel, ConfigDict

from pts.transformers.clinical_mira.moa.llm import cached_results, result_key


class Echo(BaseModel):
    model_config = ConfigDict(extra='forbid')

    value: str


def _cache(path, *texts: str, system_prompt: str = 'i', model: str = 'm') -> None:
    rows = [{'key': result_key(t, system_prompt, Echo, model), 'response': {'value': t}} for t in texts]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def _cached(path, system_prompt: str = 'i'):
    return cached_results(
        [('a', 'x'), ('c', 'new')], system_prompt=system_prompt, response_model=Echo, model='m', cache_path=path
    )


def test_cached_results_returns_only_cached_requests(tmp_path):
    _cache(tmp_path / 'c.jsonl', 'x', 'y')

    results, errors = _cached(tmp_path / 'c.jsonl')

    assert results == {'a': Echo(value='x')}
    assert errors == {}


def test_changed_system_prompt_misses_the_cache(tmp_path):
    _cache(tmp_path / 'c.jsonl', 'x')

    assert _cached(tmp_path / 'c.jsonl', system_prompt='other') == ({}, {})


def test_missing_cache_returns_nothing(tmp_path):
    assert _cached(tmp_path / 'absent.jsonl') == ({}, {})
