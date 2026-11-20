"""open_search_create_index treats an existing index as success and raises on any other error."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from loguru import logger
from opensearchpy import RequestError

from pos.tasks import open_search_create_index as module
from pos.tasks.open_search_create_index import OpenSearchCreateIndex, OpenSearchCreateIndexSpec

ROOT = Path(__file__).parents[1]


class FakeIndices:
    def __init__(self, error: Exception | None) -> None:
        self.error = error
        self.created: list[str] = []
        self.bodies: list[dict] = []

    def create(self, index, body):
        if self.error:
            raise self.error
        self.created.append(index)
        self.bodies.append(body)


@pytest.fixture
def make_task(monkeypatch):
    def make(error: Exception | None):
        client = MagicMock(indices=FakeIndices(error))
        monkeypatch.setattr(module, 'connect', lambda host, port: client)
        spec = OpenSearchCreateIndexSpec(
            name='open_search_create_index search_disease',
            dataset='search_disease',
            prefix='platform2612',
            dataset_config_path=str(ROOT / 'config' / 'datasets.yaml'),
        )
        return OpenSearchCreateIndex(spec, MagicMock()), client

    monkeypatch.chdir(ROOT)
    return make


@pytest.fixture
def messages():
    seen: list[str] = []
    sink = logger.add(lambda m: seen.append(m.record['message']), level='DEBUG')
    yield seen
    logger.remove(sink)


def _run(task):
    # the undecorated run: otter's @report turns exceptions into a failed manifest entry
    return OpenSearchCreateIndex.run.__wrapped__(task)  # ty: ignore[unresolved-attribute]


def test_create_index_created(make_task, messages):
    task, client = make_task(None)
    _run(task)
    assert client.indices.created == ['platform2612_search_disease']
    assert any('created' in m for m in messages)
    (body,) = client.indices.bodies
    mappings = json.loads((ROOT / 'config' / 'opensearch' / 'search.json').read_text())
    assert body['settings']['index']['refresh_interval'] == -1
    assert body['mappings'] == mappings['mappings']


def test_create_index_exists_ok(make_task, messages):
    task, _ = make_task(RequestError(400, 'resource_already_exists_exception'))
    _run(task)
    assert any('exists' in m for m in messages)
    assert not any('created' in m for m in messages)


def test_create_index_other_error_raises(make_task):
    task, _ = make_task(RequestError(400, 'mapper_parsing_exception'))
    with pytest.raises(RequestError, match='mapper_parsing_exception'):
        _run(task)
