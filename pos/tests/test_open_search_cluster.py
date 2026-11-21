"""The step refuses a used cluster, turns query insights off before any index and leaves no top_queries-* index."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from pos.services import opensearch as service
from pos.tasks import open_search_finish, open_search_prepare
from pos.tasks.open_search_finish import OpenSearchFinish, OpenSearchFinishError, OpenSearchFinishSpec
from pos.tasks.open_search_prepare import OpenSearchPrepare, OpenSearchPrepareError, OpenSearchPrepareSpec

ROOT = Path(__file__).parents[1]
BACKEND = yaml.safe_load((ROOT / 'config' / 'config.yaml').read_text())['steps']['backend_opensearch']


class FakeCluster:
    def __init__(self) -> None:
        self.settings: dict = {}

    def put_settings(self, body):
        self.settings.update(body['persistent'])


class FakeIndices:
    def __init__(self, names: list[str], undeletable: bool = False) -> None:
        self.names = set(names)
        self.undeletable = undeletable
        self.deleted: list[str] = []

    def get(self, index, expand_wildcards, ignore_unavailable):
        assert (index, expand_wildcards, ignore_unavailable) == ('top_queries-*', 'all', True)
        return {n: {} for n in self.names if n.startswith('top_queries-')}

    def delete(self, index, expand_wildcards):
        self.deleted.append(index)
        if not self.undeletable:
            self.names -= set(index.split(','))

    def exists(self, index):
        return index in self.names


def _client(names: list[str] | None = None, undeletable: bool = False) -> MagicMock:
    return MagicMock(cluster=FakeCluster(), indices=FakeIndices(names or [], undeletable))


def _run(cls, spec_cls, client, monkeypatch, module, **fields):
    monkeypatch.setattr(module, 'connect', lambda host, port: client)
    task = cls(spec_cls(name=f'{module.__name__.rsplit(".", 1)[1]} x', **fields), MagicMock())
    return cls.run.__wrapped__(task)


def _prepare(client, monkeypatch):
    fields = {'prefix': 'platform2612', 'dataset_config_path': str(ROOT / 'config' / 'datasets.yaml')}
    return _run(OpenSearchPrepare, OpenSearchPrepareSpec, client, monkeypatch, open_search_prepare, **fields)


def _names(tasks: list[dict]) -> list[str]:
    return [n for t in tasks for n in [t['name'], *_names(t.get('do', []))]]


def test_setting_names():
    assert set(service.QUERY_INSIGHTS_OFF) == {
        f'search.insights.top_queries.{metric}.enabled' for metric in ('latency', 'cpu', 'memory')
    }
    assert not any(service.QUERY_INSIGHTS_OFF.values())


def test_prepare_turns_query_insights_off(monkeypatch):
    client = _client(['platform2512_search_disease', 'top_queries-2026.11.20-00001'])
    _prepare(client, monkeypatch)
    assert client.cluster.settings == service.QUERY_INSIGHTS_OFF


@pytest.mark.parametrize('index', ['platform2612_downloads', 'platform2612_search_variant'])
def test_prepare_refuses_a_used_cluster(monkeypatch, index):
    client = _client([index])
    with pytest.raises(OpenSearchPrepareError, match=rf'{index}\. .*wipe that server.s data disk'):
        _prepare(client, monkeypatch)
    assert client.cluster.settings == {}


def test_prepare_checks_every_index_in_datasets_yaml(monkeypatch):
    datasets = yaml.safe_load((ROOT / 'config' / 'datasets.yaml').read_text())['opensearch']
    task = OpenSearchPrepare(
        OpenSearchPrepareSpec(
            name='open_search_prepare x',
            prefix='platform2612',
            dataset_config_path=str(ROOT / 'config' / 'datasets.yaml'),
        ),
        MagicMock(),
    )
    assert task._indices == sorted(f'platform2612_{d["index"]}' for d in datasets.values())
    (explode,) = [t for t in BACKEND if t['name'] == 'explode load all opensearch datasets']
    assert set(explode['foreach']) <= set(datasets)


def test_finish_without_top_queries(monkeypatch):
    client = _client(['platform2612_search_disease'])
    _run(OpenSearchFinish, OpenSearchFinishSpec, client, monkeypatch, open_search_finish)
    assert client.indices.deleted == []


def test_finish_deletes_top_queries(monkeypatch):
    client = _client(['top_queries-2026.11.20-00001', 'top_queries-2026.11.21-00002', 'platform2612_search_disease'])
    _run(OpenSearchFinish, OpenSearchFinishSpec, client, monkeypatch, open_search_finish)
    assert client.indices.deleted == ['top_queries-2026.11.20-00001,top_queries-2026.11.21-00002']
    assert client.indices.names == {'platform2612_search_disease'}


def test_finish_fails_if_top_queries_remain(monkeypatch):
    client = _client(['top_queries-2026.11.20-00001'], undeletable=True)
    with pytest.raises(OpenSearchFinishError, match=r'top_queries-2026\.11\.20-00001'):
        _run(OpenSearchFinish, OpenSearchFinishSpec, client, monkeypatch, open_search_finish)


def test_prepare_runs_before_every_index_and_finish_after_every_load():
    (explode,) = [t for t in BACKEND if t['name'] == 'explode load all opensearch datasets']
    (prepare,) = [t for t in BACKEND if t['name'].startswith('open_search_prepare ')]
    (finish,) = [t for t in BACKEND if t['name'].startswith('open_search_finish ')]
    assert prepare['name'] in explode['requires']
    assert not prepare.get('requires')
    assert finish['requires'] == [explode['name']]
    # every create and load is an explode child, so the explode completes only after them
    types = ('open_search_create_index', 'open_search_load')
    creates_and_loads = [n for n in _names(BACKEND) if n.split(' ')[0] in types]
    assert creates_and_loads
    assert set(creates_and_loads) <= set(_names(explode['do']))
