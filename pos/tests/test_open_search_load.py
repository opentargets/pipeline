"""open_search_load reads the JSON files under its datasets.yaml input_dir."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pos.services.opensearch import create_body, refresh_interval
from pos.tasks import open_search_load
from pos.tasks.open_search_load import OpenSearchLoad, OpenSearchLoadError, OpenSearchLoadSpec

ROOT = Path(__file__).parents[1]
DATASETS = ROOT / 'config' / 'datasets.yaml'


@pytest.mark.parametrize(
    ('dataset', 'folder'), [('search_disease', 'export/search_disease'), ('croissant', 'croissant')]
)
def test_json_files_come_from_input_dir(tmp_path, dataset, folder):
    data = tmp_path / 'release_data' / folder
    data.mkdir(parents=True)
    (data / 'part-0.json').write_text('{}\n')
    (data / 'part-0.json.crc').write_text('')
    spec = OpenSearchLoadSpec(
        name=f'open_search_load dataset {dataset}',
        dataset=dataset,
        json_parent='release_data',
        prefix='platform2612',
        dataset_config_path=str(DATASETS),
    )
    task = OpenSearchLoad(spec, MagicMock(config=MagicMock(work_path=tmp_path)))
    assert task._get_json_files() == [data / 'part-0.json']


def test_open_search_load_fails_on_no_json(tmp_path, monkeypatch):
    (tmp_path / 'release_data' / 'export' / 'search_disease').mkdir(parents=True)
    monkeypatch.setattr(open_search_load, 'connect', lambda host, port: MagicMock())
    spec = OpenSearchLoadSpec(
        name='open_search_load dataset search_disease',
        dataset='search_disease',
        json_parent='release_data',
        prefix='platform2612',
        dataset_config_path=str(DATASETS),
    )
    task = OpenSearchLoad(spec, MagicMock(config=MagicMock(work_path=tmp_path)))
    with pytest.raises(OpenSearchLoadError, match=r'no \.json files'):
        OpenSearchLoad.run.__wrapped__(task)  # ty: ignore[unresolved-attribute]


def test_open_search_load_raises_on_bulk_failures(tmp_path, monkeypatch):
    data = tmp_path / 'release_data' / 'export' / 'search_disease'
    data.mkdir(parents=True)
    (data / 'part-0.json').write_text('{"id": "a"}\n{"id": "b"}\n{"id": "c"}\n')
    client = MagicMock()
    monkeypatch.setattr(open_search_load, 'connect', lambda host, port: client)
    results = [(True, {}), (False, {'index': {'error': 'x'}}), (False, {'index': {'error': 'y'}})]
    monkeypatch.setattr(open_search_load.helpers, 'parallel_bulk', lambda *a, **kw: iter(results))
    spec = OpenSearchLoadSpec(
        name='open_search_load dataset search_disease',
        dataset='search_disease',
        json_parent='release_data',
        prefix='platform2612',
        dataset_config_path=str(DATASETS),
    )
    task = OpenSearchLoad(spec, MagicMock(config=MagicMock(work_path=tmp_path)))
    with pytest.raises(OpenSearchLoadError, match='2 documents failed to index into platform2612_search_disease'):
        OpenSearchLoad.run.__wrapped__(task)  # ty: ignore[unresolved-attribute]


def test_load_restores_refresh_interval_then_refreshes(tmp_path, monkeypatch):
    data = tmp_path / 'release_data' / 'export' / 'search_disease'
    data.mkdir(parents=True)
    (data / 'part-0.json').write_text('{"id": "a"}\n')
    client = MagicMock()
    monkeypatch.setattr(open_search_load, 'connect', lambda host, port: client)
    monkeypatch.setattr(open_search_load.helpers, 'parallel_bulk', lambda *a, **kw: iter([(True, {})]))
    monkeypatch.chdir(ROOT)
    spec = OpenSearchLoadSpec(
        name='open_search_load dataset search_disease',
        dataset='search_disease',
        json_parent='release_data',
        prefix='platform2612',
        dataset_config_path=str(DATASETS),
    )
    task = OpenSearchLoad(spec, MagicMock(config=MagicMock(work_path=tmp_path)))
    OpenSearchLoad.run.__wrapped__(task)  # ty: ignore[unresolved-attribute]
    assert [c[0] for c in client.indices.method_calls] == ['put_settings', 'refresh']
    put, refresh = client.indices.method_calls
    # search.json sets no refresh_interval: null resets it to the server default
    assert put.kwargs == {'index': 'platform2612_search_disease', 'body': {'index': {'refresh_interval': None}}}
    assert refresh.kwargs == {'index': 'platform2612_search_disease'}


@pytest.mark.parametrize(
    ('settings', 'expected'),
    [
        ({}, None),
        ({'index': {'number_of_shards': 1}}, None),
        ({'index': {'refresh_interval': '30s'}}, '30s'),
        ({'index.refresh_interval': '5s'}, '5s'),
        ({'refresh_interval': '1s'}, '1s'),
    ],
)
def test_refresh_interval_from_mapping_file(tmp_path, settings, expected):
    mappings = tmp_path / 'm.json'
    mappings.write_text(json.dumps({'settings': settings, 'mappings': {}}))
    assert refresh_interval(mappings) == expected
    body = create_body(mappings)
    assert body['settings']['index']['refresh_interval'] == -1
    assert 'index.refresh_interval' not in body['settings']
    assert 'refresh_interval' not in body['settings']


@pytest.mark.parametrize('mappings', sorted((ROOT / 'config' / 'opensearch').glob('*.json')), ids=lambda p: p.name)
def test_mapping_files_leave_refresh_to_the_server(mappings):
    # create_body turns refresh off for the load; the loaded index gets the default back
    assert refresh_interval(mappings) is None
