"""open_search_load reads the JSON files under its datasets.yaml input_dir."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pos.tasks import open_search_load
from pos.tasks.open_search_load import OpenSearchLoad, OpenSearchLoadError, OpenSearchLoadSpec

DATASETS = Path(__file__).parents[1] / 'config' / 'datasets.yaml'


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
