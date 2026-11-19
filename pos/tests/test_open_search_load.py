"""open_search_load reads the JSON files under its datasets.yaml input_dir."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pos.tasks.open_search_load import OpenSearchLoad, OpenSearchLoadSpec

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
