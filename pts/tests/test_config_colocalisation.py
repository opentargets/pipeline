"""The colocalisation step runs on Dataproc and carries the gentropy priors."""

from pathlib import Path
from typing import Any

from otter.config.yaml import parse_yaml

CONFIG_PATH = Path(__file__).parents[1] / 'config.yaml'
"""The config file that ships in the image, not a fixture."""


def _task() -> dict[str, Any]:
    [task] = parse_yaml(CONFIG_PATH)['steps']['colocalisation']
    return task


def test_the_step_is_one_pyspark_task() -> None:
    task = _task()
    assert task['name'].startswith('pyspark ')
    assert task['pyspark'] == 'colocalisation'


def test_sources_and_destination() -> None:
    task = _task()
    assert task['source'] == {
        'credible_set': 'output/credible_set',
        'credible_set_variant': 'output/credible_set_variant',
    }
    assert task['destination'] == {'colocalisation': 'output/colocalisation'}


def test_priors_and_partitions() -> None:
    assert _task()['settings'] == {
        'priorc1': 1e-4,
        'priorc2': 1e-4,
        'priorc12': 1e-5,
        'output_partitions': 200,
    }


def test_executor_sizing() -> None:
    assert _task()['properties'] == {
        'spark.executor.memory': '22g',
        'spark.executor.memoryOverhead': '4g',
        'spark.executor.cores': '4',
        'spark.memory.fraction': '0.8',
        'spark.memory.storageFraction': '0.2',
    }
