"""Every dataset a pts task writes to `output/` has its model declared, in the config as configured.

For every task in the config, without data: each `output/` destination is declared with `@writes`, and each
declared model is the model of the dataset it validates.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from pts.tasks.pyspark import Pyspark
from pts.tasks.transform import Transform

CONFIG = Path(__file__).parents[1] / 'config.yaml'


def _tasks_writing_outputs() -> list[dict[str, Any]]:
    steps: dict[str, list[dict[str, Any]]] = yaml.safe_load(CONFIG.read_text())['steps']
    found = []
    for step, tasks in steps.items():
        for task in tasks or []:
            if 'transformer' not in task and 'pyspark' not in task:
                continue
            destination = task.get('destination')
            paths = destination.values() if isinstance(destination, dict) else [destination]
            outputs = [p.rstrip('/') for p in paths if isinstance(p, str) and p.startswith('output/')]
            if outputs:
                found.append({'step': step, **task, 'outputs': outputs})
    return found


TASKS = _tasks_writing_outputs()


def test_the_config_writes_outputs() -> None:
    assert len(TASKS) > 50


@pytest.mark.parametrize('task', TASKS, ids=[f'{t["step"]}:{t["name"]}' for t in TASKS])
def test_every_output_is_declared_with_its_dataset_model(task: dict[str, Any]) -> None:
    if 'transformer' in task:
        validation = Transform.load_output_validation(task['transformer'])
    else:
        validation = Pyspark.load_output_validation(task['pyspark'])
    destination = task['destination']

    undeclared = [path for path in task['outputs'] if path not in _declared_paths(validation.outputs, destination)]
    assert not undeclared, f'{validation.name} must declare the model of {undeclared} with @writes'
    assert not validation.missing(destination), f'{validation.name} declares outputs {task["name"]} does not write'
    for key, model in validation.outputs.items():
        path = (destination[key] if key else destination).rstrip('/')
        if path.startswith('output/'):
            assert model.dataset_name == path.removeprefix('output/'), f'{validation.name} validates {path} as {model}'


def _declared_paths(outputs: dict, destination: str | dict[str, str]) -> set[str]:
    if isinstance(destination, str):
        return {destination.rstrip('/')} if None in outputs else set()
    return {destination[key].rstrip('/') for key in outputs if key in destination}
