"""Every dataset a pts task writes to `output/` is validated against its dataset model, by code.

For each such task in the config, its module must define `validate`, and `validate` must validate every
`output/` destination of the task against the model of that dataset. It is called here with the configured
destinations and a recording `validate_outputs`, so no data is needed. A new output without a model, or a
`validate` that forgets an output or names the wrong model, fails here.
"""

from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
import yaml

from pts.schemas.dataset import DatasetModel
from pts.tasks.pyspark import PYSPARK_PACKAGE, Pyspark
from pts.tasks.transform import TRANSFORMER_PACKAGE, Transform

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
def test_every_output_is_validated_against_its_dataset_model(task: dict[str, Any], monkeypatch) -> None:
    if 'transformer' in task:
        name = task['transformer']
        module = import_module(f'{TRANSFORMER_PACKAGE}.{name}')
        validate = Transform.load_validate(name)
    else:
        name = task['pyspark']
        module = import_module(f'{PYSPARK_PACKAGE}.{name}')
        validate = Pyspark.load_validate(name)
    assert validate is not None, (
        f'{task["name"]} writes {task["outputs"]}, so {module.__name__} must define validate() validating them '
        '(see pts.task_validation)'
    )

    validated: dict[str, type[DatasetModel]] = {}
    monkeypatch.setattr(module, 'validate_outputs', lambda outputs: validated.update(outputs))
    validate(task['source'], task['destination'], task.get('settings') or {}, None)

    for path in task['outputs']:
        dataset = path.removeprefix('output/')
        assert path in {p.rstrip('/') for p in validated}, f'{module.__name__}.validate() does not validate {path}'
        model = validated.get(path) or validated[f'{path}/']
        assert model.dataset_name == dataset, f'{module.__name__}.validate() validates {path} against {model.__name__}'
