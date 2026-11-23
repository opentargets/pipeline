"""Every task in config.yaml must have a registered type and a valid spec."""

import importlib
from pathlib import Path
from typing import Any

import pytest
import yaml
from otter.scratchpad.model import Scratchpad

from pps.core import task_packages
from pps.step import resolve_step
from pps.tasks.pyspark import PysparkSpec

CONFIG_PATH = Path(__file__).parent.parent / 'config.yaml'
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())
SCRATCHPAD = Scratchpad(CONFIG['scratchpad'])

TASKS = [
    pytest.param(task, id=f'{step_name}: {task["name"]}')
    for step_name, tasks in CONFIG['steps'].items()
    for task in tasks
]

PYSPARK_TASKS = [
    pytest.param(task, id=f'{step_name}: {task["name"]}')
    for step_name, tasks in CONFIG['steps'].items()
    for task in tasks
    if task['name'].startswith('pyspark ')
]


@pytest.mark.parametrize('task', PYSPARK_TASKS)
def test_pyspark_task_settings_are_valid(task: dict[str, Any]) -> None:
    spec = PysparkSpec(**SCRATCHPAD.replace_dict(task))
    _, settings_model = resolve_step(spec.step)
    settings_model.model_validate(spec.settings)


def _spec_class(task_type: str) -> type:
    for package in task_packages():
        try:
            module = importlib.import_module(f'{package}.{task_type}')
        except ModuleNotFoundError:
            continue
        return getattr(module, f'{task_type.title()}Spec')
    pytest.fail(f'no task type {task_type!r} in {task_packages()}')


@pytest.mark.parametrize('task', TASKS)
def test_task_spec_is_valid(task: dict[str, Any]) -> None:
    task_type = task['name'].split(' ')[0]
    _spec_class(task_type)(**SCRATCHPAD.replace_dict(task))
