"""Every pyspark task in config.yaml must point at a real step with valid settings."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from otter.scratchpad.model import Scratchpad

from pps.step import resolve_step
from pps.tasks.pyspark import PysparkSpec

CONFIG_PATH = Path(__file__).parent.parent / 'config.yaml'
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())
SCRATCHPAD = Scratchpad(CONFIG['scratchpad'])

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
