"""Smoke tests: the CLI loads, tasks register and the configs name registered tasks."""

import importlib
import pkgutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from click.testing import CliRunner
from otter.task.task_registry import TaskRegistry

import pos.tasks as pos_tasks
from pos.core import pos as pos_cli

ROOT = Path(__file__).parents[1]
CONFIGS = [ROOT / 'config' / 'config.yaml', ROOT / 'config' / 'config_ppp.yaml']


def _registered() -> set[str]:
    # the registration `core.pos_runner` does through `Runner.register_tasks`, without a Runner
    registry = TaskRegistry(MagicMock(), MagicMock())
    registry.register('otter.tasks')
    registry.register('pos.tasks')
    return set(registry._tasks)


def _task_names(tasks: list[dict]) -> list[str]:
    names = []
    for task in tasks:
        names.append(task['name'])
        names.extend(_task_names(task.get('do', [])))
    return names


def _config_task_types(config: Path) -> set[str]:
    steps = yaml.safe_load(config.read_text())['steps']
    return {name.split(' ')[0] for tasks in steps.values() for name in _task_names(tasks)}


def test_cli_help():
    result = CliRunner().invoke(pos_cli, ['--help'])
    assert result.exit_code == 0
    for command in ('local', 'remote', 'ftp-sync'):
        assert command in result.output


def test_tasks_register():
    for module in pkgutil.iter_modules(pos_tasks.__path__):
        importlib.import_module(f'pos.tasks.{module.name}')
    registered = _registered()
    assert {'sync_bucket', 'open_search_load', 'clickhouse_load', 'ot_croissant', 'ftp_sync'} <= registered
    used = set().union(*(_config_task_types(config) for config in CONFIGS))
    assert used <= registered


@pytest.mark.parametrize('config', CONFIGS, ids=lambda c: c.name)
def test_configs_parse(config):
    types = _config_task_types(config)
    assert types
    assert types <= _registered()
