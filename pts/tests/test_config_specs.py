"""Every task in the shipped ``config.yaml`` must build a valid otter spec.

Otter validates a task's config *before* any of our code runs: ``Spec`` requires
``name`` to be ``<task_type> <description>``, and the first word has to name a
task type the registry knows. A task that fails either check kills the whole step
at config load, and nothing else in this suite exercises that layer -- the other
tests call transformer functions directly, with the config file never parsed.

So this test runs the same three things the runner does, for every task of every
step: build the base ``Spec``, look the ``task_type`` up in a registry holding the
built-in and the pts tasks, and rebuild the spec with its concrete class after
scratchpad replacement. It is deliberately mechanical -- it makes a whole class of
"the step dies before it starts" failure visible at test time.
"""

from importlib.util import find_spec
from pathlib import Path
from typing import Any

import pytest
from otter.config.model import Config
from otter.config.yaml import parse_yaml
from otter.scratchpad.model import Scratchpad
from otter.task.model import Spec
from otter.task.task_registry import TaskRegistry

CONFIG_PATH = Path(__file__).parents[1] / 'config.yaml'
"""The config file that ships in the image, not a fixture."""


def _config_dict() -> dict[str, Any]:
    return parse_yaml(CONFIG_PATH)


def _registry() -> TaskRegistry:
    """Build the registry the way `pts.core.main` does."""
    config_dict = _config_dict()
    scratchpad = Scratchpad(sentinel_dict=config_dict.get('scratchpad', {}))
    config = Config(step=next(iter(config_dict['steps'])), steps=list(config_dict['steps']))
    registry = TaskRegistry(config, scratchpad)
    registry.register('otter.tasks')
    registry.register('pts.tasks')
    return registry


DISPATCH_KNOWN_BROKEN = {
    # pre-existing on main: `vep_view` is a polars transformer
    # (`pts/src/pts/transformers/vep_view.py`) but the step declares it as a pyspark
    # job, so the step would die with a ModuleNotFoundError. It is unreachable today
    # -- no DAG step references `pts_vep_view` -- and fixing it flips the step from
    # dataproc to GCE, which is a routing change this test has no business making.
    ('vep_view', 'pyspark generate vep view'),
}
"""Tasks whose ``transformer``/``pyspark`` module is known not to resolve.

Each entry is an ``xfail(strict=True)``, so fixing one turns into a test failure
here and the entry has to be removed -- the list cannot rot into an allowlist.
"""


def _tasks() -> list[tuple[str, dict[str, Any]]]:
    """Every task in the config, tagged with the step it belongs to."""
    return [(step_name, task) for step_name, tasks in _config_dict()['steps'].items() for task in tasks or []]


def _task_ids() -> list[str]:
    return [f'{step}:{task.get("name", "<unnamed>")}' for step, task in _tasks()]


def _dispatch_params() -> list[Any]:
    """`_tasks`, with the known-broken dispatch targets marked xfail."""
    params = []
    for step_name, task in _tasks():
        marks = (
            [pytest.mark.xfail(reason='pre-existing: module does not exist', strict=True)]
            if (step_name, task.get('name')) in DISPATCH_KNOWN_BROKEN
            else []
        )
        params.append(pytest.param(step_name, task, marks=marks))
    return params


@pytest.mark.parametrize(('step_name', 'task'), _tasks(), ids=_task_ids())
def test_every_task_in_the_config_builds_a_valid_spec(step_name: str, task: dict[str, Any]) -> None:
    """A task must pass the same validation the runner puts it through."""
    registry = _registry()
    config_dict = _config_dict()
    scratchpad = Scratchpad(sentinel_dict=config_dict.get('scratchpad', {}))

    # `Spec` rejects a name that is not `<task_type> <description>`
    spec = Spec(**task)

    # the first word has to name a registered task type
    assert spec.task_type in registry._specs, (
        f'step {step_name}: task {spec.name!r} has task type {spec.task_type!r}, which is not registered. '
        f'A task name must be `<task_type> <description>`.'
    )

    # and the concrete spec class has to accept the task's fields, both before
    # and after the scratchpad placeholders are replaced
    spec_class = registry._specs[spec.task_type]
    typed = spec_class(**spec.model_dump())
    spec_class(**scratchpad.replace_dict(typed.model_dump(), ignore_missing=typed.scratchpad_ignore_missing))


@pytest.mark.parametrize(('step_name', 'task'), _dispatch_params(), ids=_task_ids())
def test_every_transformer_and_pyspark_module_named_in_the_config_exists(step_name: str, task: dict[str, Any]) -> None:
    """The module a task dispatches to is resolved by name at runtime, so check it is there."""
    for field, package in (('transformer', 'pts.transformers'), ('pyspark', 'pts.pyspark')):
        name = task.get(field)
        if name is None:
            continue
        assert find_spec(f'{package}.{name}') is not None, (
            f'step {step_name}: task {task["name"]!r} names {field} {name!r}, but {package}.{name} does not exist'
        )


def test_task_names_are_unique_within_a_step() -> None:
    """`otter.task.load_specs` exits on a duplicate name, and `requires` keys on it."""
    for step_name, tasks in _config_dict()['steps'].items():
        names = [task['name'] for task in tasks or []]
        assert len(names) == len(set(names)), f'step {step_name} has duplicate task names: {names}'


def test_every_requires_entry_names_a_task_in_the_same_step() -> None:
    """`otter.task.load_specs` exits on a dangling dependency."""
    for step_name, tasks in _config_dict()['steps'].items():
        names = {task['name'] for task in tasks or []}
        for task in tasks or []:
            for dependency in task.get('requires', []):
                assert dependency in names, (
                    f'step {step_name}: task {task["name"]!r} requires {dependency!r}, which is not in the step'
                )


def test_ldsc_cts_steps_are_generic_and_registry_driven() -> None:
    """LDSC-CTS scales through settings rather than GWAS-specific tasks."""
    config = _config_dict()['steps']
    annotations = config['ldsc_cts_annotations']
    regressions = config['ldsc_cts_regression']
    assert len(annotations) == 1
    assert len(regressions) == 1
    forbidden = ('GCST', 'EUR', 'EAS', 'tabula_sapiens', 'gtex')
    for task in (*annotations, *regressions):
        assert not any(token in task['name'] for token in forbidden)
    annotation_settings = annotations[0]['settings']
    assert {dataset['id'] for dataset in annotation_settings['datasets']} == {
        'tabula_sapiens-celltype',
        'tabula_sapiens-tissue',
        'gtex-tissue',
    }
    assert regressions[0]['settings']['study_batch_size'] == 8
