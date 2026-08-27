"""Tests for the per-step PTS machine_type override."""

import pytest

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig
from orchestration.models.pts_step import pts_step_from_config


@pytest.fixture(scope='module')
def config() -> UnifiedPipelineConfig:
    """The real pipeline config, so the tests read the shipped yaml."""
    return UnifiedPipelineConfig()


def test_step_without_an_override_uses_the_default(config: UnifiedPipelineConfig) -> None:
    """Most steps must keep the shared machine type."""
    step = pts_step_from_config('pts_target', config)
    assert step.machine_type == config.pts_machine_type


def test_openfda_declares_its_own_machine(config: UnifiedPipelineConfig) -> None:
    """openfda's tasks need more memory per core than the default machine has."""
    step = pts_step_from_config('pts_openfda_input_preparation', config)
    assert step.machine_type == 'n1-highmem-32'
    assert step.machine_type != config.pts_machine_type


def test_declared_machine_types_look_like_machine_types(config: UnifiedPipelineConfig) -> None:
    """A typo here surfaces as a 400 from the compute API at task runtime.

    The step has already been scheduled by then, so the run loses a task to something
    a string check catches at DAG parse.
    """
    prefixes = ('n1-', 'n2-', 'n2d-', 'e2-', 'c2-', 'c3-', 'm1-', 'm2-', 'm3-')
    declared = {
        step: machine_type
        for step in config.steps('pts_')
        if (machine_type := config.step_definition(step).get('machine_type')) is not None
    }
    assert declared, 'expected at least one step to declare a machine_type'
    for step, machine_type in declared.items():
        assert machine_type.startswith(prefixes), f'{step}: {machine_type!r} is not a machine type'
        assert machine_type.count('-') >= 2, f'{step}: {machine_type!r} looks truncated'


def test_every_pts_step_resolves_a_machine_type(config: UnifiedPipelineConfig) -> None:
    """Resolution must never yield None, which the compute operator would reject.

    A step whose yaml entry is empty parses as None rather than a dict, so the
    lookup has to survive that.
    """
    for step in config.steps('pts_'):
        s = pts_step_from_config(step, config)
        assert isinstance(s.machine_type, str) and s.machine_type, f'{step} resolved {s.machine_type!r}'


def test_dag_passes_the_step_machine_type_to_the_vm(dag_bag) -> None:
    """The built task must carry the step's machine type, not the shared default.

    Resolving machine_type correctly on the step is worthless if the DAG hands the
    operator config.pts_machine_type anyway, and nothing else in the suite would
    notice that regression.
    """
    dag = dag_bag.dags['unified_pipeline']
    task = dag.get_task('pts_openfda_input_preparation.run_pts_openfda_input_preparation')
    assert task.machine_type == 'n1-highmem-32'


def test_every_gce_task_matches_its_step(dag_bag, config: UnifiedPipelineConfig) -> None:
    """Every GCE task must carry the machine type its step resolves.

    Checked across all of them rather than one example: only 13 of the pts steps run
    on a vm at all, and a Dataproc step has no run task to inspect, so sampling one
    by name is easy to get wrong.
    """
    dag = dag_bag.dags['unified_pipeline']
    checked = 0
    for step_name in config.steps('pts_'):
        step = pts_step_from_config(step_name, config)
        if not step.is_gce:
            continue
        task = dag.get_task(f'{step_name}.run_{step_name}')
        assert task.machine_type == step.machine_type, (
            f'{step_name}: task has {task.machine_type}, step resolves {step.machine_type}'
        )
        checked += 1
    assert checked > 1, f'expected several gce steps, checked {checked}'

