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


def test_every_pts_step_resolves_a_machine_type(config: UnifiedPipelineConfig) -> None:
    """Resolution must never yield None, which the compute operator would reject.

    A step whose yaml entry is empty parses as None rather than a dict, so the
    lookup has to survive that.
    """
    for step in config.steps('pts_'):
        s = pts_step_from_config(step, config)
        assert isinstance(s.machine_type, str) and s.machine_type, f'{step} resolved {s.machine_type!r}'


def test_every_gce_task_matches_its_step(dag_bag, config: UnifiedPipelineConfig) -> None:
    """Every GCE task must carry the machine type its step resolves.

    Resolving machine_type on the step is worthless if the DAG hands the operator
    the shared default anyway, and nothing else in the suite notices that.

    Checked across every GCE step rather than one example: only some PTS steps run
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
