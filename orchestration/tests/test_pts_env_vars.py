"""Tests for the environment PTS steps run with."""

import re

import pytest

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig
from orchestration.models.pts_step import pts_step_from_config


@pytest.fixture(scope='module')
def config() -> UnifiedPipelineConfig:
    """The real pipeline config, so the tests read the shipped yaml."""
    return UnifiedPipelineConfig()


def _gce_tasks(dag_bag, config: UnifiedPipelineConfig) -> list:
    """Return the run task of every PTS step that runs on a vm."""
    dag = dag_bag.dags['unified_pipeline']
    return [
        dag.get_task(f'{step_name}.run_{step_name}')
        for step_name in config.steps('pts_')
        if pts_step_from_config(step_name, config).is_gce
    ]


def _container_mounts(task) -> list[str]:
    """Return the container-side paths of the `-v host:container` volumes the vm's docker run gets."""
    return re.findall(r'-v \S+:(\S+)', task.build_volume_params())


def test_polars_spills_inside_a_mounted_volume(dag_bag, config: UnifiedPipelineConfig) -> None:
    """Polars 2 must spill into a volume the container mounts, not onto the 10 GB boot disk.

    Its default spill dir is the OS temp dir. The check reads the volumes from the docker
    command the operator builds, not from `work_path`, so it fails if the config and the
    mount ever drift apart.
    """
    tasks = _gce_tasks(dag_bag, config)
    assert len(tasks) > 1, f'expected several gce steps, found {len(tasks)}'
    for task in tasks:
        spill_dir = task.container_env['POLARS_OOC_SPILL_DIR']
        mounts = _container_mounts(task)
        assert any(spill_dir.startswith(f'{m.rstrip("/")}/') for m in mounts), (
            f'{task.task_id}: spill dir {spill_dir} is not under any mounted volume {mounts}'
        )


def test_polars_disk_budget_fits_the_work_disk(dag_bag, config: UnifiedPipelineConfig) -> None:
    """The spill budget must leave room on the work disk for the step's own data."""
    for task in _gce_tasks(dag_bag, config):
        budget_bytes = int(task.container_env['POLARS_OOC_DISK_BUDGET_MB']) * 1024**2
        assert budget_bytes < task.work_disk_size_gb * 1000**3, f'{task.task_id}: budget exceeds the work disk'
