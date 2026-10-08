"""Tests for the environment PTS steps run with."""

import pytest

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig
from orchestration.models.pts_step import pts_step_from_config


@pytest.fixture(scope='module')
def config() -> UnifiedPipelineConfig:
    """The real pipeline config, so the tests read the shipped yaml."""
    return UnifiedPipelineConfig()


def test_polars_spills_to_the_work_disk(config: UnifiedPipelineConfig) -> None:
    """Polars 2 must spill under the mounted work disk, not the 10 GB boot disk.

    Its default spill dir is the OS temp dir; a spill there fills the boot disk.
    """
    env = config.pts_env_vars('pts_search')
    work_path = config.pts.config['work_path']
    assert env['POLARS_OOC_SPILL_DIR'].startswith(f'{work_path}/')
    assert int(env['POLARS_OOC_DISK_BUDGET_MB']) * 1024**2 < config.pts_disk_size * 1000**3


def test_every_gce_task_gets_the_spill_dir(dag_bag, config: UnifiedPipelineConfig) -> None:
    """Every PTS vm task must carry the spill settings, since any step may use polars."""
    dag = dag_bag.dags['unified_pipeline']
    checked = 0
    for step_name in config.steps('pts_'):
        if not pts_step_from_config(step_name, config).is_gce:
            continue
        task = dag.get_task(f'{step_name}.run_{step_name}')
        assert task.container_env['POLARS_OOC_SPILL_DIR'] == config.pts_env_vars(step_name)['POLARS_OOC_SPILL_DIR']
        checked += 1
    assert checked > 1, f'expected several gce steps, checked {checked}'
