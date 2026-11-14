"""The association steps, after their port to polars."""

import pytest

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig
from orchestration.models.pts_step import PTSDataprocStep, PTSStep, pts_step_from_config


@pytest.fixture(scope='module')
def config() -> UnifiedPipelineConfig:
    """The real pipeline config, so the tests read the shipped yaml."""
    return UnifiedPipelineConfig()


@pytest.mark.parametrize('step', ['pts_association', 'pts_association_otf', 'pts_association_timeseries_view'])
def test_association_steps_run_on_a_vm(config: UnifiedPipelineConfig, step: str) -> None:
    """No pyspark task is left in the steps, so they leave Dataproc."""
    resolved = pts_step_from_config(step, config)
    assert isinstance(resolved, PTSStep)
    assert not isinstance(resolved, PTSDataprocStep)


def test_association_runs_on_the_default_pts_machine(config: UnifiedPipelineConfig) -> None:
    assert pts_step_from_config('pts_association', config).machine_type == config.pts_machine_type
