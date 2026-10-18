"""The mouse phenotype step, after its port to polars."""

import pytest

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig
from orchestration.models.pts_step import pts_step_from_config


@pytest.fixture(scope='module')
def config() -> UnifiedPipelineConfig:
    """The real pipeline config, so the tests read the shipped yaml."""
    return UnifiedPipelineConfig()


def test_mouse_phenotype_runs_on_a_vm(config: UnifiedPipelineConfig) -> None:
    """No pyspark task is left in the step, so it leaves Dataproc."""
    assert pts_step_from_config('pts_mouse_phenotype', config).is_gce


def test_mouse_phenotype_waits_for_target_and_impc(config: UnifiedPipelineConfig) -> None:
    assert config.step_definition('pts_mouse_phenotype')['depends_on'] == ['pts_target', 'pis_impc']
