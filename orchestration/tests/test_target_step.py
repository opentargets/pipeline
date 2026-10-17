"""The target step, after pts_pre_target was folded into it."""

import pytest

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig
from orchestration.models.pts_step import pts_step_from_config


@pytest.fixture(scope='module')
def config() -> UnifiedPipelineConfig:
    """The real pipeline config, so the tests read the shipped yaml."""
    return UnifiedPipelineConfig()


def test_target_runs_on_a_vm(config: UnifiedPipelineConfig) -> None:
    """No pyspark task is left in the step, so it leaves Dataproc."""
    assert pts_step_from_config('pts_target', config).is_gce


def test_pre_target_is_gone(config: UnifiedPipelineConfig) -> None:
    assert 'pts_pre_target' not in config.steps()


def test_target_waits_for_its_inputs_and_reactome(config: UnifiedPipelineConfig) -> None:
    assert config.step_definition('pts_target')['depends_on'] == ['pis_target', 'pis_chembl', 'pts_reactome']


def test_target_index_readers_wait_for_target(config: UnifiedPipelineConfig) -> None:
    assert config.step_definition('pts_drug_mechanism_of_action')['depends_on'] == ['pts_target']
    assert 'pts_target' in config.step_definition('gentropy_study')['depends_on']
