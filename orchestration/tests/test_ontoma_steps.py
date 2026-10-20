"""The OnToma lookup table steps, after their port to polars."""

import pytest

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig
from orchestration.models.pts_step import pts_step_from_config


@pytest.fixture(scope='module')
def config() -> UnifiedPipelineConfig:
    """The real pipeline config, so the tests read the shipped yaml."""
    return UnifiedPipelineConfig()


@pytest.mark.parametrize('step', ['pts_ontoma', 'pts_literature_ontoma'])
def test_ontoma_steps_run_on_a_vm(config: UnifiedPipelineConfig, step: str) -> None:
    """No pyspark task is left in the steps, so they leave Dataproc."""
    assert pts_step_from_config(step, config).is_gce


def test_ontoma_waits_for_its_inputs(config: UnifiedPipelineConfig) -> None:
    assert config.step_definition('pts_ontoma')['depends_on'] == ['pis_ontoma', 'pts_disease']


def test_literature_ontoma_waits_for_its_inputs(config: UnifiedPipelineConfig) -> None:
    assert config.step_definition('pts_literature_ontoma')['depends_on'] == [
        'pis_ontoma',
        'pts_disease',
        'pts_target',
        'pts_drug_molecule',
    ]
