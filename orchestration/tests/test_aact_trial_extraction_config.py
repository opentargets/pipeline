"""Tests for the AACT trial extraction pipeline config."""

import pytest

from orchestration.dags.config.aact_trial_extraction import AactTrialExtractionConfig


@pytest.fixture(scope='module')
def config() -> AactTrialExtractionConfig:
    """The real pipeline config, so the tests read the shipped yaml."""
    return AactTrialExtractionConfig()


def test_trial_models_step_runs_after_the_extraction(config: AactTrialExtractionConfig) -> None:
    """The models need the extraction's output, and only its direct dependency is listed."""
    assert 'pts_aact_trial_models' in config.steps()
    assert config.step_definition('pts_aact_trial_models')['depends_on'] == ['pts_aact_trial_extraction']


def test_machine_type_override_and_default(config: AactTrialExtractionConfig) -> None:
    """A step's own machine type wins; the others keep the global one."""
    assert config.machine_type_for('pts_aact_trial_models') == 'n2d-standard-16'
    assert config.machine_type_for('pts_aact_trial_extraction') == config.machine_type
    assert config.machine_type_for('pis_aact') == config.machine_type


def test_trial_models_step_config_points_at_the_snapshot(config: AactTrialExtractionConfig) -> None:
    """The step must write the snapshot named after the AACT version it runs on."""
    step_config = config.step_config('pts_aact_trial_models')
    assert list(step_config['steps']) == ['aact_trial_models']
    (task,) = step_config['steps']['aact_trial_models']
    assert task['snapshot'] == config.aact_version
