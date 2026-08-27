"""Tests for the per-step PTS pool_size override."""

import pytest

from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig


@pytest.fixture(scope='module')
def config() -> UnifiedPipelineConfig:
    """The real pipeline config, so the tests read the shipped yaml."""
    return UnifiedPipelineConfig()


def test_step_without_an_override_gets_no_env_var(config: UnifiedPipelineConfig) -> None:
    """Most steps must not carry PTS_POOL_SIZE at all.

    Emitting it unconditionally would pin a value into the environment, where it
    outranks the yaml, and silently defeat any later change to the default.
    """
    env = config.pts_env_vars('pts_target')
    assert 'PTS_POOL_SIZE' not in env
    assert env['PTS_STEP'] == 'target'


def test_openfda_lowers_its_pool_size(config: UnifiedPipelineConfig) -> None:
    """openfda holds whole archives in memory, so it runs fewer workers than cores."""
    env = config.pts_env_vars('pts_openfda_input_preparation')
    assert env['PTS_POOL_SIZE'] == '8'
    assert env['PTS_STEP'] == 'openfda_input_preparation'


def test_overrides_are_below_the_default_and_above_otters_floor(
    config: UnifiedPipelineConfig,
) -> None:
    """Every declared pool_size must be a usable reduction.

    A value at or above the default is a no-op or an oversubscription, and otter's
    `_validate_pool_size` rejects anything below 2 -- which would exit the step
    rather than fall back, so a bad value here breaks the run.
    """
    default = config.pts.config['pool_size']
    declared = {
        step: definition['pool_size']
        for step in config.steps('pts_')
        if (definition := config.step_definition(step)).get('pool_size') is not None
    }
    assert declared, 'expected at least one step to declare a pool_size'
    for step, size in declared.items():
        assert 1 < size < default, f'{step} pool_size {size} must be between 2 and {default - 1}'


def test_override_reaches_the_step_as_a_string(config: UnifiedPipelineConfig) -> None:
    """Env vars must be strings; a bare int fails when the operator builds the VM."""
    for step in config.steps('pts_'):
        env = config.pts_env_vars(step)
        assert all(isinstance(v, str) for v in env.values()), f'{step} has a non-str env value'
