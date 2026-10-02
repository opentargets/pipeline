"""Checks for the AACT extraction DAG's evaluation step wiring."""

from orchestration.dags.config.aact_trial_extraction import AactTrialExtractionConfig


def test_judge_runs_after_analysis_in_karenina_image() -> None:
    config = AactTrialExtractionConfig()
    judge = 'pts_aact_trial_extraction_judge'

    assert config.step_definition(judge)['depends_on'] == ['pts_aact_trial_extraction_analysis']
    assert config.step_image(judge) == config.images['pts-with-karenina']
    assert config.step_env_vars(judge)['PTS_STEP'] == 'aact_trial_extraction_judge'
    judge_spec = config.step_config(judge)['steps']['aact_trial_extraction_judge'][0]
    extraction_spec = config.step_config('pts_aact_trial_extraction')['steps']['aact_trial_extraction'][0]
    assert judge_spec['trials_needing_extraction'] == extraction_spec['destination']['trials_needing_extraction']
    assert config.step_definition(judge)['gce_secret_files']['openai-token'] == '/var/run/secrets/openai-api-key'
