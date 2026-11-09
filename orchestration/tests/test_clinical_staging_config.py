"""Tests for the clinical staging pipeline config."""

from datetime import timedelta
from pathlib import Path

import pytest
import yaml

from orchestration.dags.config.clinical_staging import ClinicalStagingConfig

PIS_CONFIG = Path(__file__).parents[2] / 'pis' / 'config.yaml'
PTS_CONFIG = Path(__file__).parents[2] / 'pts' / 'config.yaml'

FAKE_TAGS = {
    'extraction_tag': 'eeee5555-p44444444',
    'stop_reasons_tag': 'aaaa1111-p11111111',
    'drug_entities_tag': 'bbbb2222+cccc3333-p22222222',
    'disease_entities_tag': 'dddd4444-p33333333',
    'moa_tag': 'claude-opus-5-5-p55555555',
    'moa_literature_tag': 'claude-opus-5-5-p66666666',
    'moa_literature_input_release': 'R',
}


@pytest.fixture(scope='module')
def config() -> ClinicalStagingConfig:
    """The real pipeline config, so the tests read the shipped yaml."""
    return ClinicalStagingConfig()


def _release_step_text(scratchpad: dict[str, str] | None = None) -> str:
    """The pis clinical_report step as text, with ``${key}`` substituted from the scratchpad as pis does."""
    steps = yaml.safe_load(PIS_CONFIG.read_text())['steps']
    text = yaml.dump(steps['clinical_report'])
    for key, value in (scratchpad or {}).items():
        text = text.replace(f'${{{key}}}', value)
    return text


def test_steps_and_dependencies(config: ClinicalStagingConfig) -> None:
    """Only direct dependencies are listed, and each source stages its own models."""
    assert config.steps() == [
        'pis_aact',
        'pts_aact_trial_extraction',
        'pts_aact_trial_models',
        'pts_aact_moa',
        'pis_clinical_mining',
        'pts_clinical_mining_models',
        'pts_moa_literature',
    ]
    assert config.step_definition('pts_aact_trial_extraction')['depends_on'] == ['pis_aact']
    assert config.step_definition('pts_aact_trial_models')['depends_on'] == ['pts_aact_trial_extraction']
    assert config.step_definition('pts_clinical_mining_models')['depends_on'] == [
        'pis_clinical_mining',
        'pts_aact_trial_models',
    ]
    assert not config.step_definition('pis_clinical_mining').get('depends_on')


def test_release_uri_per_source(config: ClinicalStagingConfig) -> None:
    """Each step writes under its source's version directory and is pinned to that version."""
    aact = f'gs://clinical_staging/aact/{config.staging_version}'
    mining = f'gs://clinical_staging/clinical_mining/{config.staging_version}'
    literature = f'gs://clinical_staging/literature/{config.staging_version}'
    for step_name, uri in [
        ('pis_aact', aact),
        ('pts_aact_trial_extraction', aact),
        ('pts_aact_trial_models', aact),
        ('pts_aact_moa', aact),
        ('pis_clinical_mining', mining),
        ('pts_clinical_mining_models', mining),
        ('pts_moa_literature', literature),
    ]:
        assert config.release_uri(step_name) == uri
        assert config.step_config(step_name)['release_uri'] == uri
        assert config.config_uri(step_name) == f'{uri}/etc/config/{step_name}.yaml'


def test_staging_version_feeds_both_sources(config: ClinicalStagingConfig) -> None:
    """One date names both trees, the pis scratchpad variables and the pts snapshots."""
    assert config.release_uri('pis_aact').endswith(f'/{config.staging_version}')
    assert config.release_uri('pis_clinical_mining').endswith(f'/{config.staging_version}')
    assert config.step_config('pis_aact')['scratchpad']['staging_version'] == config.staging_version
    assert config.step_config('pis_clinical_mining')['scratchpad']['staging_version'] == config.staging_version
    (task,) = config.step_config('pts_aact_trial_models')['steps']['aact_trial_models']
    assert task['settings']['snapshot'] == config.staging_version
    (task,) = config.step_config('pts_clinical_mining_models')['steps']['clinical_mining_models']
    assert task['settings']['snapshot'] == config.staging_version


def test_machine_type_override_and_default(config: ClinicalStagingConfig) -> None:
    """A step's own machine type wins; the others keep the global one."""
    assert config.machine_type_for('pts_aact_trial_models') == 'n2d-standard-16'
    assert config.machine_type_for('pts_clinical_mining_models') == 'n2d-standard-16'
    assert config.machine_type_for('pts_aact_trial_extraction') == 'n1-highmem-8'
    assert config.machine_type_for('pis_aact') == config.machine_type
    assert config.machine_type_for('pis_clinical_mining') == config.machine_type


def test_release_copy_paths_use_the_pinned_tags() -> None:
    """The snapshot paths the release copies end with the tags pinned in the pis scratchpad.

    orchestration does not depend on pts; pts tests that the pins equal the tags staging writes.
    """
    text = _release_step_text({'staging_version': 'S', **FAKE_TAGS})
    stop = FAKE_TAGS['stop_reasons_tag']
    drug = FAKE_TAGS['drug_entities_tag']
    disease = FAKE_TAGS['disease_entities_tag']
    extraction = FAKE_TAGS['extraction_tag']
    for base in (
        f'gs://clinical_staging/aact/S/stop_reasons/{stop}',
        f'gs://clinical_staging/aact/S/extraction/{extraction}',
        f'gs://clinical_staging/aact/S/extraction/{extraction}/moa/{FAKE_TAGS["moa_tag"]}',
        f'gs://clinical_staging/aact/S/extraction/{extraction}/drug_entities/{drug}',
        f'gs://clinical_staging/clinical_mining/S/drug_entities/{drug}',
        f'gs://clinical_staging/clinical_mining/S/disease_entities/{disease}',
        f'gs://clinical_staging/literature/R/moa_literature/{FAKE_TAGS["moa_literature_tag"]}',
    ):
        for name in ('data.parquet', 'metadata.json'):
            assert f'{base}/{name}' in text
    assert '${' not in text


def test_release_reads_nothing_from_ot_team() -> None:
    """The release copies from gs://clinical_staging only, and downloads no model."""
    text = _release_step_text()
    assert 'ot-team' not in text
    assert 'huggingface' not in text
    assert 'aact_data' not in text


def test_step_execution_timeout_is_read_from_the_step(config: ClinicalStagingConfig) -> None:
    """Only a step that declares ``execution_timeout_hours`` has a timeout."""
    assert config.execution_timeout_for('pts_aact_moa') == timedelta(hours=30)
    for step_name in config.steps():
        if step_name not in ('pts_aact_moa', 'pts_moa_literature'):
            assert config.execution_timeout_for(step_name) is None


def test_step_env_vars_include_the_steps_container_env(config: ClinicalStagingConfig) -> None:
    """A step's ``container_env`` joins the step and config path pair; other steps get the pair only."""
    assert config.step_env_vars('pts_aact_moa') == {
        'PTS_STEP': 'aact_moa',
        'PTS_CONFIG_PATH': '/config.yaml',
        'ANTHROPIC_GOOGLE_CLOUD_WORKSPACE_ID': 'wrkspc_01PEuYsbEifFUZBttcpwP49Q',
        'ANTHROPIC_GOOGLE_CLOUD_PROJECT': 'open-targets-eu-dev',
    }
    assert config.step_env_vars('pts_aact_trial_models') == {
        'PTS_STEP': 'aact_trial_models',
        'PTS_CONFIG_PATH': '/config.yaml',
    }


def test_container_env_cannot_override_the_step_pair(config: ClinicalStagingConfig, monkeypatch) -> None:
    """A colliding ``container_env`` key is an error."""
    monkeypatch.setitem(config._steps['pts_aact_moa'], 'container_env', {'PTS_STEP': 'other'})
    with pytest.raises(ValueError, match='PTS_STEP'):
        config.step_env_vars('pts_aact_moa')


def test_aact_moa_step_config(config: ClinicalStagingConfig) -> None:
    """The step uploads only aact_moa, pinned to the AACT version directory."""
    uploaded = config.step_config('pts_aact_moa')
    assert list(uploaded['steps']) == ['aact_moa']
    assert uploaded['release_uri'] == f'gs://clinical_staging/aact/{config.staging_version}'
    assert config.machine_type_for('pts_aact_moa') == 'n1-highmem-8'
    assert not config.step_definition('pts_aact_moa').get('gce_secret_files')


def test_moa_literature_step_has_timeout_and_env(config: ClinicalStagingConfig) -> None:
    """moa_literature runs on a large machine with the aact_moa timeout and Claude workspace."""
    assert config.execution_timeout_for('pts_moa_literature') == timedelta(hours=30)
    assert config.machine_type_for('pts_moa_literature') == 'n2d-highmem-32'
    assert config.step_env_vars('pts_moa_literature') == config.step_env_vars('pts_aact_moa') | {
        'PTS_STEP': 'moa_literature'
    }
    assert list(config.step_config('pts_moa_literature')['steps']) == ['moa_literature']


def test_moa_literature_has_no_staging_dependencies(config: ClinicalStagingConfig) -> None:
    """moa_literature reads published releases, so it waits for no other step and none waits for it."""
    assert not config.step_definition('pts_moa_literature').get('depends_on')
    for step_name in config.steps():
        assert 'pts_moa_literature' not in (config.step_definition(step_name).get('depends_on') or [])


def test_moa_literature_snapshot_lands_under_the_literature_root(config: ClinicalStagingConfig) -> None:
    """The destination is the literature root, so the snapshot is gs://clinical_staging/literature/<release>/..."""
    step = config.step_config('pts_moa_literature')['steps']['moa_literature'][0]
    assert step['destination'] == {'moa': 'gs://clinical_staging/literature'}
    shipped = yaml.safe_load(PTS_CONFIG.read_text())['steps']['moa_literature'][0]['settings']['input_release']
    assert step['settings']['input_release'] == shipped
