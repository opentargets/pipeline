"""Tests for the clinical staging pipeline config."""

from pathlib import Path

import pytest
import yaml

from orchestration.dags.config.clinical_staging import ClinicalStagingConfig

PIS_CONFIG = Path(__file__).parents[2] / 'pis' / 'config.yaml'

FAKE_REVISIONS = {
    'stop_reasons_model_revision': 'aaaa1111',
    'biobert_chemical_ner_revision': 'bbbb2222',
    'drugtemist_ner_revision': 'cccc3333',
    'biobert_diseases_ner_revision': 'dddd4444',
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
        'pis_clinical_mining',
        'pts_clinical_mining_models',
    ]
    assert config.step_definition('pts_aact_trial_extraction')['depends_on'] == ['pis_aact']
    assert config.step_definition('pts_aact_trial_models')['depends_on'] == ['pts_aact_trial_extraction']
    assert config.step_definition('pts_clinical_mining_models')['depends_on'] == ['pis_clinical_mining']
    assert not config.step_definition('pis_clinical_mining').get('depends_on')


def test_release_uri_per_source(config: ClinicalStagingConfig) -> None:
    """Each step writes under its source's version directory and is pinned to that version."""
    aact = f'gs://clinical_staging/aact/{config.aact_version}'
    mining = f'gs://clinical_staging/clinical_mining/{config.clinical_mining_version}'
    for step_name, uri in [
        ('pis_aact', aact),
        ('pts_aact_trial_extraction', aact),
        ('pts_aact_trial_models', aact),
        ('pis_clinical_mining', mining),
        ('pts_clinical_mining_models', mining),
    ]:
        assert config.release_uri(step_name) == uri
        assert config.step_config(step_name)['release_uri'] == uri
        assert config.config_uri(step_name) == f'{uri}/etc/config/{step_name}.yaml'

    assert config.step_config('pis_aact')['scratchpad']['aact_version'] == config.aact_version
    assert (
        config.step_config('pis_clinical_mining')['scratchpad']['clinical_mining_version']
        == config.clinical_mining_version
    )
    (task,) = config.step_config('pts_aact_trial_models')['steps']['aact_trial_models']
    assert task['snapshot'] == config.aact_version
    (task,) = config.step_config('pts_clinical_mining_models')['steps']['clinical_mining_models']
    assert task['snapshot'] == config.clinical_mining_version


def test_machine_type_override_and_default(config: ClinicalStagingConfig) -> None:
    """A step's own machine type wins; the others keep the global one."""
    assert config.machine_type_for('pts_aact_trial_models') == 'n2d-standard-16'
    assert config.machine_type_for('pts_clinical_mining_models') == 'n2d-standard-16'
    assert config.machine_type_for('pts_aact_trial_extraction') == config.machine_type
    assert config.machine_type_for('pis_aact') == config.machine_type
    assert config.machine_type_for('pis_clinical_mining') == config.machine_type


def test_release_copy_paths_use_the_staging_tags() -> None:
    """The snapshot paths the release copies end with the tags staging writes.

    orchestration does not depend on pts, so the tag formats are the literal ones of ``revision_tags``.
    """
    text = _release_step_text({'aact_version': 'A', 'clinical_mining_version': 'M', **FAKE_REVISIONS})
    stop = FAKE_REVISIONS['stop_reasons_model_revision']
    drug = f'{FAKE_REVISIONS["biobert_chemical_ner_revision"]}+{FAKE_REVISIONS["drugtemist_ner_revision"]}'
    disease = FAKE_REVISIONS['biobert_diseases_ner_revision']
    for base in (
        f'gs://clinical_staging/aact/A/stop_reasons/{stop}',
        f'gs://clinical_staging/aact/A/drug_entities/{drug}',
        f'gs://clinical_staging/clinical_mining/M/drug_entities/{drug}',
        f'gs://clinical_staging/clinical_mining/M/disease_entities/{disease}',
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
