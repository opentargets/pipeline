"""The shipped ``moa_literature`` step configures the transformer it names."""

from pathlib import Path

import yaml

from pts.transformers.clinical_mira.moa_literature import MoaLiteratureSettings
from pts.transformers.clinical_mira.staged_outputs import MOA_LITERATURE

CONFIG_PATH = Path(__file__).parents[1] / 'config.yaml'


def _task() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())['steps']['moa_literature'][0]


def test_moa_literature_settings_validate() -> None:
    """The settings are accepted by the model and the seed is for the current tag."""
    settings = MoaLiteratureSettings(**_task()['settings'])
    assert settings.import_for_tag == MOA_LITERATURE.tag()
    assert settings.import_experiment_cache is not None


def test_moa_literature_reads_the_input_release() -> None:
    """Both sources and the settings name the same release, and the snapshot goes to the literature root."""
    task = _task()
    release = task['settings']['input_release']
    assert f'/{release}/' in task['source']['match']
    assert f'/{release}/' in task['source']['drug_molecule']
    assert task['destination'] == {'moa': 'gs://clinical_staging/literature'}
    assert set(task['source']) == {'match', 'drug_molecule'}
