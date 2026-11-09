"""The shipped ``moa_literature`` step configures the transformer it names."""

from pathlib import Path

import yaml

from pts.transformers.clinical_mira.moa_literature import MoaLiteratureSettings
from pts.transformers.clinical_mira.staged_outputs import MOA_LITERATURE

CONFIG_PATH = Path(__file__).parents[1] / 'config.yaml'


def _task() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())['steps']['moa_literature'][0]


def test_moa_literature_settings_validate() -> None:
    """The settings are accepted by the model, a seed tag is the current tag and a seed release names a seed."""
    settings = MoaLiteratureSettings(**_task()['settings'])
    if settings.import_for_tag is not None:
        assert settings.import_for_tag == MOA_LITERATURE.tag()
    if settings.import_for_release is not None:
        assert settings.import_experiment_cache is not None


def test_moa_literature_caps_compute_and_shards() -> None:
    """The step refuses to compute more than 3000 drugs and writes shards of 5000."""
    settings = MoaLiteratureSettings(**_task()['settings'])
    assert settings.max_compute == 4500
    assert settings.shard_size == 5000


def test_moa_literature_reads_the_input_release() -> None:
    """Both sources and the settings name the same release, and the snapshot goes to the literature root."""
    task = _task()
    release = task['settings']['input_release']
    assert f'/{release}/' in task['source']['match']
    assert f'/{release}/' in task['source']['drug_molecule']
    assert task['destination'] == {'moa': 'gs://clinical_staging/literature'}
    assert set(task['source']) == {'match', 'drug_molecule'}
