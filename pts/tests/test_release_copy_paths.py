"""The paths pis copies the staged snapshots from must be the ones staging writes."""

from pathlib import Path

import yaml

from pts.transformers.clinical.hf_models import revision_tags
from pts.transformers.clinical.ner import CHEMICAL_MODEL, DISEASE_MODEL, DRUGTEMIST_MODEL
from pts.transformers.clinical.stop_reasons import STOP_REASONS_MODEL

PIS_CONFIG = Path(__file__).resolve().parents[2] / 'pis' / 'config.yaml'

FAKE_REVISIONS = {
    STOP_REASONS_MODEL: 'aaaa1111',
    CHEMICAL_MODEL: 'bbbb2222',
    DRUGTEMIST_MODEL: 'cccc3333',
    DISEASE_MODEL: 'dddd4444',
}
SCRATCHPAD = {
    'aact_version': 'A',
    'clinical_mining_version': 'M',
    'stop_reasons_model_revision': FAKE_REVISIONS[STOP_REASONS_MODEL],
    'biobert_chemical_ner_revision': FAKE_REVISIONS[CHEMICAL_MODEL],
    'drugtemist_ner_revision': FAKE_REVISIONS[DRUGTEMIST_MODEL],
    'biobert_diseases_ner_revision': FAKE_REVISIONS[DISEASE_MODEL],
}


def _copy_sources() -> dict[str, list[str]]:
    """The sources of each ``copy_many`` of the pis clinical_report step, keyed by destination."""
    tasks = yaml.safe_load(PIS_CONFIG.read_text())['steps']['clinical_report']
    sources = {}
    for task in tasks:
        if 'sources' in task:
            text = yaml.dump(task['sources'])
            for key, value in SCRATCHPAD.items():
                text = text.replace(f'${{{key}}}', value)
            sources[task['destination']] = yaml.safe_load(text)
    return sources


def test_release_copy_paths_use_the_staging_tags() -> None:
    """Each snapshot directory the release copies ends with the tag ``revision_tags`` gives staging."""
    tags = revision_tags(FAKE_REVISIONS)
    sources = _copy_sources()
    expected = {
        'input/clinical_report/aact_stop_reasons': ('gs://clinical_staging/aact/A/stop_reasons', 'stop_reasons'),
        'input/clinical_report/aact_drug_entities': ('gs://clinical_staging/aact/A/drug_entities', 'drug_entities'),
        'input/clinical_report/mining_drug_entities': (
            'gs://clinical_staging/clinical_mining/M/drug_entities',
            'drug_entities',
        ),
        'input/clinical_report/mining_disease_entities': (
            'gs://clinical_staging/clinical_mining/M/disease_entities',
            'disease_entities',
        ),
    }
    for destination, (parent, name) in expected.items():
        assert sorted(sources[destination]) == [
            f'{parent}/{tags[name]}/data.parquet',
            f'{parent}/{tags[name]}/metadata.json',
        ]
