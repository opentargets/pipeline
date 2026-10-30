"""The paths pis copies the staged snapshots from must be the ones staging writes."""

from pathlib import Path

import yaml

from pts.clinical_staging.staged_outputs import DISEASE_ENTITIES, DRUG_ENTITIES, STOP_REASONS
from pts.transformers.clinical.ner import CHEMICAL_MODEL, DISEASE_MODEL, DRUGTEMIST_MODEL
from pts.transformers.clinical.stop_reasons import STOP_REASONS_MODEL

PIS_CONFIG = Path(__file__).resolve().parents[2] / 'pis' / 'config.yaml'

KINDS = (STOP_REASONS, DRUG_ENTITIES, DISEASE_ENTITIES)
HUB_REVISION_KEYS = {
    STOP_REASONS_MODEL: 'stop_reasons_model_revision',
    CHEMICAL_MODEL: 'biobert_chemical_ner_revision',
    DRUGTEMIST_MODEL: 'drugtemist_ner_revision',
    DISEASE_MODEL: 'biobert_diseases_ner_revision',
}


def _scratchpad() -> dict[str, str]:
    """The scratchpad of the pis config."""
    return yaml.safe_load(PIS_CONFIG.read_text())['scratchpad']


def _copy_sources() -> dict[str, list[str]]:
    """The sources of each ``copy_many`` of the pis clinical_report step, keyed by destination."""
    tasks = yaml.safe_load(PIS_CONFIG.read_text())['steps']['clinical_report']
    scratchpad = _scratchpad()
    sources = {}
    for task in tasks:
        if 'sources' in task:
            text = yaml.dump(task['sources'])
            for key, value in scratchpad.items():
                text = text.replace(f'${{{key}}}', str(value))
            sources[task['destination']] = yaml.safe_load(text)
    return sources


def test_pinned_tags_equal_the_code() -> None:
    """The tag pinned for each kind is the one this code computes at the pinned hub revisions."""
    scratchpad = _scratchpad()
    revisions = {model: scratchpad[key] for model, key in HUB_REVISION_KEYS.items()}
    for kind in KINDS:
        key = f'{kind.name}_tag'
        expected = kind.tag(revisions)
        assert scratchpad.get(key) == expected, f'set {key} in pis/config.yaml to {expected}'


def test_release_copy_paths_use_the_pinned_tags() -> None:
    """Each snapshot directory the release copies is named by the pinned tag of its kind."""
    scratchpad = _scratchpad()
    sources = _copy_sources()
    expected = {
        'input/clinical_report/aact_stop_reasons': (
            'gs://clinical_staging/aact/{aact_version}/stop_reasons',
            STOP_REASONS,
        ),
        'input/clinical_report/aact_drug_entities': (
            'gs://clinical_staging/aact/{aact_version}/drug_entities',
            DRUG_ENTITIES,
        ),
        'input/clinical_report/mining_drug_entities': (
            'gs://clinical_staging/clinical_mining/{clinical_mining_version}/drug_entities',
            DRUG_ENTITIES,
        ),
        'input/clinical_report/mining_disease_entities': (
            'gs://clinical_staging/clinical_mining/{clinical_mining_version}/disease_entities',
            DISEASE_ENTITIES,
        ),
    }
    for destination, (parent, kind) in expected.items():
        directory = f'{parent.format(**scratchpad)}/{scratchpad[f"{kind.name}_tag"]}'
        assert sorted(sources[destination]) == [f'{directory}/data.parquet', f'{directory}/metadata.json']
