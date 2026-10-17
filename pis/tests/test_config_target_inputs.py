"""The `target` step downloads no Ensembl species JSON and no GENCODE GFF3."""

from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).parents[1] / 'config.yaml'
"""The config file that ships in the image, not a fixture."""


def _tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = []
    for task in tasks:
        flat.append(task)
        flat.extend(_tasks(task.get('do', [])))
    return flat


def test_no_task_downloads_an_ensembl_species_json() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    tasks = [t for steps in config['steps'].values() for t in _tasks(steps)]
    assert not [t['source'] for t in tasks if 'ensembl.org' in str(t.get('source', '')) and '/json/' in t['source']]


def test_no_task_downloads_the_gencode_gff3() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    assert 'gencode_version' not in config['scratchpad']
    tasks = [t for steps in config['steps'].values() for t in _tasks(steps)]
    assert not [t['source'] for t in tasks if 'gencode' in str(t.get('source', '')).lower()]
