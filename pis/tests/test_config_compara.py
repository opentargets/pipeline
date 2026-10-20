"""The Compara homology downloads of the `target` step."""

from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).parents[1] / 'config.yaml'
"""The config file that ships in the image, not a fixture."""


def _task(tasks: list[dict[str, Any]], name: str) -> dict[str, Any]:
    (task,) = [t for t in tasks if t.get('name') == name]
    return task


def test_compara_homologies_download_only_the_compara_files() -> None:
    """Gene names come from the core dumps, so no species JSON is downloaded for homology."""
    config = yaml.safe_load(CONFIG_PATH.read_text())
    task = _task(config['steps']['target'], 'explode compara homologies')
    assert [t['name'] for t in task['do']] == ['copy ${each} protein homology', 'copy ${each} ncrna homology']
    assert [t['destination'] for t in task['do']] == [
        'input/target/homologue/homologies/protein-${each}.tsv.gz',
        'input/target/homologue/homologies/ncrna-${each}.tsv.gz',
    ]
    assert not any('gene_dictionary' in str(t) for t in config['steps']['target'])
