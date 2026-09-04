"""A consumer must not read the parent directory of a named transformer destination.

Since the shared writer landed, a transformer destination is a *directory* of parquet
parts rather than a file. Where the destination names a dataset -- it ends in something
like ``homo_sapiens.parquet`` or ``${uuid}.parquet`` -- the parent then holds only
subdirectories, and a consumer reading that parent finds no parquet at all.

The failure is silent at config time and cryptic at runtime: gentropy reported
``Parquet file is empty`` on ``intermediate/target/ensembl``, and pts_openfda reported
``UNABLE_TO_INFER_SCHEMA`` on ``intermediate/openfda``. Neither message names the path
layout. This test makes the mismatch visible in the config instead.

Reading the destination itself is fine, and so is globbing the parent -- spark reads a
directory of parts, and ``.../gene_dictionary/*.parquet`` resolves to one per match.
"""

from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).parents[1] / 'config.yaml'
"""The config file that ships in the image, not a fixture."""


def _tasks(step_tasks: Any) -> list[dict[str, Any]]:
    """Flatten a step's tasks, following the nested `do:` blocks of explode_glob."""
    out: list[dict[str, Any]] = []
    stack = list(step_tasks or [])
    while stack:
        task = stack.pop()
        if not isinstance(task, dict):
            continue
        out.append(task)
        for value in task.values():
            if isinstance(value, list):
                stack.extend(v for v in value if isinstance(v, dict))
            elif isinstance(value, dict):
                stack.append(value)
    return out


def _sources(node: Any, found: list[str]) -> None:
    """Collect every `source:` string, at any depth."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == 'source':
                if isinstance(value, str):
                    found.append(value)
                elif isinstance(value, dict):
                    found.extend(v for v in value.values() if isinstance(v, str))
            _sources(value, found)
    elif isinstance(node, list):
        for value in node:
            _sources(value, found)


def test_no_consumer_reads_the_parent_of_a_named_dataset() -> None:
    """Read the dataset, or glob the parent -- never the bare parent."""
    config = yaml.safe_load(CONFIG_PATH.read_text())
    steps = config.get('steps') or {}

    # a destination that names a dataset makes its parent a directory of directories
    named_parents: dict[str, str] = {}
    for step_name, step_tasks in steps.items():
        for task in _tasks(step_tasks):
            destination = task.get('destination')
            if 'transformer' not in task or not isinstance(destination, str):
                continue
            if '/' not in destination or not destination.endswith('.parquet'):
                continue
            named_parents[destination.rsplit('/', 1)[0]] = f'{step_name} -> {destination}'

    assert named_parents, 'expected some transformer destinations to name a dataset'

    sources: list[str] = []
    _sources(config, sources)

    offenders = [
        f'{source!r} is the parent of {named_parents[source]}' for source in set(sources) if source in named_parents
    ]
    assert not offenders, (
        'a source reads the bare parent of a named dataset, so it will find only '
        'subdirectories: ' + '; '.join(sorted(offenders))
    )
