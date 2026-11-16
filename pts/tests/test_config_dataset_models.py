"""Every dataset a pts task writes to `output/` has a dataset model, so its task validates it.

A new output without a model would be written unvalidated; this test fails until the model exists.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from pts.schemas.dataset.discovery import datasets_by_location

CONFIG = Path(__file__).parents[1] / 'config.yaml'


def _outputs() -> list[tuple[str, str, str]]:
    """(step, task, path) of every `output/` destination in the config."""
    steps: dict[str, list[dict[str, Any]]] = yaml.safe_load(CONFIG.read_text())['steps']
    found = []
    for step, tasks in steps.items():
        for task in tasks or []:
            destination = task.get('destination')
            paths = destination.values() if isinstance(destination, dict) else [destination]
            found += [
                (step, task['name'], path.rstrip('/'))
                for path in paths
                if isinstance(path, str) and path.startswith('output/')
            ]
    return found


OUTPUTS = _outputs()


def test_the_config_writes_outputs() -> None:
    assert len(OUTPUTS) > 50


@pytest.mark.parametrize(('step', 'task', 'path'), OUTPUTS, ids=[f'{s}:{p}' for s, _, p in OUTPUTS])
def test_every_output_has_a_dataset_model(step: str, task: str, path: str) -> None:
    assert path in datasets_by_location(), (
        f'step {step!r}, task {task!r} writes {path!r}, but no dataset model is written there. '
        f'Declare one in pts/src/pts/schemas/outputs/ with dataset_name = {path.removeprefix("output/")!r} '
        '(scripts/output_schemas/ can generate a first version from a written output).'
    )
