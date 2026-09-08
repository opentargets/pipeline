"""The `l2g_train` task's config hyperparameters must not drift from the model default.

`config.yaml`'s `hyperparameters` block hand-duplicates `DEFAULT_HYPERPARAMETERS` in
`l2g/model.py`, read directly off the released `classifier.skops`. Nothing ties the two
together structurally -- a config edit could silently change what gets trained without
touching the constant that documents it, or vice versa. This is the same class of hazard
as the `&l2g_features` anchor, just without an anchor available: `DEFAULT_HYPERPARAMETERS`
is a Python constant, not something `yaml.safe_load` can see, so the guard has to compare
the two explicitly instead.
"""

from pathlib import Path
from typing import Any

import yaml

from pts.transformers.l2g.model import DEFAULT_HYPERPARAMETERS

CONFIG_PATH = Path(__file__).parents[1] / 'config.yaml'
"""The config file that ships in the image, not a fixture."""


def _l2g_train_settings() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    tasks = config['steps']['l2g']
    (task,) = [task for task in tasks if task.get('name') == 'transform l2g_train']
    return task['settings']


def test_config_hyperparameters_match_the_model_default() -> None:
    assert _l2g_train_settings()['hyperparameters'] == DEFAULT_HYPERPARAMETERS
