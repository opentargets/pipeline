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


def _settings(name: str) -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    tasks = config['steps']['l2g']
    (task,) = [task for task in tasks if task.get('name') == name]
    return task['settings']


def _l2g_train_settings() -> dict[str, Any]:
    return _settings('transform l2g_train')


def test_config_hyperparameters_match_the_model_default() -> None:
    assert _l2g_train_settings()['hyperparameters'] == DEFAULT_HYPERPARAMETERS


def test_the_shap_background_size_is_configured_exactly_once() -> None:
    """Only training may set it; prediction sizes its masker from the background it loads.

    Configured in both places, the two can disagree with nothing to notice: raise training's to
    1000 and leave prediction's at 100 and `Independent` would use 100 of the 1000 rows while
    `metrics.json` recorded 1000 -- the silently-truncated background this port exists to remove.
    Unlike `features_list`, which a YAML anchor keeps in step, there is nothing structural to bind
    two copies of a scalar, so the guard is that the second copy must not exist.
    """
    assert 'shap_background_size' in _l2g_train_settings()
    assert 'shap_background_size' not in _settings('transform l2g_predict')
