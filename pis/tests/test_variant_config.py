"""The GPN-Star downloads of the `variant` step."""

import re
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).parents[1] / 'config.yaml'
"""The config file that ships in the image, not a fixture."""


def test_gpn_star_copy_is_pinned_and_complete() -> None:
    """Every chromosome file is mirrored from one pinned Hugging Face commit."""
    config: dict[str, Any] = yaml.safe_load(CONFIG_PATH.read_text())
    (task,) = [t for t in config['steps']['variant'] if t.get('name', '').startswith('explode gpn-star')]
    assert task['foreach'] == [f'llr_chr{c}.parquet' for c in [*range(1, 23), 'X', 'Y']]
    (inner,) = task['do']
    assert '/resolve/${gpn_star_revision}/data/gpn-star-hg38-m447-200m/llr/${each}' in inner['source']
    assert inner['destination'] == 'input/variant/gpn_star/${each}'
    assert re.fullmatch(r'[0-9a-f]{40}', config['scratchpad']['gpn_star_revision'])
