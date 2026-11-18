"""The models import without pts or pandera, so any package can read them."""

import subprocess
import sys

SCRIPT = """
import sys

import ot_datamodels, ot_datamodels.outputs, ot_datamodels.discovery

assert 'pandera' not in sys.modules, 'pandera imported'
assert 'pts' not in sys.modules, 'pts imported'
assert len(ot_datamodels.discovery.discover_datasets()) == 69
"""


def test_imports_without_pts_or_pandera() -> None:
    result = subprocess.run([sys.executable, '-c', SCRIPT], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
