import os

import pytest


@pytest.fixture(autouse=True)
def no_pos_environment(monkeypatch):
    """Each test starts without POS_* variables, and none it sets survive it."""
    for var in [v for v in os.environ if v.startswith('POS_')]:
        monkeypatch.delenv(var)
    yield
    # core.setup() sets POS_CONFIG_PATH outside monkeypatch; monkeypatch restores the originals after this
    for var in [v for v in os.environ if v.startswith('POS_')]:
        del os.environ[var]
