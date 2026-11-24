import os
import urllib.request
import uuid

import pytest

from pos.services.clickhouse import connect

# read at import: the autouse fixture below clears POS_* before each test
CLICKHOUSE_SERVER = os.environ.get('POS_TEST_CLICKHOUSE', 'localhost:8123')
REQUIRE_CLICKHOUSE = os.environ.get('POS_REQUIRE_CLICKHOUSE') == '1'


@pytest.fixture(autouse=True)
def no_pos_environment(monkeypatch):
    """Each test starts without POS_* variables, and none it sets survive it."""
    for var in [v for v in os.environ if v.startswith('POS_')]:
        monkeypatch.delenv(var)
    yield
    # core.setup() sets POS_CONFIG_PATH outside monkeypatch; monkeypatch restores the originals after this
    for var in [v for v in os.environ if v.startswith('POS_')]:
        del os.environ[var]


@pytest.fixture
def clickhouse_server() -> tuple[str, int]:
    """(host, port) of the live test ClickHouse, as `test_serving_sql.py` finds it: skip without one, or fail
    under `POS_REQUIRE_CLICKHOUSE=1`."""
    host, port = CLICKHOUSE_SERVER.rsplit(':', 1)
    try:
        with urllib.request.urlopen(f'http://{host}:{port}/ping', timeout=2) as r:
            reachable = r.read().decode().strip() == 'Ok.'
    except OSError:
        reachable = False
    if not reachable:
        if REQUIRE_CLICKHOUSE:
            pytest.fail(f'no ClickHouse at {CLICKHOUSE_SERVER} and POS_REQUIRE_CLICKHOUSE=1')
        pytest.skip(f'no ClickHouse at {CLICKHOUSE_SERVER} (set POS_TEST_CLICKHOUSE; POS_REQUIRE_CLICKHOUSE=1 fails)')
    return host, int(port)


@pytest.fixture
def clickhouse_database(clickhouse_server):
    """A client on a fresh database of the live test ClickHouse, dropped afterwards."""
    host, port = clickhouse_server
    database = f'pos_test_{uuid.uuid4().hex}'
    admin = connect(host=host, port=port)
    admin.command(f'CREATE DATABASE {database}')
    try:
        yield connect(host=host, port=port, database=database)
    finally:
        admin.command(f'DROP DATABASE IF EXISTS {database} SYNC')
