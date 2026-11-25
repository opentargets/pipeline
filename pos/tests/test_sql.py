"""split_statements splits ClickHouse SQL on top-level semicolons."""

from pathlib import Path

import pytest

from pos.sql import split_statements

CLICKHOUSE = Path(__file__).parents[1] / 'config' / 'clickhouse'
SCRIPTS = sorted(CLICKHOUSE.glob('**/*.sql'))


def test_split_statements():
    sql = "SELECT ';' ; -- a;b\nSELECT 1; /* x;y */ SELECT `a;b`;"
    assert split_statements(sql) == ["SELECT ';'", '-- a;b\nSELECT 1', '/* x;y */ SELECT `a;b`']


def test_split_drops_empty():
    assert split_statements('\n;\n') == []
    assert split_statements('SELECT 1;\n-- trailing comment;\n/* and; this */\n') == ['SELECT 1']


def test_split_quotes_and_escapes():
    sql = """SELECT 'it''s;' ; SELECT 'a\\';b' ; SELECT "x;y" ; SELECT 2"""
    assert split_statements(sql) == ["SELECT 'it''s;'", "SELECT 'a\\';b'", 'SELECT "x;y"', 'SELECT 2']


@pytest.mark.parametrize('script', SCRIPTS, ids=lambda p: str(p.relative_to(CLICKHOUSE)))
def test_split_real_scripts(script):
    sql = script.read_text()
    statements = split_statements(sql)
    assert statements
    assert all(s.strip() for s in statements)


@pytest.mark.parametrize(
    ('script', 'count'),
    [
        ('scripts/studies.sql', 5),
        ('scripts/credible_sets.sql', 10),  # leading comments belong to the first statement
    ],
)
def test_split_real_script_counts(script, count):
    assert len(split_statements((CLICKHOUSE / script).read_text())) == count
