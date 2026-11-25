"""split_statements splits ClickHouse SQL on top-level semicolons."""

from pathlib import Path

import pytest

from pos.sql import split_statements

# the frozen serving scripts: config/clickhouse holds none since the credible-set read models became declarations
FROZEN = Path(__file__).parent / 'fixtures' / 'serving_scripts_before'
SCRIPTS = sorted(FROZEN.glob('*.sql'))


def test_split_statements():
    sql = "SELECT ';' ; -- a;b\nSELECT 1; /* x;y */ SELECT `a;b`;"
    assert split_statements(sql) == ["SELECT ';'", '-- a;b\nSELECT 1', '/* x;y */ SELECT `a;b`']


def test_split_drops_empty():
    assert split_statements('\n;\n') == []
    assert split_statements('SELECT 1;\n-- trailing comment;\n/* and; this */\n') == ['SELECT 1']


def test_split_quotes_and_escapes():
    sql = """SELECT 'it''s;' ; SELECT 'a\\';b' ; SELECT "x;y" ; SELECT 2"""
    assert split_statements(sql) == ["SELECT 'it''s;'", "SELECT 'a\\';b'", 'SELECT "x;y"', 'SELECT 2']


@pytest.mark.parametrize('script', SCRIPTS, ids=lambda p: p.name)
def test_split_real_scripts(script):
    sql = script.read_text()
    statements = split_statements(sql)
    assert statements
    assert all(s.strip() for s in statements)


@pytest.mark.parametrize(
    ('script', 'count'),
    [
        ('credible_sets.sql', 8),  # leading comments belong to the first statement
    ],
)
def test_split_real_script_counts(script, count):
    assert SCRIPTS
    assert len(split_statements((FROZEN / script).read_text())) == count
