"""The credible-set read models read MergeTree copies of their inputs, which a last task drops."""

import re
from pathlib import Path

import yaml

from pos.clickhouse_tables import entry_tables
from pos.sql import split_statements

POS = Path(__file__).parents[1]
SCRIPTS = POS / 'config/clickhouse/scripts'
DATASETS = yaml.safe_load((POS / 'config/datasets.yaml').read_text())['clickhouse']
STEP = yaml.safe_load((POS / 'config/config.yaml').read_text())['steps']['backend_clickhouse']
BACKEND = {t['name']: t for t in STEP}
READ_MODELS = ('study_credible_sets', 'target_credible_sets', 'variant_credible_sets')
CLEANUP = 'clickhouse_load clickhouse credible_sets_build_cleanup'
DECLARED_LOADS = 'explode load declared datasets'


def _statements(script: str) -> list[str]:
    return [re.sub(r'\s+', ' ', s) for s in split_statements((POS / script).read_text())]


def _reads(name: str) -> set[str]:
    sql = (SCRIPTS / f'{name}.sql').read_text()
    code = '\n'.join(line for line in sql.splitlines() if not line.lstrip().startswith('--'))
    return set(re.findall(r'\b(\w+)_mergetree\b', code))


def _creates() -> dict[str, str]:
    """Each `<t>_mergetree` table to the script of the input dataset that creates it."""
    found = {}
    for entry in DATASETS.values():
        script = entry.get('postload_script')
        for s in _statements(script) if script else []:
            m = re.search(r'CREATE TABLE IF NOT EXISTS (\w+)_mergetree ENGINE = MergeTree', s)
            if m:
                found[m.group(1)] = script
    return found


def _declared() -> dict[str, str]:
    """Each declared `<t>_mergetree` table to the dataset whose load builds it."""
    found = {}
    for dataset, entry in DATASETS.items():
        tables = entry_tables(entry)
        found |= {t.name.removesuffix('_mergetree'): dataset for t in tables if t.name.endswith('_mergetree')}
    return found


def test_read_models_read_only_mergetree_copies():
    creates = _creates() | _declared()
    for name in READ_MODELS:
        sql = (SCRIPTS / f'{name}.sql').read_text()
        code = '\n'.join(line for line in sql.splitlines() if not line.lstrip().startswith('--'))
        assert _reads(name), name
        for t in _reads(name):
            assert t in creates, f'{name} reads {t}_mergetree, which no input script creates'
            # the RocksDB table itself is never a source of the read model
            assert not re.search(rf'\b(?:FROM|JOIN)\s+{t}\b(?!_)', code), f'{name} reads {t} directly'


def test_rocksdb_tables_are_filled_from_their_mergetree_copy_after_a_key_check():
    for t, script in _creates().items():
        statements = _statements(script)
        i = next(n for n, s in enumerate(statements) if f'CREATE TABLE IF NOT EXISTS {t}_mergetree ' in s)
        check, rocks = statements[i + 1], statements[i + 2]
        assert check.startswith('SELECT throwIf(count() != uniqExact(') and f'FROM {t}_mergetree' in check, t
        m = re.search(r'uniqExact\((\w+)\)', check)
        assert m, t
        key = m.group(1)
        expected = ' '.join([
            'CREATE TABLE IF NOT EXISTS',
            t,
            'ENGINE = EmbeddedRocksDB () PRIMARY KEY',
            key,
            'AS SELECT * FROM',
            t,
        ])
        assert rocks == f'{expected}_mergetree', t


def test_cleanup_drops_every_copy_after_the_read_models():
    statements = _statements(DATASETS['credible_sets_build_cleanup']['preload_script'])
    drops = [re.search(r'DROP TABLE IF EXISTS (\w+)_mergetree SYNC$', s) for s in statements]
    assert all(drops), statements
    dropped = {m.group(1) for m in drops if m}
    assert dropped == set(_creates()) | set(_declared())
    assert set().union(*(_reads(n) for n in READ_MODELS)) == dropped
    assert set(BACKEND[CLEANUP]['requires']) == {f'clickhouse_load clickhouse load {n}' for n in READ_MODELS}
    assert DATASETS['credible_sets_build_cleanup']['derived'] is True


def test_read_models_wait_for_the_inputs_they_read():
    task_of = {}
    for t, script in _creates().items():
        dataset = next(d for d, e in DATASETS.items() if e.get('postload_script') == script)
        task_of[t] = f'clickhouse_load clickhouse load {dataset}'
    task_of |= dict.fromkeys(_declared(), DECLARED_LOADS)
    for name in READ_MODELS:
        requires = set(BACKEND[f'clickhouse_load clickhouse load {name}']['requires'])
        for t in _reads(name):
            assert task_of[t] in requires, f'{name} reads {t}_mergetree without requiring {task_of[t]}'
