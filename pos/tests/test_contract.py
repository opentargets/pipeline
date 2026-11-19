"""Contract with pts: every dataset POS loads from a pts output is written by pts."""

from fnmatch import fnmatch
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
DATASETS = yaml.safe_load((ROOT / 'config' / 'datasets.yaml').read_text())
PTS_PREFIXES = ('output', 'view', 'export')
# written by pos itself: the croissant doc by ot_croissant, derived tables from loaded ones
POS_WRITTEN = ('croissant', 'derived/')


def _repo_root() -> Path:
    return next(p for p in Path(__file__).resolve().parents if (p / 'pis/config.yaml').is_file())


def _destinations(value, found: set[str]) -> set[str]:
    if isinstance(value, dict):
        for key, v in value.items():
            if key == 'destination':
                found.update(s.rstrip('/') for s in _leaves(v))
            else:
                _destinations(v, found)
    elif isinstance(value, list):
        for v in value:
            _destinations(v, found)
    return found


def _leaves(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _leaves(v)]
    if isinstance(value, list):
        return [s for v in value for s in _leaves(v)]
    return []


PTS_DESTINATIONS = _destinations(yaml.safe_load((_repo_root() / 'pts/config.yaml').read_text())['steps'], set())


def _inputs(*prefixes: str) -> list[tuple[str, str]]:
    """The (section/dataset, input path) pairs read under the prefixes; a glob_pattern dir joins input_dir."""
    pairs = []
    for section in ('opensearch', 'clickhouse'):
        for name, entry in DATASETS[section].items():
            path = entry['input_dir']
            if path in PTS_PREFIXES:
                path = f'{path}/{entry["glob_pattern"].rsplit("/", 1)[0]}'
            if path.split('/')[0] in prefixes:
                pairs.append((f'{section}/{name}', path))
    return pairs


def _written(path: str) -> bool:
    return any(fnmatch(d, path) for d in PTS_DESTINATIONS)


def test_destinations_found():
    assert len(PTS_DESTINATIONS) > 100
    assert 'output/disease' in PTS_DESTINATIONS


def test_no_bigquery_or_export_sections():
    assert set(DATASETS) == {'opensearch', 'clickhouse'}


@pytest.mark.parametrize(('dataset', 'path'), _inputs('output', 'view'), ids=lambda v: v)
def test_contract_output_and_view(dataset, path):
    assert _written(path), f'{dataset} reads {path}, which pts/config.yaml does not write'


@pytest.mark.parametrize(('dataset', 'path'), _inputs('export'), ids=lambda v: v)
def test_contract_export(dataset, path):
    assert _written(path), f'{dataset} reads {path}, which pts/config.yaml does not write'


def test_search_indices_read_export():
    search = {n: e['input_dir'] for n, e in DATASETS['opensearch'].items() if n != 'croissant'}
    assert len(search) == 7
    assert all(p.startswith('export/') for p in search.values()), search


def test_every_input_is_accounted_for():
    for section in ('opensearch', 'clickhouse'):
        for name, entry in DATASETS[section].items():
            path = entry['input_dir']
            assert path.split('/')[0] in PTS_PREFIXES or path.startswith(POS_WRITTEN), f'{section}/{name}: {path}'
