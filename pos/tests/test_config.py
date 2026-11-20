"""The one config: its steps, no per-release values, and the paths its tasks share."""

import re
from pathlib import Path

import yaml

from pos.run import PRODUCTS, parse_release_uri

ROOT = Path(__file__).parents[1]
CONFIG = yaml.safe_load((ROOT / 'config' / 'config.yaml').read_text())
RELEASE_NUMBER = re.compile(r'(?<![\d.])\d{2}\.?\d{2}(?![\d.])')
# derived from release_uri by pos.run
DERIVED = set(parse_release_uri('gs://b/platform-2612-1').scratchpad()).union(*PRODUCTS.values())
BACKEND_GUARDS = {'open_search_prepare cluster settings', 'clickhouse_create_database release database'}


def _specs(tasks: list[dict]) -> list[dict]:
    specs = []
    for task in tasks:
        specs.append(task)
        specs.extend(_specs(task.get('do', [])))
    return specs


def _strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _task(step: str, task_type: str) -> dict:
    (spec,) = [s for s in _specs(CONFIG['steps'][step]) if s['name'].split(' ')[0] == task_type]
    return spec


def test_config_steps():
    assert set(CONFIG['steps']) == {'backend', 'tarballs', 'bigquery', 'gcs_sync', 'aws_sync', 'ftp_sync'}


def test_ppp_config_is_gone():
    assert not (ROOT / 'config' / 'config_ppp.yaml').exists()


def test_no_hand_edited_fields():
    assert 'release_uri' not in CONFIG
    assert not DERIVED & set(CONFIG['scratchpad'])
    for value in _strings(CONFIG['scratchpad']):
        assert not RELEASE_NUMBER.search(value), value


def test_no_disk_or_snapshot_fields():
    text = (ROOT / 'config' / 'config.yaml').read_text()
    assert 'snapshot' not in text
    assert 'disk_name' not in text


def test_platform_only_steps():
    for step in ('bigquery', 'aws_sync', 'ftp_sync'):
        assert all(s.get('products') == ['platform'] for s in CONFIG['steps'][step]), step
    for step in ('backend', 'tarballs', 'gcs_sync'):
        assert not any('products' in s and 'ppp' not in s['products'] for s in CONFIG['steps'][step]), step


def test_backend_syncs_output_view_and_export():
    sources = {s['source'] for s in _specs(CONFIG['steps']['backend']) if s['name'].startswith('sync_bucket ')}
    assert sources == {'${data_source}/output', '${data_source}/view', '${data_source}/export'}


def test_croissant_lands_where_opensearch_loads_it():
    croissant = _task('backend', 'ot_croissant')
    load = _task('backend', 'open_search_load')
    assert croissant['prepared_data_parent'] == load['json_parent']


def test_derived_fields_cover_the_run():
    assert {'release', 'data_source', 'bigquery_project', 'ftp_address', 'croissant_ftp_address'} <= DERIVED


def test_backend_syncs_wait_for_the_rerun_guards():
    top = {s['name'] for s in CONFIG['steps']['backend']}
    assert top >= BACKEND_GUARDS
    syncs = [s for s in CONFIG['steps']['backend'] if s['name'].startswith('sync_bucket ')]
    assert len(syncs) == 3
    assert all(set(s['requires']) == BACKEND_GUARDS for s in syncs)
