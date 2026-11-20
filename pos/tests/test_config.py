"""The one config: its steps, no per-release values, and the paths its tasks share."""

import os
import re
import sys
from pathlib import Path

import pytest
import yaml
from loguru import logger

from pos import core
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
    assert set(CONFIG['steps']) == {
        'backend', 'tarballs', 'bigquery', 'gcs_sync', 'aws_sync', 'ftp_sync', 'publish_manifest', 'scratch_cleanup',
    }  # fmt: skip


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


@pytest.fixture
def load(tmp_path, monkeypatch):
    """Set up the runner for a step of the real config: its specs as the registry builds them, and the log."""
    monkeypatch.setattr('otter.core.init_logger_early', lambda *_: None)
    monkeypatch.setattr('otter.core.init_logger', lambda *_: None)
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(str(m)), level='INFO')

    def _load(step: str, run: str = 'platform-2612-3', scratch: bool = False):
        for var in [v for v in os.environ if v.startswith('POS_')]:
            monkeypatch.delenv(var)
        if scratch:
            monkeypatch.setenv('POS_SCRATCH', '1')
        config = str(ROOT / 'config' / 'config.yaml')
        uri = f'gs://open-targets-pipeline-runs/do/{run}'
        argv = ['pos', '-s', step, '-c', config, '-w', str(tmp_path / 'work'), '-r', uri]
        monkeypatch.setattr(sys, 'argv', argv)
        runner = core.setup()
        return runner, [runner.task_registry.build(s).spec for s in runner.specs]

    yield _load, lines
    logger.remove(sink)


def test_publish_manifest_and_scratch_cleanup_resolve(load, mocker):
    _load, lines = load
    _, specs = _load('publish_manifest')
    assert [(s.source, s.destination) for s in specs] == [
        ('gs://open-targets-pipeline-runs/do/platform-2612-3/manifest.json',
         'gs://open-targets-data-releases/26.12/manifest.json'),
        ('gs://open-targets-pipeline-runs/do/platform-2612-3/manifest.json',
         's3://open-targets-public-data-releases/platform/26.12/manifest.json'),
    ]  # fmt: skip

    _, specs = _load('publish_manifest', scratch=True)
    assert [(s.source, s.destination) for s in specs] == [
        ('gs://open-targets-pos-scratch/runs/do-platform-2612-3/manifest.json',
         'gs://open-targets-pos-scratch/platform/26.12/manifest.json'),
    ]  # fmt: skip

    with pytest.raises(SystemExit):
        _load('aws_sync', scratch=True)
    assert any('step aws_sync does not run in scratch mode' in line for line in lines)

    with pytest.raises(SystemExit):
        _load('scratch_cleanup')
    assert any('step scratch_cleanup does not run in normal mode' in line for line in lines)

    _, specs = _load('scratch_cleanup', scratch=True)
    assert [(s.project_id, s.dataset_ids) for s in specs] == [
        ('open-targets-eu-dev', ['platform_scratch', 'platform_scratch_staging']),
    ]

    runner, specs = _load('scratch_cleanup', run='ppp-2612-1', scratch=True)
    assert specs == []
    run = mocker.patch.object(core.Runner, 'run')
    mocker.patch.object(core, 'setup', return_value=runner)
    core.main()
    run.assert_not_called()
    assert any('step scratch_cleanup does not apply to ppp' in line for line in lines)


def test_ftp_and_aws_skip_scratch():
    for step in ('aws_sync', 'ftp_sync'):
        assert all(s.get('scratch') == 'skip' for s in CONFIG['steps'][step]), step

