"""Scratch mode in the entry point: the run location and the destinations are swapped."""

import os
import sys

import pytest
import yaml
from loguru import logger

from pos import core
from pos.run import SCRATCH_PRODUCTS

RELEASE_URI = 'gs://open-targets-pipeline-runs/do/platform-2612-3'
SCRATCH_URI = 'gs://open-targets-pos-scratch/runs/do-platform-2612-3'


@pytest.fixture
def setup(tmp_path, monkeypatch):
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'steps': {'echo': [{'name': 'hello_world a', 'who': '${data_source}'}]}}))
    for var in [v for v in os.environ if v.startswith('POS_')]:
        monkeypatch.delenv(var)
    argv = ['pos', '-s', 'echo', '-c', str(config), '-w', str(tmp_path / 'work'), '-r', RELEASE_URI]
    monkeypatch.setattr(sys, 'argv', argv)
    # keep otter from replacing the test's sink
    monkeypatch.setattr('otter.core.init_logger_early', lambda *_: None)
    monkeypatch.setattr('otter.core.init_logger', lambda *_: None)
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(str(m)), level='DEBUG')
    yield lambda: (core.setup(), lines)
    logger.remove(sink)


def test_scratch_replaces_release_uri(setup, monkeypatch):
    monkeypatch.setenv('POS_SCRATCH', '1')
    runner, _ = setup()
    assert runner.config.release_uri == SCRATCH_URI
    assert runner.task_registry.config is runner.config
    assert runner.scratchpad.sentinel_dict['data_source'] == RELEASE_URI
    assert runner.scratchpad.sentinel_dict['release_gcs_bucket'] == 'gs://open-targets-pos-scratch/platform'
    assert 'hello_world' in runner.task_registry._tasks


def test_scratch_logs_banner(setup, monkeypatch):
    monkeypatch.setenv('POS_SCRATCH', '1')
    _, lines = setup()
    banner = [line for line in lines if 'SCRATCH MODE' in line]
    assert len(banner) == 1
    for destination in [SCRATCH_URI, *SCRATCH_PRODUCTS['platform'].values()]:
        assert destination in banner[0]


def test_no_scratch_unchanged(setup):
    runner, lines = setup()
    assert runner.config.release_uri == RELEASE_URI
    assert runner.scratchpad.sentinel_dict['release_gcs_bucket'] == 'gs://open-targets-data-releases'
    assert not [line for line in lines if 'SCRATCH MODE' in line]


def test_scratch_rejects_bad_value(setup, monkeypatch):
    monkeypatch.setenv('POS_SCRATCH', 'true')
    with pytest.raises(SystemExit) as e:
        setup()
    assert e.value.code != 0
