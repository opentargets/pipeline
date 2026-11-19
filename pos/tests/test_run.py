"""Run derivation from release_uri and its injection into the otter scratchpad."""

import os
import sys

import pytest
import yaml
from otter.task.task_registry import TaskRegistry

from pos import core
from pos.run import PRODUCTS, Run, parse_release_uri


def test_parse_platform():
    assert parse_release_uri('gs://open-targets-pipeline-runs/do/platform-2612-3/') == Run(
        'platform', '26.12', 'platform2612', 'gs://open-targets-pipeline-runs/do/platform-2612-3'
    )


def test_parse_ppp_suffix():
    run = parse_release_uri('gs://open-targets-pipeline-runs/vh/ppp-2609-1-rerun')
    assert (run.product, run.release, run.database_namespace) == ('ppp', '26.09', 'ppp2609')


def test_parse_local_path():
    assert parse_release_uri('/tmp/runs/platform-2612-3').release == '26.12'


@pytest.mark.parametrize('uri', ['gs://x/do/2612-3', 'gs://x/do/platform-26-3', 'gs://x/do/foo-2612-3', ''])
def test_parse_rejects(uri):
    with pytest.raises(ValueError, match='<product>-YYMM-N'):
        parse_release_uri(uri)


def test_products():
    assert PRODUCTS['platform'] == {
        'release_gcs_bucket': 'gs://open-targets-data-releases',
        'aws_address': 's3://open-targets-public-data-releases/platform',
        'bigquery_project': 'open-targets-prod',
        'bigquery_dataset': 'platform',
        'bigquery_staging_dataset': 'platform_staging',
        'bigquery_location': 'eu',
    }
    assert PRODUCTS['ppp'] == {'release_gcs_bucket': 'gs://open-targets-ppp-data-releases'}


class _NoManifest:
    def __init__(self, *_):
        pass

    async def update(self, *_):
        pass


def _run_main(tmp_path, monkeypatch, mocker, release_uri):
    config = tmp_path / 'config.yaml'
    config.write_text(
        yaml.safe_dump({
            'scratchpad': {'release': 'stale'},
            'steps': {
                'echo': [
                    {'name': 'hello_world namespace', 'who': '${database_namespace}'},
                    {'name': 'hello_world bucket', 'who': '${release_gcs_bucket}/${release}'},
                    {'name': 'hello_world source', 'who': '${data_source}|${product}'},
                ]
            },
        })
    )
    for var in [v for v in os.environ if v.startswith('POS_')]:
        monkeypatch.delenv(var)
    argv = ['pos', '-s', 'echo', '-c', str(config), '-w', str(tmp_path / 'work'), '-r', release_uri]
    monkeypatch.setattr(sys, 'argv', argv)
    # the manifest is the only remote write
    monkeypatch.setattr('otter.core.Manifest', _NoManifest)
    build = mocker.spy(TaskRegistry, 'build')
    core.main()
    return {t.spec.name: t.spec.who for t in build.spy_return_list}


def test_main_injects_scratchpad(tmp_path, monkeypatch, mocker):
    built = _run_main(tmp_path, monkeypatch, mocker, 'gs://x/vh/ppp-2609-1-rerun/')
    assert built == {
        'hello_world namespace': 'ppp2609',
        'hello_world bucket': 'gs://open-targets-ppp-data-releases/26.09',
        'hello_world source': 'gs://x/vh/ppp-2609-1-rerun|ppp',
    }


def test_main_rejects_bad_run_name(tmp_path, monkeypatch, mocker):
    with pytest.raises(SystemExit) as e:
        _run_main(tmp_path, monkeypatch, mocker, 'gs://x/do/foo-2612-3')
    assert e.value.code != 0
