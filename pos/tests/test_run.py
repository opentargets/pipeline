"""Run derivation from release_uri and its injection into the otter scratchpad."""

import os
import sys

import pytest
import yaml
from otter.task.model import Spec
from otter.task.task_registry import TaskRegistry

from pos import core
from pos.run import PRODUCTS, Run, for_product, parse_release_uri


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
        'ftp_address': 'https://ftp.ebi.ac.uk/pub/databases/opentargets/platform',
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


ECHO = [
    {'name': 'hello_world namespace', 'who': '${database_namespace}'},
    {'name': 'hello_world bucket', 'who': '${release_gcs_bucket}/${release}'},
    {'name': 'hello_world source', 'who': '${data_source}|${product}'},
]


def _run_main(tmp_path, monkeypatch, mocker, release_uri, tasks=ECHO):
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'scratchpad': {'release': 'stale'}, 'steps': {'echo': tasks}}))
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


def test_croissant_addresses():
    platform = parse_release_uri('gs://x/do/platform-2612-3').scratchpad()
    assert platform['croissant_ftp_address'] == 'https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/26.12/output/'
    assert platform['croissant_aws_address'] == 's3://open-targets-public-data-releases/platform/26.12/output/'
    ppp = parse_release_uri('gs://x/vh/ppp-2609-1').scratchpad()
    assert (ppp['croissant_ftp_address'], ppp['croissant_aws_address']) == ('', '')
    assert 'aws_address' not in ppp


def test_for_product_drops_other_products_tasks():
    specs = [
        Spec(name='hello_world a'),
        Spec(name='hello_world b', products=['ppp']),
        Spec(name='hello_world c', products=['platform']),
    ]
    assert [s.name for s in for_product(specs, 'platform', 'x')] == ['hello_world a', 'hello_world c']
    assert [s.name for s in for_product(specs, 'ppp', 'x')] == ['hello_world a', 'hello_world b']


def test_for_product_refuses_a_step_with_no_task_for_it():
    with pytest.raises(ValueError, match='step aws_sync does not run for ppp'):
        for_product([Spec(name='hello_world a', products=['platform'])], 'ppp', 'aws_sync')


def test_for_product_refuses_a_dropped_requirement():
    specs = [Spec(name='hello_world a', products=['ppp']), Spec(name='hello_world b', requires=['hello_world a'])]
    with pytest.raises(ValueError, match='hello_world b requires hello_world a'):
        for_product(specs, 'platform', 'x')


def test_main_runs_only_the_products_tasks(tmp_path, monkeypatch, mocker):
    tasks = [{'name': 'hello_world all', 'who': 'a'}, {'name': 'hello_world ppp', 'who': 'p', 'products': ['ppp']}]
    assert _run_main(tmp_path, monkeypatch, mocker, 'gs://x/do/platform-2612-3', tasks) == {'hello_world all': 'a'}


def test_main_refuses_platform_only_step_for_ppp(tmp_path, monkeypatch, mocker):
    tasks = [{'name': 'hello_world a', 'who': 'a', 'products': ['platform']}]
    with pytest.raises(SystemExit) as e:
        _run_main(tmp_path, monkeypatch, mocker, 'gs://x/vh/ppp-2609-1', tasks)
    assert e.value.code != 0
