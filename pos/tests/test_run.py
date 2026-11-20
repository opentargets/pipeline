"""Run derivation from release_uri and its injection into the otter scratchpad."""

import os
import sys

import pytest
import yaml
from otter.task.model import Spec
from otter.task.task_registry import TaskRegistry

from pos import core
from pos.run import PRODUCTS, SCRATCH_PRODUCTS, Run, for_mode, for_product, parse_release_uri, scratch_enabled


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


def test_main_reads_the_package_config_without_c(tmp_path, monkeypatch):
    """The image command has no -c: the config comes from config/config.yaml under the cwd."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.chdir(root)
    argv = ['pos', '-s', 'gcs_sync', '-r', 'gs://x/do/platform-2612-3', '-w', str(tmp_path / 'work')]
    monkeypatch.setattr(sys, 'argv', argv)
    seen = {}

    def stop(self):
        seen['config_path'] = self.config.config_path
        seen['steps'] = [s.name for s in self.specs]
        raise SystemExit(0)

    monkeypatch.setattr(core.Runner, 'start', stop)
    with pytest.raises(SystemExit):
        core.main()
    assert str(seen['config_path']) == os.path.join(root, 'config', 'config.yaml')
    assert seen['steps']


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


def test_for_product_rejects_unknown_product():
    with pytest.raises(ValueError, match='unknown product plataform'):
        for_product([Spec(name='hello_world a', products=['plataform'])], 'platform', 'x')


def test_for_product_rejects_products_on_a_do_child():
    spec = Spec(name='explode x', foreach=['a'], do=[{'name': 'hello_world ${each}', 'products': ['ppp']}])
    with pytest.raises(ValueError, match='only top-level tasks'):
        for_product([spec], 'platform', 'x')


def test_prefix_and_name():
    run = parse_release_uri('gs://open-targets-pipeline-runs/do/platform-2612-3')
    assert (run.prefix, run.name) == ('do', 'platform-2612-3')
    run = parse_release_uri('gs://b/platform-2612-3')
    assert (run.prefix, run.name) == ('', 'platform-2612-3')


def test_scratch_scratchpad_platform():
    run = parse_release_uri('gs://open-targets-pipeline-runs/do/platform-2612-3')
    pad = run.scratchpad(scratch=True)
    assert pad['release_gcs_bucket'] == 'gs://open-targets-pos-scratch/platform'
    assert pad['bigquery_project'] == 'open-targets-eu-dev'
    assert pad['bigquery_dataset'] == 'platform_scratch'
    assert pad['bigquery_staging_dataset'] == 'platform_scratch_staging'
    assert pad['bigquery_location'] == 'eu'
    assert 'aws_address' not in pad
    assert 'ftp_address' not in pad
    assert pad['data_source'] == 'gs://open-targets-pipeline-runs/do/platform-2612-3'
    assert pad['croissant_aws_address'] == 's3://open-targets-public-data-releases/platform/26.12/output/'
    assert pad['croissant_ftp_address'] == run.scratchpad()['croissant_ftp_address']


def test_scratch_scratchpad_ppp():
    pad = parse_release_uri('gs://x/vh/ppp-2609-1').scratchpad(scratch=True)
    assert pad['release_gcs_bucket'] == 'gs://open-targets-pos-scratch/ppp'


def test_scratch_products_mirror_products():
    for product, fixed in PRODUCTS.items():
        assert set(SCRATCH_PRODUCTS[product]) == set(fixed) - {'aws_address', 'ftp_address'}


def test_scratch_release_uri():
    run = parse_release_uri('gs://open-targets-pipeline-runs/do/platform-2612-3')
    assert run.scratch_release_uri() == 'gs://open-targets-pos-scratch/runs/do-platform-2612-3'
    run = parse_release_uri('gs://b/platform-2612-3')
    assert run.scratch_release_uri() == 'gs://open-targets-pos-scratch/runs/platform-2612-3'


def test_scratch_enabled(monkeypatch):
    monkeypatch.delenv('POS_SCRATCH', raising=False)
    assert scratch_enabled() is False
    monkeypatch.setenv('POS_SCRATCH', '')
    assert scratch_enabled() is False
    monkeypatch.setenv('POS_SCRATCH', '1')
    assert scratch_enabled() is True
    monkeypatch.setenv('POS_SCRATCH', 'true')
    with pytest.raises(ValueError, match='POS_SCRATCH'):
        scratch_enabled()


MODE_SPECS = [
    Spec(name='hello_world both'),
    Spec(name='hello_world skip', scratch='skip'),
    Spec(name='hello_world only', scratch='only'),
]


def test_for_mode_normal():
    assert [s.name for s in for_mode(MODE_SPECS, False, 'x')] == ['hello_world both', 'hello_world skip']


def test_for_mode_scratch():
    assert [s.name for s in for_mode(MODE_SPECS, True, 'x')] == ['hello_world both', 'hello_world only']


def test_for_mode_refuses_an_empty_step():
    with pytest.raises(ValueError, match='step aws_sync does not run in scratch mode'):
        for_mode([Spec(name='hello_world a', scratch='skip')], True, 'aws_sync')
    with pytest.raises(ValueError, match='step scratch_cleanup does not run in normal mode'):
        for_mode([Spec(name='hello_world a', scratch='only')], False, 'scratch_cleanup')


def test_for_mode_refuses_a_dropped_requirement():
    specs = [Spec(name='hello_world a', scratch='skip'), Spec(name='hello_world b', requires=['hello_world a'])]
    with pytest.raises(ValueError, match='hello_world b requires hello_world a'):
        for_mode(specs, True, 'x')


def test_for_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match='unknown scratch: yes'):
        for_mode([Spec(name='hello_world a', scratch='yes')], False, 'x')


@pytest.mark.parametrize('key', [{'scratch': 'skip'}, {'products': ['ppp']}])
def test_for_mode_rejects_gating_on_a_do_child(key):
    spec = Spec(name='explode x', foreach=['a'], do=[{'name': 'hello_world ${each}', **key}])
    with pytest.raises(ValueError, match='only top-level tasks'):
        for_mode([spec], False, 'x')


def test_for_product_leaves_an_optional_step_empty():
    assert for_product([Spec(name='hello_world a', products=['platform'])], 'ppp', 'scratch_cleanup') == []
