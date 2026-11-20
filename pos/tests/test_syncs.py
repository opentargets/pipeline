"""rclone bucket syncs and the packaged FTP script."""

import configparser
import subprocess
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from loguru import logger
from otter.task.model import Spec

from pos.rclone.sync import RCLONE_CONFIG, RcloneError, Sync, remote, strip_scheme
from pos.run import for_product
from pos.tasks.rclone_copyto import RcloneCopyto, RcloneCopytoSpec
from pos.tasks.sync_bucket import SyncBucket, SyncBucketSpec

STATS = ['--stats', '1m', '--stats-one-line', '--stats-log-level', 'NOTICE']


def _rclone(mocker, returncode=0, stderr=''):
    return mocker.patch('subprocess.run', return_value=subprocess.CompletedProcess([], returncode, '', stderr))


def _run_sync_bucket(tmp_path, mocker, source, destination, returncode=0, stderr=''):
    run = _rclone(mocker, returncode, stderr)
    context = SimpleNamespace(config=SimpleNamespace(work_path=tmp_path, pool_size=7))
    task = SyncBucket(SyncBucketSpec(name='sync_bucket x', source=source, destination=destination), context)  # ty: ignore[invalid-argument-type]
    task.run.__wrapped__(task)  # ty: ignore[unresolved-attribute]
    return run


def test_sync_bucket_command(tmp_path, mocker):
    run = _run_sync_bucket(tmp_path, mocker, 'gs://bucket/x/output', 'release_data/output')
    argv = run.call_args.args[0]
    assert argv == [
        'rclone', '--config', 'config/rclone.conf', 'copy', 'gcs:bucket/x/output',
        str(tmp_path / 'release_data/output'), '--transfers', '7', '--checkers', '7', *STATS,
    ]  # fmt: skip
    assert not any(a == 'sync' or a.startswith('--delete') for a in argv)
    assert run.call_args.kwargs['capture_output'] is True


@pytest.fixture
def messages():
    seen: list[str] = []
    sink = logger.add(lambda m: seen.append(m.record['message']), level='INFO')
    yield seen
    logger.remove(sink)


def test_sync_bucket_logs_a_summary(tmp_path, mocker, messages):
    stats = 'NOTICE: 1.2 GiB / 1.2 GiB, 100%, 50 MiB/s, ETA 0s'
    _run_sync_bucket(tmp_path, mocker, 'gs://a/r/', 'gs://b/26.12/', stderr=f'NOTICE: earlier\n{stats}\n')
    (summary,) = [m for m in messages if m.startswith('rclone copy')]
    assert summary.endswith(stats)
    assert 'gs://a/r/' in summary


def test_sync_bucket_raises_with_exit_code_and_stderr_tail(tmp_path, mocker):
    stderr = '\n'.join(f'ERROR : line {i}' for i in range(80))
    with pytest.raises(RcloneError, match='exited 3') as e:
        _run_sync_bucket(tmp_path, mocker, 'gs://a/r/', 'gs://b/26.12/', returncode=3, stderr=stderr)
    message = str(e.value)
    assert 'ERROR : line 79' in message
    assert 'ERROR : line 30' in message
    assert 'ERROR : line 29' not in message


def test_rclone_sync_raises_with_stderr(tmp_path, mocker):
    run = _rclone(mocker, 1, 'ERROR : AccessDenied\n')
    sync = Sync(Path('gcs:a/r/'), Path('aws:b/26.12/'), Path('config/rclone.conf'), Path('config/aws_sync.txt'))
    with pytest.raises(RcloneError, match=r'(?s)exited 1.*AccessDenied'):
        sync.sync(sync_flags=[])
    argv = run.call_args.args[0]
    assert argv[3:6] == ['sync', 'gcs:a/r', 'aws:b/26.12']
    assert argv[6:11] == STATS


def test_sync_bucket_gcs_to_gcs(tmp_path, mocker):
    argv = _run_sync_bucket(tmp_path, mocker, 'gs://a/r/', 'gs://b/26.12/').call_args.args[0]
    assert argv[argv.index('copy') + 1 :][:2] == ['gcs:a/r/', 'gcs:b/26.12/']


def test_ftp_script_packaged():
    assert files('pos.scripts').joinpath('ftp_sync.sh').is_file()


def test_aws_sync_refuses_ppp():
    config = yaml.safe_load((Path(__file__).parents[1] / 'config' / 'config.yaml').read_text())
    specs = [Spec.model_validate(t) for t in config['steps']['aws_sync']]
    with pytest.raises(ValueError, match='step aws_sync does not run for ppp'):
        for_product(specs, 'ppp', 'aws_sync')


def test_strip_scheme_keeps_the_rclone_remote():
    assert strip_scheme('gcs:gs://b/x/') == 'gcs:b/x/'
    assert strip_scheme('aws:s3://b/x/') == 'aws:b/x/'


def test_gcs_remote_sets_no_acl_or_class():
    """A predefined ACL fails on UBLA buckets and overrides the default ACL on the others."""
    conf = configparser.ConfigParser()
    conf.read(Path(__file__).parents[1] / 'config' / 'rclone.conf')
    gcs = conf['gcs']
    assert gcs.getboolean('bucket_policy_only')
    assert not {'object_acl', 'bucket_acl', 'storage_class', 'location'} & set(gcs)


def test_remote_maps_s3_to_aws():
    assert remote('s3://c/d/manifest.json') == 'aws:c/d/manifest.json'
    assert remote('gs://a/b') == 'gcs:a/b'
    assert remote('release_data/output') == 'release_data/output'


def test_rclone_copyto_command(mocker):
    run = mocker.patch('pos.tasks.rclone_copyto.run_rclone')
    spec = RcloneCopytoSpec(name='rclone_copyto x', source='gs://a/b/manifest.json', destination='s3://c/d/manifest.json')
    task = RcloneCopyto(spec, SimpleNamespace())  # ty: ignore[invalid-argument-type]
    task.run.__wrapped__(task)  # ty: ignore[unresolved-attribute]
    argv = run.call_args.args[0]
    assert argv[:6] == ['rclone', '--config', RCLONE_CONFIG, 'copyto', 'gcs:a/b/manifest.json', 'aws:c/d/manifest.json']
    assert argv[6:] == STATS
