"""rclone bucket syncs and the packaged FTP script."""

from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from otter.task.model import Spec

from pos.rclone.sync import strip_scheme
from pos.run import for_product
from pos.tasks.sync_bucket import SyncBucket, SyncBucketSpec


def _run_sync_bucket(tmp_path, mocker, source, destination):
    run = mocker.patch('subprocess.run')
    context = SimpleNamespace(config=SimpleNamespace(work_path=tmp_path, pool_size=7))
    task = SyncBucket(SyncBucketSpec(name='sync_bucket x', source=source, destination=destination), context)  # ty: ignore[invalid-argument-type]
    task.run.__wrapped__(task)  # ty: ignore[unresolved-attribute]
    return run


def test_sync_bucket_command(tmp_path, mocker):
    run = _run_sync_bucket(tmp_path, mocker, 'gs://bucket/x/output', 'release_data/output')
    argv = run.call_args.args[0]
    assert argv == [
        'rclone', '--config', 'config/rclone.conf', 'copy', 'gcs:bucket/x/output',
        str(tmp_path / 'release_data/output'), '--transfers', '7', '--checkers', '7',
    ]  # fmt: skip
    assert not any(a == 'sync' or a.startswith('--delete') for a in argv)
    assert run.call_args.kwargs['check'] is True


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
