"""tarball streams tar | pigz | rclone rcat, keeps the member layout and fails on any stage."""

import shutil
import subprocess
import tarfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pos.tasks.tarball import Tarball, TarballError, TarballSpec, tarball_commands

ROOT = Path(__file__).parents[1]

# stand-ins on PATH: pigz is gzip; rclone rcat writes stdin to its last argument
FAKE_PIGZ = """#!/bin/sh
echo "$@" > "$FAKE_LOG/pigz.args"
[ -n "$PIGZ_EXIT" ] && exit "$PIGZ_EXIT"
exec gzip -c
"""
FAKE_RCLONE = """#!/bin/sh
echo "$@" > "$FAKE_LOG/rclone.args"
if [ -n "$RCLONE_EXIT" ]; then echo "ERROR : upload refused" >&2; exit "$RCLONE_EXIT"; fi
for last; do :; done
mkdir -p "$(dirname "$last")"
cat > "$last"
"""


@pytest.fixture
def disk(tmp_path) -> Path:
    # the ClickHouse disk root layout standalone mounts from
    root = tmp_path / 'clickhouse'
    (root / 'users.d').mkdir(parents=True)
    (root / 'users.d' / 'users.xml').write_text('<clickhouse/>\n')
    (root / 'config.d').mkdir()
    (root / 'config.d' / 'config.xml').write_text('<clickhouse/>\n')
    (root / 'data' / 'ot' / 'part').mkdir(parents=True)
    (root / 'data' / 'ot' / 'part' / 'data.bin').write_bytes(bytes(range(256)) * 64)
    return root


@pytest.fixture
def fakes(tmp_path, monkeypatch) -> Path:
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for name, script in (('pigz', FAKE_PIGZ), ('rclone', FAKE_RCLONE)):
        (bin_dir / name).write_text(script)
        (bin_dir / name).chmod(0o755)
    log = tmp_path / 'log'
    log.mkdir()
    monkeypatch.setenv('PATH', f'{bin_dir}:{Path("/usr/bin")}:{Path("/bin")}')
    monkeypatch.setenv('FAKE_LOG', str(log))
    monkeypatch.delenv('PIGZ_EXIT', raising=False)
    monkeypatch.delenv('RCLONE_EXIT', raising=False)
    return log


def _task(tmp_path, source: Path, release_uri: str | None) -> Tarball:
    config = MagicMock(work_path=tmp_path / 'work', release_uri=release_uri)
    destination = Path('disk_images/clickhouse.tgz')
    spec = TarballSpec(name='tarball clickhouse', source=source, destination=destination, threads=3)
    return Tarball(spec, MagicMock(config=config))


def _run(task: Tarball) -> Tarball:
    return Tarball.run.__wrapped__(task)  # ty: ignore[unresolved-attribute]


def _members(tgz: Path) -> list[str]:
    with tarfile.open(tgz) as t:
        return sorted(t.getnames())


def test_commands():
    assert tarball_commands('/mnt/clickhouse', 'gs://b/platform-2612-3/disk_images/clickhouse.tgz', 8) == [
        ['tar', '-C', '/mnt/clickhouse', '-cf', '-', '.'],
        ['pigz', '-p', '8'],
        ['rclone', '--config', 'config/rclone.conf', 'rcat', 'gcs:b/platform-2612-3/disk_images/clickhouse.tgz'],
    ]


def test_streams_with_the_old_layout(tmp_path, disk, fakes):
    release = tmp_path / 'platform-2612-3'
    task = _run(_task(tmp_path, disk, str(release)))
    tgz = release / 'disk_images' / 'clickhouse.tgz'
    old = tmp_path / 'old.tgz'
    subprocess.run(['tar', '-czf', str(old), '-C', str(disk), '.'], check=True)
    assert _members(tgz) == _members(old)
    assert {'./users.d/users.xml', './config.d/config.xml', './data/ot/part/data.bin'} <= set(_members(tgz))
    with tarfile.open(tgz) as t:
        member = t.extractfile('./data/ot/part/data.bin')
        assert member
        assert member.read() == bytes(range(256)) * 64
    assert (fakes / 'pigz.args').read_text().split() == ['-p', '3']
    assert (fakes / 'rclone.args').read_text().split() == ['--config', 'config/rclone.conf', 'rcat', str(tgz)]
    assert not (tmp_path / 'work' / 'disk_images' / 'clickhouse.tgz').exists()
    assert task.artifacts
    assert task.artifacts[0].destination == str(tgz)


def test_no_release_uri_writes_the_work_dir(tmp_path, disk, fakes):
    _run(_task(tmp_path, disk, None))
    tgz = tmp_path / 'work' / 'disk_images' / 'clickhouse.tgz'
    assert './users.d/users.xml' in _members(tgz)


@pytest.mark.parametrize(('stage', 'env'), [('pigz', 'PIGZ_EXIT'), ('rclone', 'RCLONE_EXIT')])
def test_any_failing_stage_fails(tmp_path, disk, fakes, monkeypatch, stage, env):
    monkeypatch.setenv(env, '3')
    with pytest.raises(TarballError, match=f'{stage} exited 3'):
        _run(_task(tmp_path, disk, str(tmp_path / 'platform-2612-3')))


def test_rclone_stderr_in_the_error(tmp_path, disk, fakes, monkeypatch):
    monkeypatch.setenv('RCLONE_EXIT', '1')
    with pytest.raises(TarballError, match='upload refused'):
        _run(_task(tmp_path, disk, str(tmp_path / 'platform-2612-3')))


def test_tar_error_reported(tmp_path, disk, fakes, monkeypatch):
    from pos.tasks import tarball

    real = tarball.tarball_commands

    def broken(source, destination, threads):
        commands = real(source, destination, threads)
        commands[0] = ['tar', '-C', str(tmp_path / 'nowhere'), '-cf', '-', '.']
        return commands

    monkeypatch.setattr(tarball, 'tarball_commands', broken)
    with pytest.raises(TarballError, match='tar exited'):
        _run(_task(tmp_path, disk, str(tmp_path / 'platform-2612-3')))


@pytest.mark.skipif(not (shutil.which('rclone') and shutil.which('pigz')), reason='needs rclone and pigz')
def test_real_rclone_local_destination(tmp_path, disk, monkeypatch):
    monkeypatch.chdir(ROOT)
    release = tmp_path / 'platform-2612-3'
    _run(_task(tmp_path, disk, str(release)))
    assert './users.d/users.xml' in _members(release / 'disk_images' / 'clickhouse.tgz')
