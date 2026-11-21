"""Tests for the VM that runs pos beside ClickHouse and OpenSearch."""

import re
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from orchestration.operators.gce import OUTPUT_FILTER
from orchestration.operators.gce_backend import COMPACT_COPY, ComputeEngineRunBackendSensor

CLICKHOUSE_IMAGE = 'clickhouse/clickhouse-server:26.8.2.7'
OPENSEARCH_IMAGE = 'opensearchproject/opensearch:3.5.0'
POS_IMAGE = 'europe-west1-docker.pkg.dev/a-project/a-repo/pos:26.12.0'
DISKS = {'clickhouse': 'ch-disk', 'opensearch': 'os-disk'}


def _sensor(**kwargs) -> ComputeEngineRunBackendSensor:
    defaults: dict[str, Any] = {
        'task_id': 'a_step',
        'instance_name': 'a-machine',
        'zone': 'a-zone',
        'container_image': POS_IMAGE,
        'container_args': ['-s', 'backend_clickhouse'],
        'container_env': {'A': '1'},
        'container_secret_files': {'a-secret': '/secrets/a-secret'},
        'work_disk_size_gb': 100,
        'data_disks': DISKS,
        'servers': {'clickhouse', 'opensearch'},
        'read_only': False,
        'clickhouse_image': CLICKHOUSE_IMAGE,
        'opensearch_image': OPENSEARCH_IMAGE,
    }
    return ComputeEngineRunBackendSensor(**{**defaults, **kwargs})


@pytest.fixture
def backend() -> str:
    return _sensor().startup_script()


@pytest.fixture
def tarballs() -> str:
    return _sensor(
        servers=(),
        read_only=True,
        clickhouse_image='',
        opensearch_image='',
        container_args=['-s', 'tarballs'],
    ).startup_script()


def _pos_run(script: str) -> int:
    return script.index('--name workload_container')


def test_backend_script_starts_servers_before_pos(backend: str) -> None:
    ch = backend.index(CLICKHOUSE_IMAGE)
    os_ = backend.index(OPENSEARCH_IMAGE)
    wait = re.search(r'^\s*wait_for_servers$', backend, re.MULTILINE)
    assert wait
    assert ch < os_ < wait.start() < _pos_run(backend)
    for flag in (
        '--ulimit nofile=262144:262144',
        '--ulimit memlock=-1:-1 --ulimit nofile=65536:65536',
        "-e OPENSEARCH_JAVA_OPTS='-Xms31g -Xmx31g'",
        '-e discovery.type=single-node',
        '-e DISABLE_SECURITY_PLUGIN=true',
        '-e bootstrap.memory_lock=true',
        '-e thread_pool.write.queue_size=-1',
        '-e search.insights.top_queries.latency.enabled=false',
        '-e search.insights.top_queries.cpu.enabled=false',
        '-e search.insights.top_queries.memory.enabled=false',
        '-v /mnt/disks/clickhouse/data:/var/lib/clickhouse',
        '-v /mnt/disks/clickhouse/config.d:/etc/clickhouse-server/config.d',
        '-v /mnt/disks/clickhouse/users.d:/etc/clickhouse-server/users.d',
        '-v /mnt/disks/opensearch/data:/usr/share/opensearch/data',
        '--name clickhouse',
        '--name opensearch',
        'chown 1000:1000 /mnt/disks/opensearch/data',
        'mkdir -p /mnt/disks/clickhouse/data',
        'mkdir -p /mnt/disks/opensearch/data',
        'docker cp pos:/app/config/clickhouse/config.d /mnt/disks/clickhouse/',
        'docker cp pos:/app/config/clickhouse/users.d /mnt/disks/clickhouse/',
    ):
        assert flag in backend, flag
    assert backend.index('docker cp pos:') < ch
    for device in DISKS:
        assert f'mkfs.ext4 -F /dev/disk/by-id/google-{device}' in backend
        assert f'/dev/disk/by-id/google-{device} /mnt/disks/{device}' in backend


def test_servers_raise_max_map_count_before_opensearch(backend: str, tarballs: str) -> None:
    sysctl = backend.index('sysctl -w vm.max_map_count=262144')
    assert sysctl < backend.index('docker run -d --name opensearch')
    assert 'max_map_count' not in tarballs
    subprocess.run(['bash', '-n'], input=backend, text=True, check=True)


def test_pos_runs_on_the_host_network_with_work_disk_and_secrets(backend: str) -> None:
    pos = backend[_pos_run(backend) :]
    assert '--network host' in pos
    assert '-v /mnt/disks/work:/mnt/disks/work' in pos
    assert '-v /home/app/secrets/a-secret:/secrets/a-secret' in pos
    assert '-e A=1' in pos
    assert f'{POS_IMAGE} \\' in pos
    assert f'2>&1 | {OUTPUT_FILTER}' in pos
    assert '/mnt/disks/clickhouse' not in pos[: pos.index('rc=')]


def _wait_function(script: str) -> str:
    match = re.search(r'^\s*wait_for_servers\(\) \{\n.*?^\s*\}$', script, re.MULTILINE | re.DOTALL)
    assert match, 'wait_for_servers function not found'
    return match.group(0)


def test_backend_script_health_timeout_exits_nonzero() -> None:
    fn = _wait_function(_sensor(health_timeout_s=2).startup_script())
    assert 'exit 1' in fn
    start = time.monotonic()
    result = subprocess.run(
        ['bash', '-c', f'set -euo pipefail\ncurl() {{ return 7; }}\n{fn}\nwait_for_servers\necho unreachable'],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert time.monotonic() - start < 10
    assert 'unreachable' not in result.stdout
    assert 'healthy' in result.stdout + result.stderr


@pytest.mark.parametrize(
    'health',
    ['{"cluster_name":"x","status":"green"}', '{"cluster_name":"x","status":"yellow"}'],
)
def test_backend_script_health_wait_accepts_green_and_yellow(health: str) -> None:
    fn = _wait_function(_sensor(health_timeout_s=2).startup_script())
    stub = f"""curl() {{ case "$*" in *_cluster/health*) echo '{health}';; *) echo Ok.;; esac; }}"""
    result = subprocess.run(
        ['bash', '-c', f'set -euo pipefail\n{stub}\n{fn}\nwait_for_servers\necho ready'],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert 'ready' in result.stdout


def test_backend_script_health_wait_rejects_red() -> None:
    fn = _wait_function(_sensor(health_timeout_s=1).startup_script())
    stub = """curl() { case "$*" in *_cluster/health*) echo '{"status":"red"}';; *) echo Ok.;; esac; }"""
    result = subprocess.run(
        ['bash', '-c', f'set -euo pipefail\n{stub}\n{fn}\nwait_for_servers\necho ready'],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1


def test_backend_script_quiesces_and_keeps_exit_code(backend: str) -> None:
    pos = _pos_run(backend)
    stop = backend.index('docker stop -t 300 opensearch clickhouse')
    sync = backend.index('sync', stop)
    umount = backend.index('umount /mnt/disks/clickhouse /mnt/disks/opensearch', sync)
    assert pos < stop < sync < umount < backend.rindex('exit "$rc"')
    assert 'rc=${PIPESTATUS[0]}' in backend[pos:stop]
    assert backend.rstrip().endswith('exit "$rc"')
    subprocess.run(['bash', '-n'], input=backend, text=True, check=True)


@pytest.mark.parametrize('pos_rc', [0, 3])
def test_backend_script_quiesces_whatever_pos_returns(backend: str, pos_rc: int) -> None:
    """Run the script from the pos run on, with docker, sudo, sync and umount stubbed."""
    tail = backend[backend.index('set +e') :]
    stubs = f"""
sudo() {{ shift 2; "$@"; }}
docker() {{ if [ "$1" = run ]; then echo pos; return {pos_rc}; fi; echo "docker $1"; }}
sync() {{ echo sync; }}
umount() {{ echo "umount $*"; }}
"""
    result = subprocess.run(
        ['bash', '-c', f'set -euo pipefail\n{stubs}\n{tail}'], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == pos_rc, result.stderr
    assert result.stdout.splitlines() == [
        'pos',
        f'pos exited with {pos_rc}',
        'docker stop',
        'sync',
        'umount /mnt/disks/clickhouse /mnt/disks/opensearch',
    ]


def test_pos_exit_code_survives_set_e_and_the_filter() -> None:
    """The capture idiom the script uses keeps a failing container's code through the filter."""
    workload = "printf 'out\\n'; exit 3"
    script = (
        f'set -euo pipefail\nset +e\n( {workload} ) 2>&1 | {OUTPUT_FILTER}\n'
        'rc=${PIPESTATUS[0]}\nset -e\necho after\nexit "$rc"'
    )
    result = subprocess.run(['bash', '-c', script], capture_output=True, text=True)
    assert result.returncode == 3
    assert result.stdout.splitlines() == ['out', 'after']


def test_tarballs_script_mounts_read_only(tarballs: str) -> None:
    assert 'mkfs.ext4 -F /dev/disk/by-id/google-clickhouse' not in tarballs
    assert 'mkfs.ext4 -F /dev/disk/by-id/google-opensearch' not in tarballs
    for device in DISKS:
        assert f'mount -o ro /dev/disk/by-id/google-{device} /mnt/disks/{device}' in tarballs
    assert 'noload' not in tarballs
    assert 'umount' not in tarballs
    assert 'clickhouse-server' not in tarballs
    assert 'opensearchproject' not in tarballs
    assert 'wait_for_servers' not in tarballs
    assert 'docker stop' not in tarballs
    pos = tarballs[_pos_run(tarballs) :]
    assert '-v /mnt/disks/clickhouse:/mnt/clickhouse:ro' in pos
    assert '-v /mnt/disks/opensearch:/mnt/opensearch:ro' in pos
    assert '--network host' in pos
    assert tarballs.rstrip().endswith('exit "$rc"')
    subprocess.run(['bash', '-n'], input=tarballs, text=True, check=True)


def test_declare_instance_attaches_named_disks() -> None:
    disks = _sensor().declare_instance().disks
    assert [d.device_name for d in disks] == ['', 'work-disk', 'clickhouse', 'opensearch']
    assert disks[0].boot
    data = {d.device_name: d for d in disks[2:]}
    for device, name in DISKS.items():
        assert data[device].source == f'zones/a-zone/disks/{name}'
        assert data[device].auto_delete is False
        assert data[device].mode == 'READ_WRITE'

    ro = _sensor(servers=(), read_only=True).declare_instance().disks
    assert [d.mode for d in ro[2:]] == ['READ_ONLY', 'READ_ONLY']


def test_data_disks_is_a_template_field() -> None:
    assert 'data_disks' in ComputeEngineRunBackendSensor.template_fields


def test_servers_need_both_images() -> None:
    with pytest.raises(ValueError, match='image'):
        _sensor(clickhouse_image='')


def test_servers_cannot_run_on_read_only_disks() -> None:
    with pytest.raises(ValueError, match='read'):
        _sensor(read_only=True)


@pytest.mark.parametrize(
    'disks',
    [{'clickhouse': 'ch-disk'}, {**DISKS, 'extra': 'x-disk'}, {'ch': 'ch-disk', 'opensearch': 'os-disk'}],
)
def test_servers_need_exactly_the_server_disks(disks: dict[str, str]) -> None:
    with pytest.raises(ValueError, match='data_disks'):
        _sensor(data_disks=disks)


COMPACT_DISKS = {'clickhouse': 'ch-build', 'clickhouse-compact': 'ch-disk', 'opensearch': 'os-disk'}
COMPACT = {'clickhouse': 'clickhouse-compact'}
COPY_MOUNTS = '-v /mnt/disks/clickhouse:/src:ro -v /mnt/disks/clickhouse-compact:/dst'
UMOUNT_ALL = 'umount /mnt/disks/clickhouse /mnt/disks/clickhouse-compact /mnt/disks/opensearch'


@pytest.fixture
def compact() -> str:
    return _sensor(data_disks=COMPACT_DISKS, compact=COMPACT).startup_script()


def test_compact_copies_in_the_pos_image_between_stop_and_umount(compact: str) -> None:
    stop = compact.index('docker stop -t 300 opensearch clickhouse')
    copy = compact.index('docker run --rm --network none --entrypoint sh')
    sync = compact.index('sync', copy)
    umount = compact.index(UMOUNT_ALL)
    assert _pos_run(compact) < stop < copy < sync < umount < compact.rindex('exit "$rc"')
    assert compact.rstrip().endswith('exit "$rc"')
    assert COPY_MOUNTS in compact[copy:sync]
    assert f"{POS_IMAGE} -c '{COMPACT_COPY}'" in compact[copy:sync]
    assert 'rsync' not in compact
    for device in COMPACT_DISKS:
        assert f'mkfs.ext4 -F /dev/disk/by-id/google-{device}' in compact
        assert f'mount -o discard,defaults /dev/disk/by-id/google-{device} /mnt/disks/{device}' in compact
    # the servers run on the source disk only
    assert '/mnt/disks/clickhouse-compact' not in compact[compact.index('docker cp') : compact.index('rc=')]
    subprocess.run(['bash', '-n'], input=compact, text=True, check=True)


def test_without_compact_the_script_has_no_copy(backend: str) -> None:
    assert '--entrypoint' not in backend
    assert '/src' not in backend


def _run_tail(script: str, pos_rc: int, copy: str) -> subprocess.CompletedProcess[str]:
    """Run the script from the pos run on, with docker, sudo, sync and umount stubbed; `copy` is the copy container."""
    tail = script[script.index('set +e') :]
    stubs = f"""
sudo() {{ shift 2; "$@"; }}
docker() {{
    case "$*" in
        *--entrypoint*) echo "copy $*" >&2; {copy}
            ;;
        run*) echo pos; return {pos_rc};;
        *) echo "docker $1";;
    esac
}}
sync() {{ echo sync; }}
umount() {{ echo "umount $*"; }}
"""
    return subprocess.run(
        ['bash', '-c', f'set -euo pipefail\n{stubs}\n{tail}'], capture_output=True, text=True, timeout=10
    )


def test_compact_copy_that_succeeds_keeps_pos_exit_code(compact: str) -> None:
    result = _run_tail(compact, 0, 'return 0')
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ['pos', 'pos exited with 0', 'docker stop', 'sync', UMOUNT_ALL]
    [call] = result.stderr.splitlines()
    assert COPY_MOUNTS in call
    assert POS_IMAGE in call


@pytest.mark.parametrize('code', [1, 2])
def test_compact_copy_or_verify_failure_fails(compact: str, code: int) -> None:
    result = _run_tail(compact, 0, f'echo "/dst differs from /src after the copy"; return {code}')
    assert result.returncode == code
    assert f'copying clickhouse onto clickhouse-compact failed with {code}' in result.stdout
    assert result.stdout.splitlines()[-2:] == ['sync', UMOUNT_ALL]


def test_compact_is_skipped_when_pos_fails(compact: str) -> None:
    result = _run_tail(compact, 3, 'return 0')
    assert result.returncode == 3
    assert result.stderr == ''
    assert result.stdout.splitlines() == ['pos', 'pos exited with 3', 'docker stop', 'sync', UMOUNT_ALL]


def _copy_locally(tmp_path: Path, prepare_dst: str = '') -> subprocess.CompletedProcess[str]:
    """Run the copy command with /src and /dst under `tmp_path`."""
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    (src / 'data/store').mkdir(parents=True)
    (src / 'data/store/a.bin').write_text('a')
    (src / 'data/store/b.bin').hardlink_to(src / 'data/store/a.bin')
    (src / 'data/link').symlink_to('store/a.bin')
    dst.mkdir()
    if prepare_dst:
        (dst / prepare_dst).write_text('stale')
    command = COMPACT_COPY.replace('/src', str(src)).replace('/dst', str(dst))
    return subprocess.run(['sh', '-c', command], capture_output=True, text=True, timeout=10)


def test_compact_copy_command_copies_and_verifies(tmp_path: Path) -> None:
    result = _copy_locally(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'dst/data/link').is_symlink()
    assert (tmp_path / 'dst/data/store/b.bin').read_text() == 'a'


def _gnu_cp() -> bool:
    return subprocess.run(['cp', '--version'], capture_output=True, check=False).returncode == 0


@pytest.mark.skipif(not _gnu_cp(), reason="the pos image's GNU cp keeps hard links; BSD cp -a does not")
def test_compact_copy_command_keeps_hard_links(tmp_path: Path) -> None:
    assert _copy_locally(tmp_path).returncode == 0
    dst = tmp_path / 'dst/data/store'
    assert (dst / 'a.bin').stat().st_ino == (dst / 'b.bin').stat().st_ino


def test_compact_copy_command_fails_when_the_destination_differs(tmp_path: Path) -> None:
    result = _copy_locally(tmp_path, prepare_dst='stale.bin')
    assert result.returncode == 1
    assert 'differs' in result.stdout


def test_compact_attaches_the_destination_disk() -> None:
    disks = _sensor(data_disks=COMPACT_DISKS, compact=COMPACT).declare_instance().disks
    assert [d.device_name for d in disks[2:]] == list(COMPACT_DISKS)
    assert disks[3].source == 'zones/a-zone/disks/ch-disk'
    assert disks[3].mode == 'READ_WRITE'


@pytest.mark.parametrize(
    ('disks', 'compact'),
    [
        # destination not attached
        (DISKS, COMPACT),
        # source not attached
        ({**DISKS, 'opensearch-compact': 'x'}, {'unknown': 'opensearch-compact'}),
        # a destination that is itself a server disk
        (DISKS, {'clickhouse': 'opensearch'}),
        # a destination that is its own source
        (COMPACT_DISKS, {'clickhouse-compact': 'clickhouse-compact'}),
        # an extra disk that is neither a server disk nor a destination
        ({**COMPACT_DISKS, 'extra': 'x'}, COMPACT),
    ],
)
def test_compact_devices_must_be_data_disks(disks: dict[str, str], compact: dict[str, str]) -> None:
    with pytest.raises(ValueError, match=r'data_disks|compact'):
        _sensor(data_disks=disks, compact=compact)


def test_compact_needs_writable_disks() -> None:
    with pytest.raises(ValueError, match='compact'):
        _sensor(servers=(), read_only=True, data_disks=COMPACT_DISKS, compact=COMPACT)


def test_servers_must_be_known() -> None:
    with pytest.raises(ValueError, match='redis'):
        _sensor(servers={'redis'})


def test_server_needs_its_image() -> None:
    with pytest.raises(ValueError, match='image'):
        _sensor(servers={'clickhouse'}, data_disks={'clickhouse': 'a'}, opensearch_image='', clickhouse_image='')
    _sensor(servers={'opensearch'}, data_disks={'opensearch': 'd'}, clickhouse_image='')


def test_server_disks_match() -> None:
    _sensor(servers={'opensearch'}, data_disks={'opensearch': 'd'})
    with pytest.raises(ValueError, match='data_disks'):
        _sensor(servers={'opensearch'}, data_disks={'opensearch': 'd', 'clickhouse': 'e'})
    _sensor(
        servers={'clickhouse'},
        data_disks={'clickhouse': 'a', 'clickhouse-compact': 'b'},
        compact={'clickhouse': 'clickhouse-compact'},
    )


def test_clickhouse_only_script() -> None:
    script = _sensor(
        servers={'clickhouse'},
        data_disks={'clickhouse': 'a', 'clickhouse-compact': 'b'},
        compact={'clickhouse': 'clickhouse-compact'},
        opensearch_image='',
    ).startup_script()
    for text in (
        'docker run -d --name clickhouse',
        'docker cp pos:/app/config/clickhouse/config.d',
        'localhost:8123/ping',
        'docker stop -t 300 clickhouse',
    ):
        assert text in script, text
    assert script.index('docker stop') < script.index('--entrypoint sh')
    for text in ('opensearch', '9200', 'vm.max_map_count'):
        assert text not in script, text
    subprocess.run(['bash', '-n'], input=script, text=True, check=True)


def test_opensearch_only_script() -> None:
    script = _sensor(
        servers={'opensearch'},
        data_disks={'opensearch': 'd'},
        clickhouse_image='',
        container_args=['-s', 'backend_opensearch'],
    ).startup_script()
    for text in (
        'vm.max_map_count=262144',
        '-Xms31g -Xmx31g',
        'localhost:9200/_cluster/health',
        'docker stop -t 300 opensearch',
    ):
        assert text in script, text
    for text in ('clickhouse', '8123', 'docker cp pos:'):
        assert text not in script, text
    subprocess.run(['bash', '-n'], input=script, text=True, check=True)
