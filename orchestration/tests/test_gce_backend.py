"""Tests for the VM that runs pos beside ClickHouse and OpenSearch."""

import re
import subprocess
import time
from typing import Any

import pytest

from orchestration.operators.gce import OUTPUT_FILTER
from orchestration.operators.gce_backend import ComputeEngineRunBackendSensor

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
        'container_args': ['-s', 'backend'],
        'container_env': {'A': '1'},
        'container_secret_files': {'a-secret': '/secrets/a-secret'},
        'work_disk_size_gb': 100,
        'data_disks': DISKS,
        'servers': True,
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
        servers=False,
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
        'mkdir -p /mnt/disks/clickhouse/data /mnt/disks/opensearch/data',
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

    ro = _sensor(servers=False, read_only=True).declare_instance().disks
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
SRC_DST = '/mnt/disks/clickhouse/ /mnt/disks/clickhouse-compact/'
COPY = f'rsync -aHAX --numeric-ids --delete {SRC_DST}'
VERIFY = f'rsync -aHAX --numeric-ids --delete --dry-run --itemize-changes {SRC_DST}'
UMOUNT_ALL = 'umount /mnt/disks/clickhouse /mnt/disks/clickhouse-compact /mnt/disks/opensearch'


@pytest.fixture
def compact() -> str:
    return _sensor(data_disks=COMPACT_DISKS, compact=COMPACT).startup_script()


def test_compact_copies_and_verifies_between_stop_and_umount(compact: str) -> None:
    stop = compact.index('docker stop -t 300 opensearch clickhouse')
    copy = compact.index(COPY)
    verify = compact.index(VERIFY)
    sync = compact.index('sync', verify + len(VERIFY))
    umount = compact.index(UMOUNT_ALL)
    assert _pos_run(compact) < stop < copy < verify < sync < umount < compact.rindex('exit "$rc"')
    assert compact.rstrip().endswith('exit "$rc"')
    for device in COMPACT_DISKS:
        assert f'mkfs.ext4 -F /dev/disk/by-id/google-{device}' in compact
        assert f'mount -o discard,defaults /dev/disk/by-id/google-{device} /mnt/disks/{device}' in compact
    # the servers run on the source disk only
    assert '/mnt/disks/clickhouse-compact' not in compact[compact.index('docker cp') : compact.index('rc=')]
    subprocess.run(['bash', '-n'], input=compact, text=True, check=True)


def test_without_compact_the_script_has_no_copy(backend: str) -> None:
    assert 'rsync' not in backend


def _run_tail(script: str, pos_rc: int, rsync: str) -> subprocess.CompletedProcess[str]:
    """Run the script from the pos run on, with docker, sudo, sync, umount and rsync stubbed."""
    tail = script[script.index('set +e') :]
    stubs = f"""
sudo() {{ shift 2; "$@"; }}
docker() {{ if [ "$1" = run ]; then echo pos; return {pos_rc}; fi; echo "docker $1"; }}
sync() {{ echo sync; }}
umount() {{ echo "umount $*"; }}
rsync() {{ {rsync}
}}
"""
    return subprocess.run(
        ['bash', '-c', f'set -euo pipefail\n{stubs}\n{tail}'], capture_output=True, text=True, timeout=10
    )


def test_compact_copy_and_clean_verify_keep_pos_exit_code(compact: str) -> None:
    result = _run_tail(compact, 0, 'echo "rsync $*" >&2')
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-2:] == ['sync', UMOUNT_ALL]
    assert 'docker stop' in result.stdout
    assert result.stderr.splitlines() == [COPY, VERIFY]


def test_compact_verify_with_changes_fails(compact: str) -> None:
    result = _run_tail(compact, 0, """case "$*" in *--dry-run*) echo '>f+++++++++ data/store/a.bin';; esac""")
    assert result.returncode != 0
    assert 'data/store/a.bin' in result.stdout
    assert 'differs' in result.stdout
    assert result.stdout.splitlines()[-2:] == ['sync', UMOUNT_ALL]


@pytest.mark.parametrize('failing', ['*--dry-run*', '*'])
def test_compact_rsync_failure_fails(compact: str, failing: str) -> None:
    result = _run_tail(compact, 0, f'case "$*" in {failing}) return 23;; esac')
    assert result.returncode == 23
    assert 'failed' in result.stdout
    assert result.stdout.splitlines()[-2:] == ['sync', UMOUNT_ALL]


def test_compact_is_skipped_when_pos_fails(compact: str) -> None:
    result = _run_tail(compact, 3, 'echo called >&2')
    assert result.returncode == 3
    assert 'called' not in result.stderr
    assert result.stdout.splitlines() == ['pos', 'pos exited with 3', 'docker stop', 'sync', UMOUNT_ALL]


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
        _sensor(servers=False, read_only=True, data_disks=COMPACT_DISKS, compact=COMPACT)
