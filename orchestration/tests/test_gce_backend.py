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
