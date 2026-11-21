"""A step VM that runs pos beside ClickHouse and OpenSearch, or on their disks read-only."""

from __future__ import annotations

from collections.abc import Sequence

from google.cloud import compute_v1

from orchestration.operators.gce import CONTAINER_NAME, OUTPUT_FILTER, ComputeEngineRunContainerizedWorkloadSensor

HEALTH_POLL_S = 5
SERVER_DEVICES = {'clickhouse', 'opensearch'}
# run in the pos image with the source at /src (read-only) and the destination at /dst
COMPACT_COPY = (
    'cp -a /src/. /dst/ || { e=$?; echo "cp failed"; exit $e; }; '
    'diff -rq --no-dereference /src /dst || { echo "/dst differs from /src after the copy"; exit 1; }'
)


class ComputeEngineRunBackendSensor(ComputeEngineRunContainerizedWorkloadSensor):
    """Runs pos on a VM with existing data disks attached, optionally beside the servers.

    Each data disk is mounted on the host at `/mnt/disks/<device>` (COS only allows
    writes under `/mnt/disks`).

    Args:
        data_disks: Device name (`clickhouse`, `opensearch`) to the name of an existing disk.
        servers: Start ClickHouse and OpenSearch on the data disks before pos, and stop
            them after it.
        read_only: Attach the data disks read-only and mount them into pos at `/mnt/<device>`.
            Otherwise they are formatted.
        clickhouse_image: ClickHouse server image, needed with `servers`.
        opensearch_image: OpenSearch image, needed with `servers`.
        health_timeout_s: How long to wait for both servers to be healthy. On timeout the
            script exits 1 with the servers still running, so the disks are unfit to snapshot.
        compact: Source device to destination device, both in `data_disks`. If pos exits 0,
            each source is copied onto its destination after the servers stop, and the copy is
            verified, in a container of `container_image`; a failed copy or verify fails the script.
        **kwargs: Passed to `ComputeEngineRunContainerizedWorkloadSensor`.
    """

    template_fields: Sequence[str] = (
        *ComputeEngineRunContainerizedWorkloadSensor.template_fields,
        'data_disks',
        'clickhouse_image',
        'opensearch_image',
    )

    def __init__(
        self,
        *,
        data_disks: dict[str, str],
        servers: bool,
        read_only: bool,
        clickhouse_image: str = '',
        opensearch_image: str = '',
        health_timeout_s: int = 600,
        compact: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if servers and not (clickhouse_image and opensearch_image):
            raise ValueError('servers need clickhouse_image and opensearch_image')
        if servers and read_only:
            raise ValueError('servers cannot run on read-only disks')
        compact = compact or {}
        if compact and read_only:
            raise ValueError('compact cannot copy onto read-only disks')
        if not set(compact) | set(compact.values()) <= set(data_disks):
            raise ValueError('compact devices must be in data_disks')
        if set(compact.values()) & (set(compact) | SERVER_DEVICES) or len(set(compact.values())) < len(compact):
            raise ValueError('compact destinations must be distinct, and neither sources nor server devices')
        if servers and set(data_disks) != SERVER_DEVICES | set(compact.values()):
            raise ValueError(f'servers need data_disks for exactly {sorted(SERVER_DEVICES)} plus compact destinations')
        self.data_disks = data_disks
        self.servers = servers
        self.read_only = read_only
        self.clickhouse_image = clickhouse_image
        self.opensearch_image = opensearch_image
        self.health_timeout_s = health_timeout_s
        self.compact = compact

    def build_volume_params(self) -> str:
        """Add the data disks, read-only, to the parent's volumes when `read_only`."""
        base = super().build_volume_params()
        if not self.read_only:
            return base
        vs = [f'    -v /mnt/disks/{d}:/mnt/{d}:ro \\' for d in self.data_disks]
        return '\n'.join(vs if base == '\\' else [base, *vs])

    def mount_lines(self) -> str:
        lines = []
        for d in self.data_disks:
            dev = f'/dev/disk/by-id/google-{d}'
            lines.append(f'            mkdir -p /mnt/disks/{d}')
            if self.read_only:
                lines.append(f'            mount -o ro {dev} /mnt/disks/{d}')
            else:
                lines.append(f'            mkfs.ext4 -F {dev}')
                lines.append(f'            mount -o discard,defaults {dev} /mnt/disks/{d}')
        return '\n'.join(lines) + '\n'

    def server_lines(self) -> str:
        ch = '/mnt/disks/clickhouse'
        os_ = '/mnt/disks/opensearch'
        t = self.health_timeout_s
        return f"""\
            sudo -u app docker pull {self.container_image}
            docker create --name pos {self.container_image}
            docker cp pos:/app/config/clickhouse/config.d {ch}/
            docker cp pos:/app/config/clickhouse/users.d {ch}/
            docker rm pos
            mkdir -p {ch}/data {os_}/data
            chown 1000:1000 {os_}/data
            docker run -d --name clickhouse --network host --ulimit nofile=262144:262144 \\
                --log-driver="gcplogs" \\
                -v {ch}/data:/var/lib/clickhouse \\
                -v {ch}/config.d:/etc/clickhouse-server/config.d \\
                -v {ch}/users.d:/etc/clickhouse-server/users.d \\
                {self.clickhouse_image}
            sysctl -w vm.max_map_count=262144
            docker run -d --name opensearch --network host \\
                --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 \\
                --log-driver="gcplogs" \\
                -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true \\
                -e OPENSEARCH_JAVA_OPTS='-Xms31g -Xmx31g' -e bootstrap.memory_lock=true \\
                -e thread_pool.write.queue_size=-1 \\
                -e search.insights.top_queries.latency.enabled=false \\
                -e search.insights.top_queries.cpu.enabled=false \\
                -e search.insights.top_queries.memory.enabled=false \\
                -v {os_}/data:/usr/share/opensearch/data \\
                {self.opensearch_image}
            wait_for_servers() {{
                local deadline=$((SECONDS + {t})) health
                until curl -sf localhost:8123/ping >/dev/null \\
                    && health=$(curl -sf localhost:9200/_cluster/health) \\
                    && [[ $health == *'"status":"green"'* || $health == *'"status":"yellow"'* ]]; do
                    if (( SECONDS >= deadline )); then
                        echo "ClickHouse and OpenSearch not healthy after {t} s"
                        exit 1
                    fi
                    sleep {HEALTH_POLL_S}
                done
            }}
            wait_for_servers
"""

    def compact_lines(self) -> str:
        """Copy each compact source onto its destination in the pos image and verify it, only while `rc` is 0."""
        lines = ''
        for src, dst in self.compact.items():
            lines += f"""\
            if [ "$rc" -eq 0 ]; then
                docker run --rm --network none --entrypoint sh \\
                    -v /mnt/disks/{src}:/src:ro -v /mnt/disks/{dst}:/dst \\
                    {self.container_image} -c '{COMPACT_COPY}' \\
                    || {{ rc=$?; echo "copying {src} onto {dst} failed with $rc"; }}
            fi
"""
        return lines

    def workload_lines(self) -> str:
        """Run pos on the host network, keep its exit code, then stop the servers."""
        arg = ' '.join(self.container_args or [])
        stop = ''
        if self.servers:
            stop += '            docker stop -t 300 opensearch clickhouse\n'
        stop += self.compact_lines()
        if not self.read_only:
            # umount leaves a clean filesystem to snapshot
            mounts = ' '.join(f'/mnt/disks/{d}' for d in self.data_disks)
            stop += f'            sync\n            umount {mounts}\n'
        return f"""\
            set +e
            sudo -u app docker run \\
                --name {CONTAINER_NAME} \\
                --network host \\
                {self.build_env_params()}
                {self.build_volume_params()}
                --log-driver="gcplogs" \\
                {self.container_image} \\
                {self.container_command} {arg} 2>&1 | {OUTPUT_FILTER}
            rc=${{PIPESTATUS[0]}}
            set -e
            echo "pos exited with $rc"
{stop}            exit "$rc"
"""

    def setup_lines(self) -> str:
        """The parent's setup, then the data disks and, with `servers`, the servers."""
        return super().setup_lines() + self.mount_lines() + (self.server_lines() if self.servers else '')

    def declare_instance(self) -> compute_v1.Instance:
        """The parent's instance with each data disk attached, kept on deletion."""
        instance = super().declare_instance()
        mode = 'READ_ONLY' if self.read_only else 'READ_WRITE'
        for device, name in self.data_disks.items():
            instance.disks.append(
                compute_v1.AttachedDisk(
                    source=f'zones/{self.zone}/disks/{name}',
                    device_name=device,
                    auto_delete=False,
                    mode=mode,
                )
            )
        return instance
