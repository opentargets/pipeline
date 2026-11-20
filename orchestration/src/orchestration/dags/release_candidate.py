"""DAG that builds a release candidate: ClickHouse and OpenSearch disk snapshots of a run, and a hand-off."""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import Any

from airflow.sdk import DAG, Param, chain, task
from airflow.sdk.definitions.param import ParamsDict
from airflow.task.trigger_rule import TriggerRule
from google.cloud.storage import Client

from orchestration.dags.config.release import HandOff, ReleaseConfig
from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig
from orchestration.operators.gce import DeleteInstanceOperator
from orchestration.operators.gce_backend import ComputeEngineRunBackendSensor
from orchestration.operators.gce_disks import (
    CreateDiskOperator,
    CreateSnapshotOperator,
    DeleteDiskOperator,
    SnapshotGuardOperator,
    _delete_snapshot,
)
from orchestration.utils import GCSPath, resource_name, strhash
from orchestration.utils.common import GCP_PROJECT_PLATFORM, shared_dag_args
from orchestration.utils.labels import Labels
from orchestration.utils.release import candidate_names

BACKEND_ARGS = ['-w', '/mnt/disks/work', '-p', '8']
DEVICES = {'clickhouse': 'ch', 'opensearch': 'os'}
logger = logging.getLogger(__name__)


def _name(key: str) -> str:
    return f"{{{{ ti.xcom_pull(task_ids='names')['{key}'] }}}}"


def snapshot_path(name: str) -> str:
    return f'projects/{GCP_PROJECT_PLATFORM}/global/snapshots/{name}'


def upload_hand_off(names: dict[str, Any], run_id: str, now: datetime | None = None) -> str:
    """Write the hand-off to `<release_uri>/release_candidate.json`; fail if it exists."""
    rc = ReleaseConfig()
    up = UnifiedPipelineConfig()
    hand_off = HandOff(
        release_uri=names['release_uri'],
        run=f'{names["prefix"]}/{names["run"]}',
        product=names['product'],
        release=names['release'],
        snapshots={
            'clickhouse': snapshot_path(names['ch_snapshot']),
            'opensearch': snapshot_path(names['os_snapshot']),
        },
        disk_size_gb={'clickhouse': rc.clickhouse_disk_gb, 'opensearch': rc.opensearch_disk_gb},
        pos_image=up.pos_image,
        pos_version=up.pos_version,
        clickhouse_image=rc.clickhouse_image,
        opensearch_image=rc.opensearch_image,
        created=now or datetime.now(tz=UTC),
        airflow_run_id=run_id,
    )
    uri = HandOff.uri(names['release_uri'])
    bucket, path = GCSPath(uri).split()
    blob = Client(project=GCP_PROJECT_PLATFORM).bucket(bucket).blob(path)
    blob.upload_from_string(
        hand_off.model_dump_json(indent=2), content_type='application/json', if_generation_match=0
    )
    return uri


def cleanup_snapshots(ti: Any) -> None:
    """Delete the snapshots this run created; a snapshot task that did not succeed pushed no name."""
    for task_id in ('snapshot_clickhouse', 'snapshot_opensearch'):
        name = ti.xcom_pull(task_ids=task_id)
        if name:
            _delete_snapshot(GCP_PROJECT_PLATFORM, name, logger)


with DAG(
    dag_id='release_candidate',
    description='Build a release candidate: ClickHouse and OpenSearch snapshots of a unified-pipeline run',
    default_args=shared_dag_args,
    catchup=False,
    schedule=None,
    user_defined_filters={'strhash': strhash},
    tags=['release'],
    params=ParamsDict({
        'release_uri': Param(
            type='string',
            description='The run to build from, e.g. gs://open-targets-pipeline-runs/do/platform-2612-3.',
        ),
    }),
) as dag:
    rc = ReleaseConfig()
    up = UnifiedPipelineConfig()
    labels = Labels({'tool': 'pos', 'step': 'backend'})
    vm_name = resource_name('release_candidate_backend')
    sizes = {'clickhouse': rc.clickhouse_disk_gb, 'opensearch': rc.opensearch_disk_gb}

    @task(task_id='names')
    def names(release_uri: str) -> dict[str, Any]:
        return asdict(candidate_names(release_uri))

    n = names('{{ params.release_uri }}')

    guard = SnapshotGuardOperator(
        task_id='guard',
        snapshot_names=[_name('ch_snapshot'), _name('os_snapshot')],
        blob_uri=HandOff.uri('{{ params.release_uri }}'),
    )

    creates = [
        CreateDiskOperator(
            task_id=f'create_disk_{device}',
            disk_name=_name(f'{short}_disk'),
            size_gb=sizes[device],
            labels=labels,
        )
        for device, short in DEVICES.items()
    ]

    run = ComputeEngineRunBackendSensor(
        task_id='run_backend',
        instance_name=vm_name,
        labels=labels,
        container_image=up.pos_image,
        container_args=['-s', 'backend', '-r', '{{ params.release_uri }}', *BACKEND_ARGS],
        data_disks={device: _name(f'{short}_disk') for device, short in DEVICES.items()},
        servers=True,
        read_only=False,
        clickhouse_image=rc.clickhouse_image,
        opensearch_image=rc.opensearch_image,
        machine_type=rc.backend_machine_type,
        work_disk_size_gb=rc.work_disk_gb,
        boot_disk_size_gb=rc.boot_disk_gb,
        source_image=up.cos_image,
        max_run_duration=timedelta(hours=rc.backend_max_run_hours),
        retries=0,
        deferrable=True,
    )

    delete_vm = DeleteInstanceOperator(
        task_id='delete_vm_backend',
        resource_id=vm_name,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    snapshots = [
        CreateSnapshotOperator(
            task_id=f'snapshot_{device}',
            disk_name=_name(f'{short}_disk'),
            snapshot_name=_name(f'{short}_snapshot'),
            labels={**labels, 'run': '{{ run_id }}', 'release': _name('yymm')},
        )
        for device, short in DEVICES.items()
    ]

    @task(task_id='write_hand_off')
    def write_hand_off(names: Any, run_id: str) -> str:
        return upload_hand_off(names, run_id)

    hand_off = write_hand_off(n, '{{ run_id }}')

    deletes = [
        DeleteDiskOperator(
            task_id=f'delete_disk_{device}',
            disk_name=_name(f'{short}_disk'),
            trigger_rule=TriggerRule.ALL_DONE,
        )
        for device, short in DEVICES.items()
    ]

    @task(task_id='fail_cleanup_snapshots', trigger_rule=TriggerRule.ONE_FAILED)
    def fail_cleanup_snapshots(ti: Any = None) -> None:
        cleanup_snapshots(ti)

    cleanup = fail_cleanup_snapshots()

    chain(n, guard, creates, run, delete_vm)
    for s in snapshots:
        run >> s
        delete_vm >> s
        s >> hand_off
        s >> deletes
        s >> cleanup
    delete_vm >> deletes
    hand_off >> cleanup
