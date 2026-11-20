"""DAG that builds a release candidate: ClickHouse and OpenSearch disk snapshots of a run, and a hand-off."""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import Any

from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import DAG, Param, chain, task
from airflow.sdk.definitions.param import ParamsDict
from airflow.task.trigger_rule import TriggerRule
from google.api_core.exceptions import PreconditionFailed
from google.cloud.storage import Client

from orchestration.dags.config.release import HandOff, ReleaseConfig
from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig
from orchestration.operators.gce import DeleteInstanceOperator
from orchestration.operators.gce_backend import ComputeEngineRunBackendSensor
from orchestration.operators.gce_disks import (
    CreateDiskOperator,
    CreateSnapshotOperator,
    SnapshotGuardOperator,
    delete_disk,
    delete_snapshot,
)
from orchestration.utils import GCSPath, resource_name, strhash
from orchestration.utils.common import GCP_PROJECT_PLATFORM, GCP_ZONE, shared_dag_args
from orchestration.utils.labels import TemplatedLabels
from orchestration.utils.release import candidate_names, created_this_try, task_states

BACKEND_ARGS = ['-w', '/mnt/disks/work', '-p', '8']
DEVICES = {'clickhouse': 'ch', 'opensearch': 'os'}
SNAPSHOT_TASKS = {f'snapshot_{device}': 'return_value' for device in DEVICES}
logger = logging.getLogger(__name__)


def _name(key: str) -> str:
    return f"{{{{ ti.xcom_pull(task_ids='names')['{key}'] }}}}"


def snapshot_path(name: str) -> str:
    return f'projects/{GCP_PROJECT_PLATFORM}/global/snapshots/{name}'


def candidate(release_uri: str) -> dict[str, Any]:
    """The run's names plus the images and disk sizes the build uses, read once."""
    rc = ReleaseConfig()
    up = UnifiedPipelineConfig()
    return {
        **asdict(candidate_names(release_uri.rstrip('/'))),
        'pos_image': up.pos_image,
        'pos_version': up.pos_version,
        'clickhouse_image': rc.clickhouse_image,
        'opensearch_image': rc.opensearch_image,
        'clickhouse_disk_gb': rc.clickhouse_disk_gb,
        'opensearch_disk_gb': rc.opensearch_disk_gb,
    }


def upload_hand_off(names: dict[str, Any], run_id: str, now: datetime | None = None) -> str:
    """Write the hand-off to `<release_uri>/release_candidate.json`; fail if another run wrote it."""
    hand_off = HandOff(
        release_uri=names['release_uri'],
        run=f'{names["prefix"]}/{names["run"]}',
        product=names['product'],
        release=names['release'],
        snapshots={
            'clickhouse': snapshot_path(names['ch_snapshot']),
            'opensearch': snapshot_path(names['os_snapshot']),
        },
        disk_size_gb={'clickhouse': names['clickhouse_disk_gb'], 'opensearch': names['opensearch_disk_gb']},
        pos_image=names['pos_image'],
        pos_version=names['pos_version'],
        clickhouse_image=names['clickhouse_image'],
        opensearch_image=names['opensearch_image'],
        created=now or datetime.now(tz=UTC),
        airflow_run_id=run_id,
    )
    uri = HandOff.uri(names['release_uri'])
    bucket, path = GCSPath(uri).split()
    blob = Client(project=GCP_PROJECT_PLATFORM).bucket(bucket).blob(path)
    try:
        blob.upload_from_string(
            hand_off.model_dump_json(indent=2), content_type='application/json', if_generation_match=0
        )
    except PreconditionFailed:
        # a retry after an upload that succeeded
        if HandOff.model_validate_json(blob.download_as_text()).airflow_run_id != run_id:
            raise
    return uri


def cleanup_disk(ti: Any, device: str) -> None:
    """Delete the working disk if this try created it."""
    producer = {f'create_disk_{device}': 'disk'}
    states = task_states(ti, ['guard', *producer])
    for name in created_this_try(ti, producer, states):
        delete_disk(GCP_PROJECT_PLATFORM, GCP_ZONE, name, logger)


def cleanup_snapshots(ti: Any) -> None:
    """Delete the snapshots this try created, unless the hand-off was written."""
    states = task_states(ti, ['guard', 'write_hand_off', *SNAPSHOT_TASKS])
    if states.get('write_hand_off') == 'success':
        return
    for name in created_this_try(ti, SNAPSHOT_TASKS, states):
        delete_snapshot(GCP_PROJECT_PLATFORM, name, logger)


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
    vm_name = resource_name('release_candidate_backend')

    def labels(**templated: str) -> TemplatedLabels:
        return TemplatedLabels({'tool': 'pos', 'step': 'backend'}, templated={'product': _name('product'), **templated})

    @task(task_id='names')
    def names(release_uri: str) -> dict[str, Any]:
        return candidate(release_uri)

    n = names('{{ params.release_uri }}')

    guard = SnapshotGuardOperator(
        task_id='guard',
        snapshot_names=[_name('ch_snapshot'), _name('os_snapshot')],
        disk_names=[_name('ch_disk'), _name('os_disk')],
        blob_uri=HandOff.uri(_name('release_uri')),
    )

    creates = [
        CreateDiskOperator(
            task_id=f'create_disk_{device}',
            disk_name=_name(f'{short}_disk'),
            size_gb=_name(f'{device}_disk_gb'),
            labels=labels(run='{{ run_id }}'),
        )
        for device, short in DEVICES.items()
    ]

    run = ComputeEngineRunBackendSensor(
        task_id='run_backend',
        instance_name=vm_name,
        labels=labels(),
        container_image=_name('pos_image'),
        container_args=['-s', 'backend', '-r', '{{ params.release_uri }}', *BACKEND_ARGS],
        data_disks={device: _name(f'{short}_disk') for device, short in DEVICES.items()},
        servers=True,
        read_only=False,
        clickhouse_image=_name('clickhouse_image'),
        opensearch_image=_name('opensearch_image'),
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
            labels=labels(run='{{ run_id }}', release=_name('yymm')),
        )
        for device, short in DEVICES.items()
    ]

    # Airflow fills `run_id` from the context
    @task(task_id='write_hand_off')
    def write_hand_off(names: Any, run_id: Any = None) -> str:
        return upload_hand_off(names, run_id)

    hand_off = write_hand_off(n)

    def delete_disk_task(device: str) -> Any:
        @task(task_id=f'delete_disk_{device}', trigger_rule=TriggerRule.ALL_DONE)
        def delete_disk_(ti: Any = None) -> None:
            cleanup_disk(ti, device)

        return delete_disk_()

    deletes = [delete_disk_task(device) for device in DEVICES]

    @task(task_id='fail_cleanup_snapshots', trigger_rule=TriggerRule.ALL_DONE)
    def fail_cleanup_snapshots(ti: Any = None) -> None:
        cleanup_snapshots(ti)

    cleanup = fail_cleanup_snapshots()

    # the only leaf: the run fails unless the hand-off and every clean-up succeeded
    done = EmptyOperator(task_id='done', trigger_rule=TriggerRule.ALL_SUCCESS)

    chain(n, guard, creates, run, delete_vm)
    for s in snapshots:
        run >> s
        delete_vm >> s
        s >> hand_off
        s >> deletes
        s >> cleanup
    delete_vm >> deletes
    hand_off >> cleanup
    [hand_off, delete_vm, cleanup, *deletes] >> done
