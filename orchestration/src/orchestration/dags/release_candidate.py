"""DAG that builds a release candidate: ClickHouse and OpenSearch disk snapshots of a run, and a hand-off."""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import Any

from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import DAG, Param, task
from airflow.sdk.definitions.param import ParamsDict
from airflow.task.trigger_rule import TriggerRule
from google.api_core.exceptions import PreconditionFailed
from google.cloud.storage import Client

from orchestration.dags.config.release import HandOff, ReleaseConfig
from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig
from orchestration.operators.gce import DeleteInstanceOperator
from orchestration.operators.gce_backend import ComputeEngineRunBackendSensor
from orchestration.operators.gce_disks import CreateDiskOperator, CreateSnapshotOperator, delete_disk
from orchestration.utils import GCSPath, resource_name, strhash
from orchestration.utils.common import GCP_PROJECT_PLATFORM, GCP_ZONE, shared_dag_args
from orchestration.utils.labels import TemplatedLabels
from orchestration.utils.release import SIDES, candidate_names, created_this_try, plan_candidate, task_states

BACKEND_ARGS = ['-w', '/mnt/disks/work', '-p', '8']
# working-disk name prefix to task suffix
DISK_TASKS = {'ch_build': 'clickhouse_build', 'ch': 'clickhouse', 'os': 'opensearch'}
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
        'clickhouse_build_disk_gb': rc.clickhouse_build_disk_gb,
        'clickhouse_disk_gb': rc.clickhouse_disk_gb,
        'opensearch_disk_gb': rc.opensearch_disk_gb,
    }


def upload_hand_off(
    names: dict[str, Any], plan: dict[str, dict[str, Any]], run_id: str, now: datetime | None = None
) -> str:
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
        # built: the config size; reused: the snapshot's own
        disk_size_gb={
            'clickhouse': plan['clickhouse']['disk_size_gb'],
            'opensearch': plan['opensearch']['disk_size_gb'],
        },
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
    states = task_states(ti, ['plan', *producer])
    for name in created_this_try(ti, producer, states, guard='plan'):
        delete_disk(GCP_PROJECT_PLATFORM, GCP_ZONE, name, logger)


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
    machine_type = {'clickhouse': rc.clickhouse_machine_type, 'opensearch': rc.opensearch_machine_type}
    work_disk_gb = {'clickhouse': rc.clickhouse_work_disk_gb, 'opensearch': rc.opensearch_work_disk_gb}
    # ClickHouse builds on the large disk and is copied onto the snapshotted one
    data_disks = {
        'clickhouse': {'clickhouse': _name('ch_build_disk'), 'clickhouse-compact': _name('ch_disk')},
        'opensearch': {'opensearch': _name('os_disk')},
    }
    compact = {'clickhouse': {'clickhouse': 'clickhouse-compact'}, 'opensearch': None}

    def labels(side: str, **templated: str) -> TemplatedLabels:
        return TemplatedLabels(
            {'tool': 'pos', 'step': f'backend_{side}'}, templated={'product': _name('product'), **templated}
        )

    @task(task_id='names')
    def names(release_uri: str) -> dict[str, Any]:
        return candidate(release_uri)

    @task(task_id='plan')
    def plan(names: Any) -> dict[str, dict[str, Any]]:
        return plan_candidate(names, GCP_PROJECT_PLATFORM, GCP_ZONE)

    # Airflow fills `run_id` from the context
    @task(task_id='write_hand_off', trigger_rule=TriggerRule.NONE_FAILED)
    def write_hand_off(names: Any, plan: Any, run_id: Any = None) -> str:
        return upload_hand_off(names, plan, run_id)

    def build_task(side: str) -> Any:
        @task.short_circuit(task_id=f'build_{side}', ignore_downstream_trigger_rules=False)
        def build(plan: Any) -> bool:
            return plan[side]['action'] == 'build'

        return build

    def delete_disk_task(device: str) -> Any:
        @task(task_id=f'delete_disk_{device}', trigger_rule=TriggerRule.ALL_DONE)
        def delete_disk_(ti: Any = None) -> None:
            cleanup_disk(ti, device)

        return delete_disk_()

    n = names('{{ params.release_uri }}')
    p = plan(n)
    hand_off = write_hand_off(n, p)
    # the only leaf: the run fails if the hand-off or any clean-up failed
    done = EmptyOperator(task_id='done', trigger_rule=TriggerRule.NONE_FAILED)
    hand_off >> done

    for side, (short, disk_shorts) in SIDES.items():
        vm_name = resource_name(f'release_candidate_{side}')
        devices = {DISK_TASKS[d]: d for d in disk_shorts}
        creates = [
            CreateDiskOperator(
                task_id=f'create_disk_{device}',
                disk_name=_name(f'{d}_disk'),
                size_gb=_name(f'{device}_disk_gb'),
                labels=labels(side, run='{{ run_id }}'),
            )
            for device, d in devices.items()
        ]
        run = ComputeEngineRunBackendSensor(
            task_id=f'run_backend_{side}',
            instance_name=vm_name,
            labels=labels(side),
            container_image=_name('pos_image'),
            container_args=['-s', f'backend_{side}', '-r', '{{ params.release_uri }}', *BACKEND_ARGS],
            data_disks=data_disks[side],
            compact=compact[side],
            servers={side},
            read_only=False,
            clickhouse_image=_name('clickhouse_image') if side == 'clickhouse' else '',
            opensearch_image=_name('opensearch_image') if side == 'opensearch' else '',
            machine_type=machine_type[side],
            work_disk_size_gb=work_disk_gb[side],
            boot_disk_size_gb=rc.boot_disk_gb,
            source_image=up.cos_image,
            max_run_duration=timedelta(hours=rc.backend_max_run_hours),
            retries=0,
            deferrable=True,
        )
        delete_vm = DeleteInstanceOperator(
            task_id=f'delete_vm_{side}',
            resource_id=vm_name,
            trigger_rule=TriggerRule.ALL_DONE,
        )
        snapshot = CreateSnapshotOperator(
            task_id=f'snapshot_{side}',
            disk_name=_name(f'{short}_disk'),
            snapshot_name=_name(f'{short}_snapshot'),
            labels=labels(side, run='{{ run_id }}', release=_name('yymm')),
            provenance={
                'pos_image': _name('pos_image'),
                'server_image': _name(f'{side}_image'),
                'airflow_run_id': '{{ run_id }}',
            },
        )

        # a reused side skips its creates; the rest of the branch follows trigger rules
        build_task(side)(p) >> creates
        for create in creates:
            create >> run
        run >> delete_vm >> snapshot
        run >> snapshot
        snapshot >> hand_off
        delete_vm >> done
        for device in devices:
            delete = delete_disk_task(device)
            delete_vm >> delete
            snapshot >> delete
            delete >> done
