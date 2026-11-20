"""DAG that publishes a release candidate: tarballs, bucket and BigQuery copies, then the manifest."""

from __future__ import annotations

import logging
import re
from typing import Any

from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import DAG, Param, task
from airflow.sdk.definitions.param import ParamsDict
from airflow.task.trigger_rule import TriggerRule
from google.api_core.exceptions import NotFound
from google.cloud.storage import Client

from orchestration.dags.config.release import HandOff, ReleaseConfig
from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig
from orchestration.operators.gce import ComputeEngineRunContainerizedWorkloadSensor, DeleteInstanceOperator
from orchestration.operators.gce_backend import ComputeEngineRunBackendSensor
from orchestration.operators.gce_disks import CreateDiskOperator, delete_disk
from orchestration.utils import GCSPath, resource_name, strhash
from orchestration.utils.common import GCP_PROJECT_PLATFORM, GCP_ZONE, shared_dag_args
from orchestration.utils.labels import TemplatedLabels
from orchestration.utils.release import candidate_names, created_this_try, task_states

DEVICES = ('clickhouse', 'opensearch')
WORK_DIR = '/mnt/disks/work'
TARBALLS_POOL = 16
SMALL_STEP_POOL = 8
GATED = ('aws_sync', 'bigquery', 'scratch_cleanup')
_VERSION_RE = re.compile(r'^(\d+)\.(\d+)\.(\d+)(?:rc(\d+))?$')
logger = logging.getLogger(__name__)


def _value(key: str) -> str:
    return f"{{{{ ti.xcom_pull(task_ids='read_hand_off', key='values')['{key}'] }}}}"


def pos_args(step: str, pool: int) -> list[str]:
    return ['-s', step, '-r', '{{ params.release_uri }}', '-w', WORK_DIR, '-p', str(pool)]


def pos_version_key(version: str) -> tuple[int, ...] | None:
    """`26.12.0rc5` or `26.12.0` as a sortable tuple, a final after its rcs; None for any other form."""
    match = _VERSION_RE.fullmatch(version)
    if match is None:
        return None
    major, minor, patch, rc = match.groups()
    final = (1, 0) if rc is None else (0, int(rc))
    return (int(major), int(minor), int(patch), *final)


def check_scratch(pos_version: str, minimum: str) -> None:
    """Raise unless `pos_version` is a release or rc at least `minimum`."""
    have, need = pos_version_key(pos_version), pos_version_key(minimum)
    if have is None or need is None:
        raise ValueError(f"scratch needs a pos release or rc version, got '{pos_version}' (minimum '{minimum}')")
    if have < need:
        raise ValueError(f'scratch needs pos {minimum} or later, the candidate has {pos_version}')


def step_runs(step: str, product: str, scratch: bool) -> bool:
    """Whether a gated step applies to this product and mode."""
    platform = product == 'platform'
    return {
        'bigquery': platform,
        'aws_sync': platform and not scratch,
        'scratch_cleanup': platform and scratch,
    }[step]


def gate_open(gates: dict[str, bool], step: str) -> bool:
    return gates[step] is True


def restored_disk(name: str) -> str:
    restored = f'{name}-r'
    if len(restored) > 63:
        raise ValueError(f"'{restored}' is longer than 63 characters")
    return restored


def load_hand_off(release_uri: str, scratch: bool) -> dict[str, Any]:
    """Read and check the run's hand-off; return it with the values, env and gates the DAG uses."""
    if not isinstance(scratch, bool):
        raise TypeError(f'scratch must be a boolean, got {scratch!r}')
    rc = ReleaseConfig()
    names = candidate_names(release_uri)
    uri = HandOff.uri(release_uri)
    bucket, path = GCSPath(uri).split()
    try:
        text = Client(project=GCP_PROJECT_PLATFORM).bucket(bucket).blob(path).download_as_text()
    except NotFound as e:
        raise FileNotFoundError(f'no release candidate at {uri}') from e
    hand_off = HandOff.model_validate_json(text)
    if hand_off.release_uri.rstrip('/') != release_uri.rstrip('/'):
        raise ValueError(f'hand-off release_uri {hand_off.release_uri} is not {release_uri}')
    if hand_off.product != names.product:
        raise ValueError(f'hand-off product {hand_off.product} is not {names.product}')
    if set(hand_off.snapshots) != set(DEVICES) or set(hand_off.disk_size_gb) != set(DEVICES):
        raise ValueError(f'hand-off needs snapshots and disk sizes for {list(DEVICES)}')
    if scratch:
        check_scratch(hand_off.pos_version, rc.scratch_min_pos_version)

    env = {'POS_SCRATCH': '1'} if scratch else {}
    # scratch writes nothing to S3 and has no AWS secret
    aws = {} if scratch else {'AWS_SHARED_CREDENTIALS_FILE': rc.aws_credentials_path}
    disks = {'clickhouse': names.ch_disk, 'opensearch': names.os_disk}
    return {
        'hand_off': hand_off.model_dump(mode='json'),
        'values': {
            'release_uri': hand_off.release_uri,
            'product': hand_off.product,
            'pos_image': hand_off.pos_image,
            **{f'{d}_snapshot': hand_off.snapshots[d] for d in DEVICES},
            **{f'{d}_disk_gb': hand_off.disk_size_gb[d] for d in DEVICES},
            **{f'{d}_disk': restored_disk(disks[d]) for d in DEVICES},
        },
        'env': env,
        'aws_env': {**env, **aws},
        'aws_secrets': {} if scratch else {rc.aws_secret: rc.aws_credentials_path},
        'gates': {step: step_runs(step, hand_off.product, scratch) for step in GATED},
    }


def cleanup_restored_disk(ti: Any, device: str) -> None:
    """Delete the restored disk if this try created it."""
    producer = {f'restore_disk_{device}': 'disk'}
    states = task_states(ti, ['read_hand_off', *producer])
    for name in created_this_try(ti, producer, states, guard='read_hand_off'):
        delete_disk(GCP_PROJECT_PLATFORM, GCP_ZONE, name, logger)


with DAG(
    dag_id='release',
    description='Publish a release candidate: tarballs, release buckets, BigQuery and the manifest',
    default_args=shared_dag_args,
    catchup=False,
    schedule=None,
    max_active_runs=1,
    user_defined_filters={'strhash': strhash},
    tags=['release'],
    params=ParamsDict({
        'release_uri': Param(
            type='string',
            description='The run whose release candidate to publish, e.g. gs://open-targets-pipeline-runs/do/platform-2612-3.',
        ),
        'scratch': Param(
            False,
            type='boolean',
            description='Publish to the scratch bucket and datasets instead of the public release.',
        ),
    }),
) as dag:
    rc = ReleaseConfig()
    up = UnifiedPipelineConfig()

    def labels(step: str, **templated: str) -> TemplatedLabels:
        return TemplatedLabels({'tool': 'pos', 'step': step}, templated={'product': _value('product'), **templated})

    @task(task_id='read_hand_off', multiple_outputs=True)
    def read_hand_off(params: Any = None) -> dict[str, Any]:
        return load_hand_off(params['release_uri'], params['scratch'])

    r: Any = read_hand_off()

    restores = [
        CreateDiskOperator(
            task_id=f'restore_disk_{device}',
            disk_name=_value(f'{device}_disk'),
            size_gb=_value(f'{device}_disk_gb'),
            source_snapshot=_value(f'{device}_snapshot'),
            labels=labels('tarballs', run='{{ run_id }}'),
        )
        for device in DEVICES
    ]

    tarballs_vm = resource_name('release_tarballs')
    run_tarballs = ComputeEngineRunBackendSensor(
        task_id='run_tarballs',
        instance_name=tarballs_vm,
        labels=labels('tarballs'),
        container_image=_value('pos_image'),
        container_args=pos_args('tarballs', TARBALLS_POOL),
        container_env=r['env'],
        vm_service_account=rc.pos_release_service_account,
        data_disks={device: _value(f'{device}_disk') for device in DEVICES},
        servers=False,
        read_only=True,
        machine_type=rc.tarballs_machine_type,
        work_disk_size_gb=rc.small_step_work_disk_gb,
        boot_disk_size_gb=rc.boot_disk_gb,
        source_image=up.cos_image,
        deferrable=True,
    )
    delete_vm_tarballs = DeleteInstanceOperator(
        task_id='delete_vm_tarballs',
        resource_id=tarballs_vm,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    def delete_restored_disk_task(device: str) -> Any:
        @task(task_id=f'delete_restored_disk_{device}', trigger_rule=TriggerRule.ALL_DONE)
        def delete_restored_disk(ti: Any = None) -> None:
            cleanup_restored_disk(ti, device)

        return delete_restored_disk()

    deletes = [delete_restored_disk_task(device) for device in DEVICES]

    def small_step(
        step: str, env: Any, secrets: Any = None, trigger_rule: str = TriggerRule.ALL_SUCCESS
    ) -> tuple[ComputeEngineRunContainerizedWorkloadSensor, DeleteInstanceOperator]:
        vm = resource_name(f'release_{step}')
        run = ComputeEngineRunContainerizedWorkloadSensor(
            task_id=f'run_{step}',
            instance_name=vm,
            labels=labels(step),
            container_image=_value('pos_image'),
            container_args=pos_args(step, SMALL_STEP_POOL),
            container_env=env,
            container_secret_files=secrets,
            vm_service_account=rc.pos_release_service_account,
            machine_type=rc.small_step_machine_type,
            work_disk_size_gb=rc.small_step_work_disk_gb,
            boot_disk_size_gb=rc.boot_disk_gb,
            source_image=up.cos_image,
            deferrable=True,
            trigger_rule=trigger_rule,
        )
        delete = DeleteInstanceOperator(
            task_id=f'delete_vm_{step}',
            resource_id=vm,
            trigger_rule=TriggerRule.ALL_DONE,
        )
        run >> delete
        if step in GATED:
            # skips only the run and its VM delete; later tasks follow their trigger rules
            @task.short_circuit(task_id=f'gate_{step}', ignore_downstream_trigger_rules=False)
            def gate(gates: dict[str, bool]) -> bool:
                return gate_open(gates, step)

            gate(r['gates']) >> [run, delete]
        return run, delete

    gcs_sync, delete_vm_gcs_sync = small_step('gcs_sync', r['env'])
    aws_sync, delete_vm_aws_sync = small_step('aws_sync', r['aws_env'], r['aws_secrets'])
    bigquery, delete_vm_bigquery = small_step('bigquery', r['env'])
    publish, delete_vm_publish = small_step(
        'publish_manifest', r['aws_env'], r['aws_secrets'], trigger_rule=TriggerRule.NONE_FAILED
    )
    # after every other step whatever its outcome, so a failed scratch run drops its datasets;
    # min one success keeps it off when read_hand_off failed
    cleanup, delete_vm_cleanup = small_step(
        'scratch_cleanup', r['env'], trigger_rule=TriggerRule.ALL_DONE_MIN_ONE_SUCCESS
    )

    # the only leaf: ALL_SUCCESS would skip it as soon as a gated step is skipped, and a skipped
    # leaf lets the run succeed whatever fails after
    done = EmptyOperator(task_id='done', trigger_rule=TriggerRule.NONE_FAILED)

    r >> restores
    restores >> run_tarballs >> delete_vm_tarballs
    for restore, delete in zip(restores, deletes, strict=True):
        [restore, delete_vm_tarballs] >> delete
    run_tarballs >> gcs_sync >> aws_sync
    [gcs_sync, aws_sync, bigquery] >> publish
    # not aws_sync: skipped whenever cleanup runs, and a skipped upstream skips this rule
    [run_tarballs, gcs_sync, bigquery, publish] >> cleanup
    [
        run_tarballs,
        delete_vm_tarballs,
        *deletes,
        gcs_sync,
        aws_sync,
        bigquery,
        publish,
        cleanup,
        delete_vm_gcs_sync,
        delete_vm_aws_sync,
        delete_vm_bigquery,
        delete_vm_publish,
        delete_vm_cleanup,
    ] >> done
