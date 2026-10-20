"""DAG that stages every clinical_report input that needs a model or an LLM.

This runs outside the release. For each source — the AACT monthly archive and the
clinical mining inputs (TTD, EMA, PMDA) — it runs the LLM extraction and the NER
and stop-reason models, and publishes write-once, revision-tagged snapshots under
``gs://clinical_staging/<source>/<version>`` for ``unified_pipeline`` to copy in.

The handoff to the release is the same one every ingestion pipeline here uses:
a version-pinned path that a human names in ``pis/config.yaml``. There is no
Airflow dependency between the two, deliberately. If a release pins a version
that was never staged, its copy fails on a missing path.

Record stores under ``gs://clinical_staging/store`` are keyed on the text, model
and model revision, so a new version only runs the models over texts not seen
before. See ``orchestration/docs/datasources/clinical_staging/README.md``.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING

from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import DAG, chain, task_group
from airflow.task.trigger_rule import TriggerRule

from orchestration.dags.config.clinical_staging import ClinicalStagingConfig
from orchestration.operators.gce import ComputeEngineRunContainerizedWorkloadSensor, DeleteInstanceOperator
from orchestration.operators.gcs import UploadStringOperator
from orchestration.utils import resource_name, strhash, to_yaml
from orchestration.utils.common import shared_dag_args
from orchestration.utils.labels import Labels

if TYPE_CHECKING:
    from typing import Any

with DAG(
    dag_id='clinical_staging',
    description='Open Targets — clinical staging',
    default_args=shared_dag_args,
    catchup=False,
    schedule=None,
    user_defined_filters={'strhash': strhash},
    tags=['clinical_staging', 'preprocessing'],
) as dag:
    logger = logging.getLogger(__name__)
    config = ClinicalStagingConfig()
    steps: dict[str, dict[str, Any]] = {}  # registry of tasks, used to build dependencies

    logger.info(f'aact {config.aact_version}, clinical_mining {config.clinical_mining_version}, bucket {config.bucket}')

    # ==============================================================================================
    # Every step follows the same shape, whether it is a PIS or a PTS one:
    #
    #   u. Upload — the step's config to GCS.
    #   r. Run    — the step in a GCE VM, and wait until it finishes.
    #   t. Delete — the VM.
    #   e. End    — succeeds only if the run did, so a failed step never
    #               starts the ones that depend on it. The delete still runs
    #               either way, so it cannot stand in for this.
    #
    # There is no diffing here. Unlike a release, this pipeline is already
    # incremental where it counts: the extraction only sends the model the
    # trials that are not in its cache. Rerunning a step that has nothing to do
    # is cheap by construction.
    # ==============================================================================================
    for step_name in config.steps():

        @task_group(group_id=step_name)
        def staging_step(step_name: str) -> None:
            stage, step = step_name.split('_', 1)
            step_definition = config.step_definition(step_name)

            config_uri = config.config_uri(step_name)
            vm_name = resource_name(step_name.replace('_', '-'))

            u = UploadStringOperator(
                task_id=f'upload_config_{step_name}',
                contents=to_yaml(config.step_config(step_name)),
                dst_uri=config_uri,
                overwrite=True,
            )

            r = ComputeEngineRunContainerizedWorkloadSensor(
                task_id=f'run_{step_name}',
                instance_name=vm_name,
                labels=Labels({'tool': stage, 'step': step}),
                container_image=config.step_image(step_name),
                container_env=config.step_env_vars(step_name),
                container_files={config_uri: '/config.yaml'},
                container_secret_files=step_definition.get('gce_secret_files'),
                machine_type=config.machine_type_for(step_name),
                boot_disk_size_gb=config.boot_disk_size,
                work_disk_size_gb=config.disk_size,
                deferrable=True,
            )

            t = DeleteInstanceOperator(
                task_id=f'delete_vm_{step_name}',
                resource_id=vm_name,
                trigger_rule=TriggerRule.NONE_SKIPPED,
                execution_timeout=timedelta(seconds=300),
            )

            e = EmptyOperator(
                task_id=f'end_{step_name}',
                trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
            )

            chain(u, r, t)
            chain(r, e)
            chain(t, e)
            steps[step_name] = {'start': u, 'end': e}

        staging_step(step_name)

    # ==============================================================================================
    # After creating all the tasks, we tie them together by creating dependencies.
    for step_name, step_tasks in steps.items():
        for dep in config.step_definition(step_name).get('depends_on', []):
            step_tasks['start'].set_upstream(steps[dep]['end'])

if __name__ == '__main__':
    dag.test()
