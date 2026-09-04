"""DAG that extracts structured drug and disease evidence from AACT clinical trials.

This runs outside the release. It downloads the pinned AACT monthly archive and
asks an LLM to read every trial it has not read before, publishing the result to
``gs://aact_data/<aact_version>`` for ``unified_pipeline`` to copy in.

The handoff to the release is the same one every ingestion pipeline here uses:
a version-pinned path that a human names in ``pis/config.yaml``. There is no
Airflow dependency between the two, deliberately.

The two pin their AACT versions independently. This pipeline can run ahead —
extracting each monthly archive as it lands keeps the cache warm and makes the
next release cheap — while a release moves onto a new archive only when someone
bumps it, and several releases can share one. If a release ever pins an archive
that was never extracted, its copy fails on a missing path, which is a better
outcome than quietly reading the wrong snapshot.

Because the extraction is cached on a hash of the exact prompt, model and output
schema, rerunning this DAG after a new AACT archive only pays for the trials
that are genuinely new or whose text changed upstream. See :py:mod:`pts.result_cache`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from airflow.sdk import DAG, chain, task_group

from orchestration.dags.config.aact_trial_extraction import AactTrialExtractionConfig
from orchestration.operators.gce import ComputeEngineRunContainerizedWorkloadSensor, DeleteInstanceOperator
from orchestration.operators.gcs import UploadStringOperator
from orchestration.utils import resource_name, to_yaml
from orchestration.utils.common import shared_dag_args
from orchestration.utils.labels import Labels

if TYPE_CHECKING:
    from typing import Any

with DAG(
    dag_id='aact_trial_extraction',
    description='Open Targets — AACT clinical trial LLM extraction',
    default_args=shared_dag_args,
    catchup=False,
    schedule=None,
    tags=['aact_trial_extraction', 'preprocessing'],
) as dag:
    logger = logging.getLogger(__name__)
    config = AactTrialExtractionConfig()
    steps: dict[str, dict[str, Any]] = {}  # registry of tasks, used to build dependencies

    logger.info(f'aact version {config.aact_version}, publishing to {config.snapshot_uri}')

    # ==============================================================================================
    # Every step follows the same shape, whether it is a PIS or a PTS one:
    #
    #   u. Upload — the step's config to GCS.
    #   r. Run    — the step in a GCE VM, and wait until it finishes.
    #   t. Delete — the VM.
    #
    # There is no diffing here. Unlike a release, this pipeline is already
    # incremental where it counts: the extraction only sends the model the
    # trials that are not in its cache. Rerunning a step that has nothing to do
    # is cheap by construction.
    # ==============================================================================================
    for step_name in config.steps():

        @task_group(group_id=step_name)
        def extraction_step(step_name: str) -> None:
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
                machine_type=config.machine_type,
                work_disk_size_gb=config.disk_size,
                deferrable=True,
            )

            t = DeleteInstanceOperator(
                task_id=f'delete_vm_{step_name}',
                resource_id=vm_name,
            )

            chain(u, r, t)
            steps[step_name] = {'start': u, 'end': t}

        extraction_step(step_name)

    # ==============================================================================================
    # After creating all the tasks, we tie them together by creating dependencies.
    for step_name, step_tasks in steps.items():
        for dep in config.step_definition(step_name).get('depends_on', []):
            step_tasks['start'].set_upstream(steps[dep]['end'])

if __name__ == '__main__':
    dag.test()
