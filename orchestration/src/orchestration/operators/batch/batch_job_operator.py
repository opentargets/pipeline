"""Batch Job Operator."""

from __future__ import annotations

import time
from typing import Any

from airflow.providers.google.cloud.operators.cloud_batch import CloudBatchSubmitJobOperator
from airflow.sdk import Context
from google.cloud import batch_v1

from orchestration.models.batch import BatchIndexRow, BatchJobOperatorSpec
from orchestration.utils.common import GCP_PROJECT_GENETICS, GCP_REGION
from orchestration.utils.labels import Labels


class BatchJobOperator(CloudBatchSubmitJobOperator):
    """Generic Batch Job operator.

    This operator has to be used in conjunction to the BatchIndexOperator.
    It runs the google batch jobs defined by the BatchIndexOperator.
    """

    def __init__(
        self,
        job_name: str,
        batch_index_row: BatchIndexRow,
        batch_job_spec: BatchJobOperatorSpec,
        project_id: str = GCP_PROJECT_GENETICS,
        region: str = GCP_REGION,
        labels: Labels | None = None,
        **kwargs,
    ) -> None:
        # fall back to the labels declared in the job spec so the config keeps a say
        self.labels = labels or Labels(batch_job_spec.job.labels)
        job = batch_job_spec.job.build(task_environments=batch_index_row.environments, labels=dict(self.labels))
        super().__init__(
            project_id=project_id,
            region=region,
            job_name=f'{job_name}-job-{batch_index_row.idx}-{time.strftime("%Y%m%d-%H%M%S")}',
            job=job,
            deferrable=False,
            **kwargs,
        )

    def execute(self, context: Context) -> dict:
        """Stamp the run label on the job, then submit it.

        The job payload is built in ``__init__`` because Airflow templates it, but the run
        label is only knowable at execution time, so it is applied to the already-rendered
        payload here. It goes on both the job and the allocation policy: Batch only puts the
        allocation policy labels on the VMs and disks it creates, and those are the resources
        the billing export charges for.

        Args:
            context: Airflow's task rendering context.

        Returns:
            The finished job, as returned by the parent operator.
        """
        labels = Labels({**self.labels})
        labels.add_dag_run_id(context)
        self.labels = labels
        # the parent operator normalises the payload to a dict so that airflow can template it
        job: dict[str, Any] = batch_v1.Job.to_dict(self.job) if isinstance(self.job, batch_v1.Job) else dict(self.job)
        job['labels'] = dict(labels)
        job.setdefault('allocation_policy', {})['labels'] = dict(labels)
        self.job = job
        return super().execute(context)
