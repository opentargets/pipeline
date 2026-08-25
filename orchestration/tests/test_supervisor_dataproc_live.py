"""Live checks against the real Dataproc Job Controller API.

Skipped unless RUN_DATAPROC_TESTS is set, because these need credentials and network.
Run with: RUN_DATAPROC_TESTS=1 uv run --frozen pytest tests/test_supervisor_dataproc_live.py -rxs
"""

from __future__ import annotations

import os

import pytest
from google.cloud import dataproc_v1

from orchestration.supervisor.dataproc import job_execution, job_executions
from orchestration.utils.common import GCP_PROJECT_PLATFORM, GCP_REGION

pytestmark = pytest.mark.skipif(
    not os.environ.get('RUN_DATAPROC_TESTS'),
    reason='needs Dataproc credentials, set RUN_DATAPROC_TESTS=1 to run',
)

KNOWN_JOB_ID = 'up-pts-f5014-pts_drug_molecule-c516h'
"""A real job, verified live 2026-08-24: state DONE, ~19m22s of RUNNING to DONE."""

KNOWN_RUN = 'manual__2026-07-21t15-07-47-545737-00-00'
"""The same cleaned run label `test_supervisor_usage_live.py`'s `KNOWN_RUN` verifies.

Both come from the same `Labels.add_dag_run_id`/`clean_label` path, which is why a
`StepUsage.run` value can be passed straight into `job_executions` unchanged -- see
that function's docstring.
"""

KNOWN_RUN_DRUG_MOLECULE_JOB_STATES = {'DONE', 'CANCELLED'}
"""Verified live 2026-08-24: this run's `pts_drug_molecule` jobs are exactly a
`DONE` job (`...-c516h`) and a `CANCELLED` re-run (`...-gttbk`) -- the several-jobs
case this module's docstring documents, not a theoretical one.
"""


@pytest.fixture
def client() -> dataproc_v1.JobControllerClient:
    return dataproc_v1.JobControllerClient(client_options={'api_endpoint': f'{GCP_REGION}-dataproc.googleapis.com:443'})


class TestLiveJob:
    def test_the_known_job_has_a_recovered_step_and_a_positive_execution_time(
        self, client: dataproc_v1.JobControllerClient
    ) -> None:
        job = client.get_job(project_id=GCP_PROJECT_PLATFORM, region=GCP_REGION, job_id=KNOWN_JOB_ID)
        execution = job_execution(job)
        assert execution.job_id == KNOWN_JOB_ID
        assert execution.step == 'pts_drug_molecule'
        assert execution.state == 'DONE'
        assert execution.execution_seconds is not None
        assert execution.execution_seconds > 0


class TestLiveRun:
    def test_the_known_run_reports_both_jobs_of_the_re_run_step_separately(
        self, client: dataproc_v1.JobControllerClient
    ) -> None:
        executions = job_executions(client, project=GCP_PROJECT_PLATFORM, region=GCP_REGION, run=KNOWN_RUN)
        assert executions, 'the run bills real Dataproc jobs; an empty result means the filter stopped matching'
        drug_molecule = [e for e in executions if e.step == 'pts_drug_molecule']
        assert len(drug_molecule) == 2
        assert {e.state for e in drug_molecule} == KNOWN_RUN_DRUG_MOLECULE_JOB_STATES

    def test_every_reported_job_actually_carries_this_run_label(self, client: dataproc_v1.JobControllerClient) -> None:
        """The server-side filter, not a client-side one: nothing here re-checks the label."""
        executions = job_executions(client, project=GCP_PROJECT_PLATFORM, region=GCP_REGION, run=KNOWN_RUN)
        assert executions
        assert all(e.job_id.startswith('up-') for e in executions)
