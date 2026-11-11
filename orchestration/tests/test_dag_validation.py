"""Try to parse and validate the DAGs."""

from datetime import timedelta
from typing import cast

from airflow.models import DagBag

from orchestration.operators.dataproc import CreateClusterOperator
from orchestration.operators.gce import ComputeEngineRunContainerizedWorkloadSensor
from orchestration.operators.gcs import UploadFileOperator


def test_no_import_errors(dag_bag: DagBag) -> None:
    """Test for import errors."""
    assert not dag_bag.import_errors, f'DAG import failures. Errors: {dag_bag.import_errors}'
    assert len(dag_bag.dags) > 0, 'No DAGs found. Check the DAG folder path and ensure DAGs are defined correctly.'


def test_requires_tags(dag_bag: DagBag) -> None:
    """Tags should be defined for each DAG."""
    for dag in dag_bag.dags.values():
        assert dag.tags, 'DAG should have at least one tag defined.'


def test_owner_len_greater_than_five(dag_bag: DagBag) -> None:
    """Owner should be defined for each DAG and be longer than 5 characters."""
    for dag in dag_bag.dags.values():
        assert len(dag.owner) > 5, 'DAG owner should be longer than 5 characters.'


def test_desc_len_greater_than_fifteen(dag_bag: DagBag) -> None:
    """Description should be defined for each DAG and be longer than 30 characters."""
    for dag in dag_bag.dags.values():
        if isinstance(dag.description, str):
            assert len(dag.description) > 30


def test_owner_not_airflow(dag_bag: DagBag) -> None:
    """Owner should not be 'airflow'."""
    for dag in dag_bag.dags.values():
        assert str.lower(dag.owner) != 'airflow'


def test_three_or_less_retries(dag_bag: DagBag) -> None:
    """Retries should be 3 or less."""
    for dag in dag_bag.dags.values():
        assert dag.default_args['retries'] <= 3


def test_pts_init_action_is_uploaded_before_every_pts_cluster(dag_bag: DagBag) -> None:
    """Every pts step creates its cluster itself, so each create must wait for the upload."""
    dag = dag_bag.dags['unified_pipeline']
    upload = cast(UploadFileOperator, dag.get_task('upload_pts_init_action'))
    assert str(upload.src_path).endswith('assets/install_dependencies_on_cluster.sh')
    assert str(upload.dst_uri).endswith('/etc/bin/install_dependencies_on_cluster.sh')
    assert upload.retries

    creates = {t.task_id for t in dag.tasks if t.task_id.startswith('pts_') and '.create_cluster_' in t.task_id}
    assert creates, 'expected pts steps that create a cluster'
    assert upload.downstream_task_ids == creates
    for task_id in creates:
        cluster_config = cast(dict, cast(CreateClusterOperator, dag.get_task(task_id)).cluster_config)
        actions = cluster_config['initialization_actions']
        assert [a['executable_file'] for a in actions] == [str(upload.dst_uri)], task_id


def test_clinical_report_runs_on_a_vm(dag_bag: DagBag) -> None:
    """clinical_report runs in a pts container on its own VM, with no cluster."""
    dag = dag_bag.dags['unified_pipeline']
    group = [t.task_id for t in dag.tasks if t.task_id.startswith('pts_clinical_report.')]
    assert not [t for t in group if '.create_cluster_' in t], group
    run = cast(
        ComputeEngineRunContainerizedWorkloadSensor,
        dag.get_task('pts_clinical_report.run_pts_clinical_report'),
    )
    assert run.machine_type == 'n2d-standard-16'


def test_aact_moa_runs_after_the_extraction(dag_bag: DagBag) -> None:
    """aact_moa waits for the extraction, runs with its timeout and the Claude workspace, and gates nothing else."""
    dag = dag_bag.dags['clinical_staging']
    start = dag.get_task('pts_aact_moa.upload_config_pts_aact_moa')
    assert start.upstream_task_ids == {'pts_aact_trial_extraction.end_pts_aact_trial_extraction'}
    run = cast(ComputeEngineRunContainerizedWorkloadSensor, dag.get_task('pts_aact_moa.run_pts_aact_moa'))
    assert run.execution_timeout == timedelta(hours=30)
    assert run.container_env is not None
    assert run.container_env['PTS_STEP'] == 'aact_moa'
    assert run.container_env['ANTHROPIC_GOOGLE_CLOUD_PROJECT'] == 'open-targets-eu-dev'
    assert run.machine_type == 'n1-highmem-8'
    assert not dag.get_task('pts_aact_moa.end_pts_aact_moa').downstream_task_ids
    other = cast(
        ComputeEngineRunContainerizedWorkloadSensor,
        dag.get_task('pts_aact_trial_models.run_pts_aact_trial_models'),
    )
    assert other.execution_timeout is None


def test_drug_mechanism_of_action_follows_its_candidate_inputs(dag_bag: DagBag) -> None:
    """drug_mechanism_of_action waits for target, clinical_report and chembl_molecule, and the DAG stays acyclic."""
    dag = dag_bag.dags['unified_pipeline']
    moa_run = dag.get_task('pts_drug_mechanism_of_action.run_pts_drug_mechanism_of_action')
    upstream = moa_run.get_flat_relative_ids(upstream=True)
    for step in ('pts_target', 'pts_clinical_report', 'pts_chembl_molecule'):
        assert f'{step}.end_{step}' in upstream
    assert not any(task_id.startswith('pts_drug_mechanism_of_action_aact') for task_id in dag.task_ids)
    dag.topological_sort()


def test_moa_literature_waits_for_its_copies_with_its_timeout_and_env(dag_bag: DagBag) -> None:
    """moa_literature waits for the pis copy of its inputs only, runs with its timeout and the Claude workspace,
    and gates nothing."""
    dag = dag_bag.dags['clinical_staging']
    start = dag.get_task('pts_moa_literature.upload_config_pts_moa_literature')
    assert start.upstream_task_ids == {'pis_moa_literature.end_pis_moa_literature'}
    assert not dag.get_task('pis_moa_literature.upload_config_pis_moa_literature').upstream_task_ids
    run = cast(ComputeEngineRunContainerizedWorkloadSensor, dag.get_task('pts_moa_literature.run_pts_moa_literature'))
    assert run.execution_timeout == timedelta(hours=30)
    assert run.container_env is not None
    assert run.container_env['PTS_STEP'] == 'moa_literature'
    assert run.container_env['ANTHROPIC_GOOGLE_CLOUD_PROJECT'] == 'open-targets-eu-dev'
    assert run.machine_type == 'n2d-highmem-32'
    assert not dag.get_task('pts_moa_literature.end_pts_moa_literature').downstream_task_ids


def test_only_the_llm_steps_carry_a_vm_service_account(dag_bag: DagBag) -> None:
    """The two LLM tasks attach the dedicated account; every other VM task keeps the default."""
    account = 'clinical-staging-llm@open-targets-eu-dev.iam.gserviceaccount.com'
    llm_runs = {'pts_aact_moa.run_pts_aact_moa', 'pts_moa_literature.run_pts_moa_literature'}
    dag = dag_bag.dags['clinical_staging']
    runs = [t for t in dag.tasks if isinstance(t, ComputeEngineRunContainerizedWorkloadSensor)]
    assert len(runs) > 2
    for task in runs:
        assert task.vm_service_account == (account if task.task_id in llm_runs else None), task.task_id
