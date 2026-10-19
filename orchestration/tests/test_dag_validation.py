"""Try to parse and validate the DAGs."""

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
