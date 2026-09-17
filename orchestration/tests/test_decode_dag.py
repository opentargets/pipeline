"""deCODE specific tests."""

from datetime import timedelta

from airflow.models import DagBag

DECODE_CONFIG_PATH = 'src/orchestration/dags/config/decode_ingestion.yaml'


def test_decode_heavy_tasks_have_execution_timeout(dag_bag: DagBag) -> None:
    """The deCODE harmonisation tasks must be time-bounded."""
    dag = dag_bag.dags['decode_ingestion']
    tasks = {t.task_id: t for t in dag.tasks}
    heavy = ['smp_harmonisation', 'raw_harmonisation']

    for task_id in heavy:
        assert task_id in tasks, f'{task_id} missing from the decode dag'
        timeout = tasks[task_id].execution_timeout
        assert timeout is not None, f'{task_id} has no execution_timeout'
        assert timeout == timedelta(hours=4), f'{task_id} timeout is {timeout}'
