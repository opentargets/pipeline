"""Tests that a failed step still takes its vm down, without hiding the failure."""

from airflow.models import DagBag
from airflow.task.trigger_rule import TriggerRule


def _delete_tasks(dag_bag: DagBag) -> list:
    dag = dag_bag.dags['unified_pipeline']
    return [t for t in dag.tasks if t.task_id.split('.')[-1].startswith('delete_vm_')]


def test_delete_vm_runs_when_the_step_fails(dag_bag: DagBag) -> None:
    """Under ALL_SUCCESS the delete is skipped and the instance leaks.

    Instance names are deterministic, so a leaked vm makes the next attempt die on a
    409 before it runs anything -- the step becomes unretryable without deleting the
    instance by hand.
    """
    tasks = _delete_tasks(dag_bag)
    assert tasks, 'expected delete_vm tasks in the dag'
    for task in tasks:
        assert task.trigger_rule == TriggerRule.NONE_SKIPPED, (
            f'{task.task_id} has {task.trigger_rule}, which will not run after a failed step'
        )


def test_delete_vm_is_skipped_when_the_step_is(dag_bag: DagBag) -> None:
    """No vm is created when the differ skips a step, and the operator raises on 404.

    This is why the rule is NONE_SKIPPED rather than ALL_DONE: ALL_DONE would fire the
    delete on the skip path and fail the group on a missing instance.
    """
    for task in _delete_tasks(dag_bag):
        assert task.trigger_rule != TriggerRule.ALL_DONE, (
            f'{task.task_id} would delete a vm that was never created when the step is skipped'
        )


def test_end_still_sees_a_failed_run(dag_bag: DagBag) -> None:
    """A successful cleanup must not report the step as succeeded.

    The delete now runs after a failed run task, so any `end_` reachable only through
    the delete would see success and mark the whole group succeeded. Every `end_` needs
    a direct edge from its run task for the failure to surface.
    """
    dag = dag_bag.dags['unified_pipeline']
    checked = 0
    for task in _delete_tasks(dag_bag):
        group = task.task_id.rsplit('.', 1)[0]
        step_name = group.split('.')[-1]
        end = dag.get_task(f'{group}.end_{step_name}')
        run_id = f'{group}.run_{step_name}'
        assert run_id in end.upstream_task_ids, (
            f'{end.task_id} does not depend on {run_id}; a failed run would be masked '
            f'by the successful delete. upstream={sorted(end.upstream_task_ids)}'
        )
        checked += 1
    assert checked > 1, f'expected several delete_vm tasks, checked {checked}'
