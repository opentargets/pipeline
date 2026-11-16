"""Tests that a step runs when a step it depends on ran in the same DagRun."""

from typing import cast
from unittest.mock import MagicMock

from airflow.models import DagBag
from airflow.sdk import Context

from orchestration.operators.diff import DiffOperator


class _TaskInstance:
    """Holds XComs per task id, pulling the way the Airflow 3 task SDK does.

    Without `task_ids` the SDK reads only the calling task's own XComs.
    """

    def __init__(self, task_id: str, xcoms: dict[str, list[str]]) -> None:
        self.task_id = task_id
        self.xcoms = xcoms
        self.pushed: list[str] | None = None

    def xcom_pull(self, key: str, task_ids: list[str] | None = None):
        if task_ids is None:
            return self.xcoms.get(self.task_id)
        return [self.xcoms.get(t) for t in task_ids]

    def xcom_push(self, key: str, value: list[str]) -> None:
        self.pushed = value


def _operator(step: str, depends_on: list[str]) -> DiffOperator:
    config = MagicMock()
    config.step_definition.return_value = {'depends_on': depends_on}
    op = DiffOperator(task_id=f'diff_{step}', step_name=step, differs=[], config=config)
    op.client = MagicMock()
    return op


def test_a_step_runs_when_a_step_it_depends_on_ran() -> None:
    op = _operator('pts_b', ['pts_a'])
    ti = _TaskInstance('pts_b.diff_pts_b', {'pts_a.diff_pts_a': ['pts_a']})

    assert op.choose_branch(cast(Context, {'task_instance': ti})) == 'pts_b.upload_config_pts_b'
    assert ti.pushed is not None
    assert 'pts_b' in ti.pushed


def test_a_step_is_skipped_when_no_step_it_depends_on_ran() -> None:
    op = _operator('pts_b', ['pts_a'])
    ti = _TaskInstance('pts_b.diff_pts_b', {})

    assert op.choose_branch(cast(Context, {'task_instance': ti})) == 'pts_b.end_pts_b'


def test_upstream_diff_task_ids_exist_in_the_dag(dag_bag: DagBag) -> None:
    """The pulled task ids must be real, or the cascade silently reads nothing."""
    dag = dag_bag.dags['unified_pipeline']
    diff_tasks = {t.task_id for t in dag.tasks if t.task_id.split('.')[-1].startswith('diff_')}
    checked = 0
    for task in dag.tasks:
        if not isinstance(task, DiffOperator):
            continue
        for step in task.config.step_definition(task.step_name).get('depends_on', []):
            assert f'{step}.diff_{step}' in diff_tasks, f'{task.task_id} depends on {step}, which has no diff task'
            checked += 1
    assert checked > 1, f'expected several depends_on edges, checked {checked}'
