"""Tests for StepResultDiffer, which makes a failed step run when it is cleared."""

from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from airflow.models import DagBag

from orchestration.operators.differs.step_result_differ import StepResultDiffer


@contextmanager
def _manifest(contents: dict[str, Any]):
    """Serve `contents` where the differ would load the manifest from GCS."""
    path = MagicMock()
    path.load.return_value = contents
    io_manager = MagicMock()
    io_manager.resolve.return_value = path
    with patch(
        'orchestration.operators.differs.step_result_differ.IOManager',
        return_value=io_manager,
    ):
        yield


def _is_diff(step_name: str) -> bool:
    config = MagicMock()
    config.manifest_uri.return_value = 'gs://bucket/run/manifest.json'
    return StepResultDiffer().is_diff(step_name=step_name, config=config, client=MagicMock())


@pytest.mark.parametrize(
    'result',
    [
        pytest.param('failure', id='failed'),
        pytest.param('aborted', id='aborted'),
        pytest.param('pending', id='pending, a step that started and never finished'),
    ],
)
def test_a_step_that_did_not_succeed_must_run(result: str) -> None:
    """Anything other than success has to run when the step is cleared.

    `pending` matters as much as `failure`: it is what a hung step, or one whose vm
    disappeared underneath it, leaves in the manifest.
    """
    with _manifest({'steps': {'pts_x': {'result': result, 'artifacts': []}}}):
        assert _is_diff('pts_x') is True


def test_a_succeeded_step_does_not_need_to_run() -> None:
    """The differ must not force a rerun of work that is already done."""
    with _manifest({'steps': {'pts_x': {'result': 'success', 'artifacts': [{'destination': 'gs://b/o'}]}}}):
        assert _is_diff('pts_x') is False


def test_a_step_missing_from_the_manifest_must_run() -> None:
    """A step that has never run has nothing recorded, so it cannot be skipped."""
    with _manifest({'steps': {'pts_other': {'result': 'success'}}}):
        assert _is_diff('pts_x') is True


def test_the_differ_is_registered_for_every_pis_and_pts_step(dag_bag: DagBag) -> None:
    """A stage that omits the differ silently reinstates the skip.

    Checked on the built DAG rather than by reading the source, so adding a stage
    without the differ fails here rather than in a run.
    """
    dag = dag_bag.dags['unified_pipeline']
    checked = 0
    for task in dag.tasks:
        name = task.task_id.split('.')[-1]
        if not name.startswith('diff_'):
            continue
        step_name = name.removeprefix('diff_')
        if not step_name.startswith(('pis_', 'pts_')):
            continue
        differs = getattr(task, 'differs', None)
        assert differs is not None, f'{task.task_id} exposes no differs to check'
        kinds = {type(d).__name__ for d in differs}
        assert 'StepResultDiffer' in kinds, f'{task.task_id} has only {sorted(kinds)}'
        checked += 1
    assert checked > 1, f'expected many pis/pts diff tasks, checked {checked}'


def test_gentropy_steps_do_not_get_the_differ(dag_bag: DagBag) -> None:
    """Gentropy steps are absent from otter's manifest.

    Registering the differ there would make every gentropy step report a difference on
    every pass, so none of them could ever be skipped.
    """
    dag = dag_bag.dags['unified_pipeline']
    for task in dag.tasks:
        name = task.task_id.split('.')[-1]
        if name.startswith('diff_gentropy'):
            differs = getattr(task, 'differs', None)
            assert differs is not None, f'{task.task_id} exposes no differs to check'
            kinds = {type(d).__name__ for d in differs}
            assert 'StepResultDiffer' not in kinds, f'{task.task_id} should not carry it'
