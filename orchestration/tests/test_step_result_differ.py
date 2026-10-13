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

    Narrowing this to `failure` would let a hung step, which leaves `pending`, go on
    skipping itself.
    """
    with _manifest({'steps': {'pts_x': {'result': result, 'artifacts': []}}}):
        assert _is_diff('pts_x') is True


def test_a_succeeded_step_does_not_need_to_run() -> None:
    """The differ must not force a rerun of work that is already done.

    Without this, always reporting a difference would pass the test above.
    """
    with _manifest({'steps': {'pts_x': {'result': 'success', 'artifacts': [{'destination': 'gs://b/o'}]}}}):
        assert _is_diff('pts_x') is False


def test_the_differ_is_registered_for_pis_and_pts_only(dag_bag: DagBag) -> None:
    """Both directions matter, so they are checked together on the built DAG.

    Omitting it from a pis or pts stage reinstates the skip. Adding it to gentropy,
    whose steps are absent from otter's manifest, stops those ever skipping.
    """
    dag = dag_bag.dags['unified_pipeline']
    seen = {'pis_pts': 0, 'gentropy': 0}

    for task in dag.tasks:
        name = task.task_id.split('.')[-1]
        if not name.startswith('diff_'):
            continue
        step_name = name.removeprefix('diff_')

        # Absent must fail, not pass vacuously -- the shape of the bug being fixed.
        differs = getattr(task, 'differs', None)
        assert differs is not None, f'{task.task_id} exposes no differs to check'
        kinds = {type(d).__name__ for d in differs}

        if step_name.startswith(('pis_', 'pts_')):
            assert 'StepResultDiffer' in kinds, f'{task.task_id} has only {sorted(kinds)}'
            seen['pis_pts'] += 1
        elif step_name.startswith('gentropy'):
            assert 'StepResultDiffer' not in kinds, f'{task.task_id} should not carry it'
            seen['gentropy'] += 1

    assert seen['pis_pts'] > 1, f'expected many pis/pts diff tasks, saw {seen["pis_pts"]}'
    assert seen['gentropy'] > 1, f'expected many gentropy diff tasks, saw {seen["gentropy"]}'
