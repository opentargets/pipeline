"""Unit test configuration for the project."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from typing import Any

import pytest
from airflow.models import DagBag
from airflow.task.trigger_rule import TriggerRule


@pytest.fixture(params=['src/orchestration/dags'])
def dag_bag(request: pytest.FixtureRequest) -> DagBag:
    """Return a DAG bag for testing."""
    # example dags are their own bundles since airflow 3.3, so a folder bag never loads them
    return DagBag(dag_folder=request.param)


def _simulate(
    dag: Any, gates: Mapping[str, bool], failed: Collection[str] = frozenset(), prefix: str = 'gate_'
) -> dict[str, str]:
    """Final task states, by the trigger-rule outcomes of `trigger_rule_dep` once every upstream is done.

    Tasks in `failed` fail when they run; every other task succeeds. `gates` maps a gate's task id,
    less `prefix`, to whether it is open. A closed gate skips its direct downstream, as
    `ShortCircuitOperator(ignore_downstream_trigger_rules=False)`.
    """
    states: dict[str, str] = {}
    gate_skipped: set[str] = set()
    for t in dag.topological_sort():
        up = [states[u] for u in t.upstream_task_ids]
        bad = any(s in {'failed', 'upstream_failed'} for s in up)
        rule = t.trigger_rule
        if t.task_id in gate_skipped:
            state = 'skipped'
        elif rule == TriggerRule.ALL_SUCCESS and bad:
            state = 'upstream_failed'
        elif rule == TriggerRule.ALL_SUCCESS and 'skipped' in up:
            state = 'skipped'
        elif rule == TriggerRule.NONE_FAILED and bad:
            state = 'upstream_failed'
        elif rule == TriggerRule.ALL_DONE_MIN_ONE_SUCCESS and 'skipped' in up:
            state = 'skipped'
        elif rule == TriggerRule.ALL_DONE_MIN_ONE_SUCCESS and 'success' not in up:
            state = 'upstream_failed'
        else:
            assert rule in {
                TriggerRule.ALL_SUCCESS, TriggerRule.ALL_DONE, TriggerRule.NONE_FAILED,
                TriggerRule.ALL_DONE_MIN_ONE_SUCCESS,
            }  # fmt: skip
            state = 'failed' if t.task_id in failed else 'success'
            if state == 'success' and t.task_id.startswith(prefix) and not gates[t.task_id.removeprefix(prefix)]:
                gate_skipped |= t.downstream_task_ids
        states[t.task_id] = state
    return states


@pytest.fixture
def simulate() -> Callable[..., dict[str, str]]:
    """Simulate a DAG run's final task states; see `_simulate`."""
    return _simulate
