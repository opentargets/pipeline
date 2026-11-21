"""The release DAGs' TaskFlow tasks bind their arguments the way Airflow does at run time."""

import json
from dataclasses import asdict
from pathlib import Path
from types import FunctionType
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from airflow.models import DagBag
from airflow.sdk import XComArg
from airflow.sdk.bases.decorator import DecoratedOperator
from airflow.sdk.definitions.context import KNOWN_CONTEXT_KEYS

from orchestration.dags.config.release import HandOff
from orchestration.utils.release import candidate_names

DAGS = Path('src/orchestration/dags')
RELEASE_DAGS = ('release_candidate', 'release')
SENTINEL_RUN_ID = 'manual__sentinel-run-id'
URI = 'gs://open-targets-pipeline-runs/do/platform-2612-3'
NAMES: dict[str, Any] = {
    **asdict(candidate_names(URI)),
    'pos_image': 'europe-west1-docker.pkg.dev/p/r/pos:t',
    'pos_version': '26.12.0rc6',
    'clickhouse_image': 'clickhouse/clickhouse-server:t',
    'opensearch_image': 'opensearchproject/opensearch:t',
    'clickhouse_disk_gb': 10,
    'opensearch_disk_gb': 10,
}
PLAN = {
    'clickhouse': {'action': 'reuse', 'disk_size_gb': 170},
    'opensearch': {'action': 'build', 'disk_size_gb': 10},
}
# upstream XComs, resolved, by producing task
XCOMS = {'names': NAMES, 'plan': PLAN}


def _dag(dag_id: str) -> Any:
    bag = DagBag(dag_folder=str(DAGS / f'{dag_id}.py'))
    assert not bag.import_errors
    return bag.dags[dag_id]


def _taskflow_tasks() -> list[tuple[str, DecoratedOperator]]:
    return [
        (f'{dag_id}.{t.task_id}', t)
        for dag_id in RELEASE_DAGS
        for t in _dag(dag_id).tasks
        if isinstance(t, DecoratedOperator)
    ]


TASKS = _taskflow_tasks()


def test_both_dags_have_taskflow_tasks() -> None:
    ids = {name for name, _ in TASKS}
    assert {
        'release_candidate.write_hand_off',
        'release_candidate.names',
        'release_candidate.plan',
        'release_candidate.build_clickhouse',
        'release.read_hand_off',
    } <= ids


@pytest.mark.parametrize('task', [t for _, t in TASKS], ids=[name for name, _ in TASKS])
def test_no_explicit_argument_shadows_a_context_key(task: DecoratedOperator) -> None:
    # every context key, as at run time; Airflow raises if a positional argument's parameter is one
    context = {key: MagicMock(name=key) for key in KNOWN_CONTEXT_KEYS}
    task.determine_kwargs(context)


def test_write_hand_off_records_the_run_id_from_the_context() -> None:
    task = cast(DecoratedOperator, _dag('release_candidate').get_task('write_hand_off'))
    context = {**{key: MagicMock(name=key) for key in KNOWN_CONTEXT_KEYS}, 'run_id': SENTINEL_RUN_ID}
    args = [XCOMS[a.operator.task_id] if isinstance(a, XComArg) else a for a in task.op_args]
    assert args == [NAMES, PLAN]
    kwargs = {**task.op_kwargs, **task.determine_kwargs(context)}
    client = MagicMock()
    # DagBag loads the DAG file as its own module; patch the one the callable runs in
    func = cast(FunctionType, task.python_callable)
    with patch.dict(func.__globals__, {'Client': client}):
        uri = func(*args, **kwargs)
    assert uri == HandOff.uri(URI)
    upload = cast(MagicMock, client.return_value.bucket.return_value.blob.return_value.upload_from_string)
    body = json.loads(upload.call_args.args[0])
    assert body['airflow_run_id'] == SENTINEL_RUN_ID
    assert body['disk_size_gb'] == {'clickhouse': 170, 'opensearch': 10}
