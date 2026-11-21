"""Tests for the release-candidate DAG and the unified pipeline's switch to it."""

import copy
import json
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from airflow.models import DagBag
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.task.trigger_rule import TriggerRule
from google.api_core.exceptions import PreconditionFailed

from orchestration.dags import release_candidate
from orchestration.dags.config.release import HandOff, ReleaseConfig
from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig
from orchestration.operators.gce import DeleteInstanceOperator
from orchestration.operators.gce_backend import ComputeEngineRunBackendSensor
from orchestration.utils.release import candidate_names

URI = 'gs://open-targets-pipeline-runs/do/platform-2612-3'
PPP_URI = 'gs://open-targets-pipeline-runs/sz/ppp-2612-1'
# images and sizes as captured by `names`; distinct from the config so a test sees where they come from
CAPTURED = {
    'pos_image': 'europe-west1-docker.pkg.dev/p/r/pos:captured',
    'pos_version': 'captured',
    'clickhouse_image': 'clickhouse/clickhouse-server:captured',
    'opensearch_image': 'opensearchproject/opensearch:captured',
    'clickhouse_build_disk_gb': 678,
    'clickhouse_disk_gb': 123,
    'opensearch_disk_gb': 45,
}
NAMES = {**asdict(candidate_names(URI)), **CAPTURED}
PPP_NAMES = {**asdict(candidate_names(PPP_URI)), **CAPTURED}
RUN_ID = 'manual__2026-11-20T10:00:00+00:00'
DAGS = Path('src/orchestration/dags')


class _TI:
    """XComs keyed by task id (return value) or by `<task id>:<key>`, and this run's task states."""

    dag_id = 'release_candidate'
    run_id = RUN_ID

    def __init__(self, xcoms: dict[str, Any], states: Mapping[str, str | None] | None = None) -> None:
        self.xcoms = xcoms
        self.states = states or {}

    def get_task_states(self, dag_id: str, task_ids: list[str], run_ids: list[str]) -> dict[str, Any]:
        assert (dag_id, run_ids) == (self.dag_id, [self.run_id])
        return {self.run_id: {t: self.states[t] for t in task_ids if t in self.states}}

    def xcom_pull(self, task_ids: str, key: str = 'return_value', **_: Any) -> Any:
        return self.xcoms.get(task_ids if key == 'return_value' else f'{task_ids}:{key}')


CREATED = {
    'create_disk_clickhouse_build:disk': NAMES['ch_build_disk'],
    'create_disk_clickhouse:disk': NAMES['ch_disk'],
    'create_disk_opensearch:disk': NAMES['os_disk'],
}
DISK_TASKS = ('clickhouse_build', 'clickhouse', 'opensearch')


def _context(names: dict[str, Any] = NAMES, xcoms: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        'params': {'release_uri': names['release_uri']},
        'run_id': RUN_ID,
        'ti': _TI({'names': names, **(xcoms or {})}),
    }


@pytest.fixture(scope='module')
def dag():
    bag = DagBag(dag_folder=str(DAGS / 'release_candidate.py'))
    assert not bag.import_errors
    return bag.dags['release_candidate']


def _render(dag, task_id: str, field: str, names: dict[str, Any] = NAMES, xcoms: dict[str, Any] | None = None) -> Any:
    task = dag.get_task(task_id)
    # rendering nested template fields mutates the value; keep the DAG's copy raw
    value = copy.deepcopy(getattr(task, field))
    return task.render_template(value, _context(names, xcoms), dag.get_template_env())


def test_release_uri_is_a_required_param(dag) -> None:
    assert 'release_uri' in dag.params
    assert dag.params.get_param('release_uri').has_value is False
    assert dag.tags == {'release'}
    assert dag.max_active_runs != 1


def test_order(dag) -> None:
    up = {t.task_id: t.upstream_task_ids for t in dag.tasks}
    assert up['guard'] == {'names'}
    for d in DISK_TASKS:
        assert up[f'create_disk_{d}'] == {'guard'}
    assert up['run_backend'] == {f'create_disk_{d}' for d in DISK_TASKS}
    assert up['delete_vm_backend'] == {'run_backend'}
    for s in ('snapshot_clickhouse', 'snapshot_opensearch'):
        assert 'delete_vm_backend' in up[s]
        # a failed backend must not reach the snapshots through the ALL_DONE VM delete
        assert 'run_backend' in up[s]
    assert {'snapshot_clickhouse', 'snapshot_opensearch'} <= up['write_hand_off']
    for d in DISK_TASKS:
        assert {'delete_vm_backend', 'snapshot_clickhouse', 'snapshot_opensearch'} <= up[f'delete_disk_{d}']
        assert dag.get_task(f'delete_disk_{d}').trigger_rule == TriggerRule.ALL_DONE
    assert dag.get_task('delete_vm_backend').trigger_rule == TriggerRule.ALL_DONE
    assert isinstance(dag.get_task('delete_vm_backend'), DeleteInstanceOperator)
    assert dag.get_task('write_hand_off').trigger_rule == TriggerRule.ALL_SUCCESS
    dag.topological_sort()


def test_done_is_the_only_leaf_and_fails_on_any_failure(dag) -> None:
    assert [t.task_id for t in dag.leaves] == ['done']
    done = dag.get_task('done')
    assert done.trigger_rule == TriggerRule.ALL_SUCCESS
    assert done.upstream_task_ids == {
        'write_hand_off',
        'delete_vm_backend',
        'delete_disk_clickhouse_build',
        'delete_disk_clickhouse',
        'delete_disk_opensearch',
        'fail_cleanup_snapshots',
    }


def test_names_captures_the_images_and_sizes() -> None:
    rc = ReleaseConfig()
    up = UnifiedPipelineConfig()
    assert release_candidate.candidate(URI) == {
        **asdict(candidate_names(URI)),
        'pos_image': up.pos_image,
        'pos_version': up.pos_version,
        'clickhouse_image': rc.clickhouse_image,
        'opensearch_image': rc.opensearch_image,
        'clickhouse_build_disk_gb': rc.clickhouse_build_disk_gb,
        'clickhouse_disk_gb': rc.clickhouse_disk_gb,
        'opensearch_disk_gb': rc.opensearch_disk_gb,
    }


def test_names_are_templated_from_the_names_task(dag) -> None:
    assert _render(dag, 'create_disk_clickhouse_build', 'disk_name') == NAMES['ch_build_disk']
    assert _render(dag, 'create_disk_clickhouse', 'disk_name') == NAMES['ch_disk']
    assert _render(dag, 'create_disk_opensearch', 'disk_name') == NAMES['os_disk']
    assert _render(dag, 'snapshot_clickhouse', 'snapshot_name') == NAMES['ch_snapshot']
    assert _render(dag, 'snapshot_clickhouse', 'disk_name') == NAMES['ch_disk']
    assert _render(dag, 'snapshot_opensearch', 'snapshot_name') == NAMES['os_snapshot']
    assert _render(dag, 'snapshot_opensearch', 'disk_name') == NAMES['os_disk']
    assert _render(dag, 'guard', 'snapshot_names') == [NAMES['ch_snapshot'], NAMES['os_snapshot']]
    assert _render(dag, 'guard', 'disk_names') == [NAMES['ch_build_disk'], NAMES['ch_disk'], NAMES['os_disk']]
    assert _render(dag, 'guard', 'blob_uri') == HandOff.uri(URI)
    assert _render(dag, 'run_backend', 'data_disks') == {
        'clickhouse': NAMES['ch_build_disk'],
        'clickhouse-compact': NAMES['ch_disk'],
        'opensearch': NAMES['os_disk'],
    }


def test_trailing_slash_release_uri_finds_the_hand_off(dag) -> None:
    names = release_candidate.candidate(f'{URI}/')
    assert names['release_uri'] == URI
    context = {'params': {'release_uri': f'{URI}/'}, 'run_id': RUN_ID, 'ti': _TI({'names': names})}
    task = dag.get_task('guard')
    blob_uri = task.render_template(copy.deepcopy(task.blob_uri), context, dag.get_template_env())
    assert blob_uri == f'{URI}/release_candidate.json'


def test_disks_are_sized_from_the_names_task(dag) -> None:
    assert str(_render(dag, 'create_disk_clickhouse_build', 'size_gb')) == '678'
    assert str(_render(dag, 'create_disk_clickhouse', 'size_gb')) == '123'
    assert str(_render(dag, 'create_disk_opensearch', 'size_gb')) == '45'


CREATED_OK = {
    'guard': 'success',
    'create_disk_clickhouse_build': 'success',
    'create_disk_clickhouse': 'success',
    'create_disk_opensearch': 'failed',
}


@pytest.mark.parametrize(
    ('states', 'deleted'),
    [
        # created this try, and a create whose wait failed after the insert
        (CREATED_OK, [NAMES['ch_build_disk'], NAMES['ch_disk'], NAMES['os_disk']]),
        # a refused guard after a clear: the creates never started, their old XComs remain
        ({'guard': 'failed', **dict.fromkeys(f'create_disk_{d}' for d in DISK_TASKS)}, []),
        ({'guard': 'upstream_failed'}, []),
        # guard passed, but a create was skipped or cleared and kept an old XCom
        ({'guard': 'success', 'create_disk_clickhouse_build': None, 'create_disk_clickhouse': 'upstream_failed',
          'create_disk_opensearch': None}, []),
    ],
)
def test_disk_deletes_remove_only_disks_this_try_created(states, deleted) -> None:
    calls = []
    with patch.object(release_candidate, 'delete_disk') as delete:
        for device in DISK_TASKS:
            release_candidate.cleanup_disk(_TI(CREATED, states), device)
        calls = [c.args[2] for c in delete.call_args_list]
    assert calls == deleted


def test_a_create_that_found_the_disk_pushed_nothing_and_deletes_nothing() -> None:
    states = {'guard': 'success', 'create_disk_clickhouse': 'failed'}
    with patch.object(release_candidate, 'delete_disk') as delete:
        release_candidate.cleanup_disk(_TI({}, states), 'clickhouse')
    delete.assert_not_called()


def test_run_backend(dag) -> None:
    rc = ReleaseConfig()
    up = UnifiedPipelineConfig()
    run = cast(ComputeEngineRunBackendSensor, dag.get_task('run_backend'))
    assert isinstance(run, ComputeEngineRunBackendSensor)
    assert run.servers is True
    assert run.read_only is False
    assert run.compact == {'clickhouse': 'clickhouse-compact'}
    assert run.machine_type == 'n2d-highmem-96'
    assert run.max_run_duration == timedelta(hours=4)
    assert run.retries == 0
    assert run.vm_service_account is None
    assert run.work_disk_size_gb == rc.work_disk_gb
    assert run.boot_disk_size_gb == rc.boot_disk_gb
    assert run.source_image == up.cos_image
    assert run.deferrable is True
    assert run.labels['tool'] == 'pos'
    assert run.labels['step'] == 'backend'
    assert run.container_args is not None
    assert run.container_args[:4] == ['-s', 'backend', '-r', '{{ params.release_uri }}']
    args = ' '.join(_render(dag, 'run_backend', 'container_args'))
    assert args == f'-s backend -r {URI} -w /mnt/disks/work -p 8'


def test_backend_and_hand_off_record_the_same_images(dag) -> None:
    rendered = {
        'pos_image': _render(dag, 'run_backend', 'container_image'),
        'clickhouse_image': _render(dag, 'run_backend', 'clickhouse_image'),
        'opensearch_image': _render(dag, 'run_backend', 'opensearch_image'),
    }
    assert rendered == {k: CAPTURED[k] for k in rendered}
    body = _upload(NAMES)
    assert {k: body[k] for k in rendered} == rendered
    assert body['pos_version'] == CAPTURED['pos_version']
    assert body['disk_size_gb'] == {'clickhouse': 123, 'opensearch': 45}


@pytest.mark.parametrize(('names', 'product'), [(NAMES, 'platform'), (PPP_NAMES, 'ppp')])
def test_labels_carry_the_product(dag, names: dict[str, Any], product: str) -> None:
    vm = _render(dag, 'run_backend', 'labels', names)
    assert vm['product'] == product
    assert (vm['tool'], vm['step']) == ('pos', 'backend')
    run = 'manual__2026-11-20t10-00-00-00-00'
    for task_id in (f'create_disk_{d}' for d in DISK_TASKS):
        assert dict(_render(dag, task_id, 'labels', names)) == {**vm, 'run': run}
    for task_id in ('snapshot_clickhouse', 'snapshot_opensearch'):
        labels = dict(_render(dag, task_id, 'labels', names))
        assert labels == {**vm, 'run': run, 'release': '2612'}
        assert dag.get_task(task_id).trigger_rule == TriggerRule.ALL_SUCCESS


def test_vm_labels_are_cleaned_after_rendering(dag) -> None:
    vm = _render(dag, 'run_backend', 'labels', {**NAMES, 'product': 'Not Clean'})
    assert vm['product'] == 'not-clean'


def test_snapshot_cleanup_waits_for_both_snapshots_and_the_hand_off(dag) -> None:
    task = dag.get_task('fail_cleanup_snapshots')
    assert task.trigger_rule == TriggerRule.ALL_DONE
    assert {'snapshot_clickhouse', 'snapshot_opensearch', 'write_hand_off'} <= task.upstream_task_ids
    assert task.downstream_task_ids == {'done'}


SNAPPED = {'snapshot_clickhouse': NAMES['ch_snapshot'], 'snapshot_opensearch': NAMES['os_snapshot']}


@pytest.mark.parametrize(
    ('states', 'deleted'),
    [
        # one snapshot failed while the other was running: ALL_DONE waits, the finished one goes
        (
            {'guard': 'success', 'snapshot_clickhouse': 'success', 'snapshot_opensearch': 'failed',
             'write_hand_off': 'upstream_failed'},
            [NAMES['ch_snapshot']],
        ),
        # both made, the hand-off failed
        (
            {'guard': 'success', 'snapshot_clickhouse': 'success', 'snapshot_opensearch': 'success',
             'write_hand_off': 'failed'},
            [NAMES['ch_snapshot'], NAMES['os_snapshot']],
        ),
        # a finished candidate
        (
            {'guard': 'success', 'snapshot_clickhouse': 'success', 'snapshot_opensearch': 'success',
             'write_hand_off': 'success'},
            [],
        ),
        # a cleared run whose guard refused: the old XComs of the published candidate remain
        (
            {'guard': 'failed', 'snapshot_clickhouse': 'upstream_failed', 'snapshot_opensearch': 'upstream_failed',
             'write_hand_off': 'upstream_failed'},
            [],
        ),
        # the guard passed but the snapshots never ran this try: their old XComs are ignored
        (
            {'guard': 'success', 'snapshot_clickhouse': 'upstream_failed', 'snapshot_opensearch': None,
             'write_hand_off': 'upstream_failed'},
            [],
        ),
    ],
)  # fmt: skip
def test_snapshot_cleanup_deletes_only_what_this_try_created(states, deleted) -> None:
    ti = _TI({k: v for k, v in SNAPPED.items() if states.get(k) == 'success' or states['guard'] == 'failed'}, states)
    with patch.object(release_candidate, 'delete_snapshot') as delete:
        release_candidate.cleanup_snapshots(ti)
    assert [c.args[1] for c in delete.call_args_list] == deleted


def _upload(names: dict[str, Any]) -> dict[str, Any]:
    now = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
    with patch.object(release_candidate, 'Client') as client:
        uri = release_candidate.upload_hand_off(names, 'manual__run', now=now)
    assert uri == HandOff.uri(names['release_uri'])
    client.return_value.bucket.assert_called_once_with('open-targets-pipeline-runs')
    bucket = client.return_value.bucket.return_value
    bucket.blob.assert_called_once_with(f'{names["prefix"]}/{names["run"]}/release_candidate.json')
    upload = cast(MagicMock, bucket.blob.return_value.upload_from_string)
    assert upload.call_args.kwargs['if_generation_match'] == 0
    return HandOff.model_validate(json.loads(upload.call_args.args[0])).model_dump(mode='json')


def test_hand_off_records_the_snapshotted_disk_sizes() -> None:
    assert _upload(release_candidate.candidate(URI))['disk_size_gb'] == {'clickhouse': 170, 'opensearch': 80}


def test_hand_off_is_built_and_uploaded_without_overwrite() -> None:
    assert _upload(NAMES) == {
        'release_uri': URI,
        'run': 'do/platform-2612-3',
        'product': 'platform',
        'release': '26.12',
        'snapshots': {
            'clickhouse': 'projects/open-targets-eu-dev/global/snapshots/do-platform-2612-3-ch',
            'opensearch': 'projects/open-targets-eu-dev/global/snapshots/do-platform-2612-3-os',
        },
        'disk_size_gb': {'clickhouse': 123, 'opensearch': 45},
        'pos_image': CAPTURED['pos_image'],
        'pos_version': CAPTURED['pos_version'],
        'clickhouse_image': CAPTURED['clickhouse_image'],
        'opensearch_image': CAPTURED['opensearch_image'],
        'created': '2026-11-20T12:00:00Z',
        'airflow_run_id': 'manual__run',
    }


def _unified(monkeypatch: pytest.MonkeyPatch, release_candidate: bool):
    init = UnifiedPipelineConfig.__init__

    def patched(self: UnifiedPipelineConfig) -> None:
        init(self)
        self.release_candidate = release_candidate

    monkeypatch.setattr(UnifiedPipelineConfig, '__init__', patched)
    bag = DagBag(dag_folder=str(DAGS / 'unified_pipeline.py'))
    assert not bag.import_errors
    return bag.dags['unified_pipeline']


def test_unified_pipeline_does_not_trigger_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    assert UnifiedPipelineConfig().release_candidate is False
    assert 'trigger_release_candidate' not in _unified(monkeypatch, False).task_ids


def test_unified_pipeline_triggers_the_release_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    dag = _unified(monkeypatch, True)
    task = cast(TriggerDagRunOperator, dag.get_task('trigger_release_candidate'))
    assert isinstance(task, TriggerDagRunOperator)
    assert task.trigger_dag_id == 'release_candidate'
    assert task.conf == {'release_uri': UnifiedPipelineConfig().release_uri}
    assert task.wait_for_completion is False
    assert task.trigger_rule == TriggerRule.ALL_SUCCESS
    ends = {t.task_id for t in dag.tasks if t.task_id.split('.')[-1].startswith('end_')}
    assert ends
    assert task.upstream_task_ids == ends


def test_hand_off_upload_uses_the_stripped_release_uri() -> None:
    names = {**NAMES, **release_candidate.candidate(f'{URI}/'), **CAPTURED}
    assert _upload(names)['release_uri'] == URI


@pytest.mark.parametrize('writer', ['manual__run', 'manual__other'])
def test_hand_off_already_written(writer: str) -> None:
    existing = HandOff.model_validate({**_upload(NAMES), 'airflow_run_id': writer}).model_dump_json()
    with patch.object(release_candidate, 'Client') as client:
        blob = client.return_value.bucket.return_value.blob.return_value
        blob.upload_from_string.side_effect = PreconditionFailed('exists')
        blob.download_as_text.return_value = existing
        if writer == 'manual__run':
            assert release_candidate.upload_hand_off(NAMES, 'manual__run') == HandOff.uri(URI)
        else:
            with pytest.raises(PreconditionFailed):
                release_candidate.upload_hand_off(NAMES, 'manual__run')
