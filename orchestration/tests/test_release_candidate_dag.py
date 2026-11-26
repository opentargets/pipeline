"""Tests for the release-candidate DAG and the unified pipeline's switch to it."""

import copy
import json
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import FunctionType
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from airflow.models import DagBag
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk.bases.decorator import DecoratedOperator
from airflow.task.trigger_rule import TriggerRule
from google.api_core.exceptions import PreconditionFailed

from orchestration.dags import release_candidate
from orchestration.dags.config.release import HandOff, ReleaseConfig
from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig
from orchestration.operators.gce import DeleteInstanceOperator
from orchestration.operators.gce_backend import ComputeEngineRunBackendSensor
from orchestration.utils.common import GCP_PROJECT_PLATFORM, GCP_ZONE
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
SIDES = ('clickhouse', 'opensearch')
# side: its working disks, by task suffix
SIDE_DISKS = {'clickhouse': ('clickhouse_build', 'clickhouse'), 'opensearch': ('opensearch',)}
DISK_TASKS = ('clickhouse_build', 'clickhouse', 'opensearch')
BUILD_BOTH = {s: {'action': 'build', 'disk_size_gb': CAPTURED[f'{s}_disk_gb']} for s in SIDES}


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


def _callable(dag, task_id: str) -> FunctionType:
    return cast(FunctionType, cast(DecoratedOperator, dag.get_task(task_id)).python_callable)


def _run(simulate, dag, builds: Mapping[str, bool] | None = None, failed: set[str] | None = None) -> dict[str, str]:
    return simulate(dag, builds or dict.fromkeys(SIDES, True), failed or set(), prefix='build_')


def test_release_uri_is_a_required_param(dag) -> None:
    assert 'release_uri' in dag.params
    assert dag.params.get_param('release_uri').has_value is False
    assert dag.tags == {'release'}
    assert dag.max_active_runs != 1


def test_task_ids(dag) -> None:
    assert set(dag.task_ids) == {
        'names',
        'plan',
        'build_clickhouse',
        'build_opensearch',
        'create_disk_clickhouse_build',
        'create_disk_clickhouse',
        'create_disk_opensearch',
        'run_backend_clickhouse',
        'run_backend_opensearch',
        'delete_vm_clickhouse',
        'delete_vm_opensearch',
        'snapshot_clickhouse',
        'snapshot_opensearch',
        'delete_disk_clickhouse_build',
        'delete_disk_clickhouse',
        'delete_disk_opensearch',
        'write_hand_off',
        'done',
    }


def test_order(dag) -> None:
    up = {t.task_id: t.upstream_task_ids for t in dag.tasks}
    assert up['plan'] == {'names'}
    assert up['write_hand_off'] == {'names', 'plan', 'snapshot_clickhouse', 'snapshot_opensearch'}
    for side in SIDES:
        build = dag.get_task(f'build_{side}')
        assert up[f'build_{side}'] == {'plan'}
        assert build.ignore_downstream_trigger_rules is False
        for d in SIDE_DISKS[side]:
            assert up[f'create_disk_{d}'] == {f'build_{side}'}
            assert up[f'delete_disk_{d}'] == {f'delete_vm_{side}', f'snapshot_{side}'}
            assert dag.get_task(f'delete_disk_{d}').trigger_rule == TriggerRule.ALL_DONE
        assert up[f'run_backend_{side}'] == {f'create_disk_{d}' for d in SIDE_DISKS[side]}
        assert up[f'delete_vm_{side}'] == {f'run_backend_{side}'}
        # a failed backend must not reach the snapshot through the ALL_DONE VM delete
        assert up[f'snapshot_{side}'] == {f'run_backend_{side}', f'delete_vm_{side}'}
        assert dag.get_task(f'snapshot_{side}').trigger_rule == TriggerRule.ALL_SUCCESS
        assert dag.get_task(f'delete_vm_{side}').trigger_rule == TriggerRule.ALL_DONE
        assert isinstance(dag.get_task(f'delete_vm_{side}'), DeleteInstanceOperator)
    assert dag.get_task('write_hand_off').trigger_rule == TriggerRule.NONE_FAILED
    dag.topological_sort()


def test_done_is_the_only_leaf(dag) -> None:
    assert [t.task_id for t in dag.leaves] == ['done']
    done = dag.get_task('done')
    assert done.trigger_rule == TriggerRule.NONE_FAILED
    assert done.upstream_task_ids == {
        'write_hand_off',
        'delete_vm_clickhouse',
        'delete_vm_opensearch',
        'delete_disk_clickhouse_build',
        'delete_disk_clickhouse',
        'delete_disk_opensearch',
    }


def test_sides_are_independent(dag) -> None:
    shared = {'names', 'plan', 'write_hand_off', 'done'}
    for side, other in (SIDES, SIDES[::-1]):
        tasks = {t for t in dag.task_ids if side in t}
        assert tasks
        seen: set[str] = set()
        stack = list(tasks)
        while stack:
            for d in dag.get_task(stack.pop()).downstream_task_ids - shared - seen:
                seen.add(d)
                stack.append(d)
        assert not {t for t in seen if other in t}, side


def test_build_both(dag, simulate) -> None:
    states = _run(simulate, dag)
    assert set(states.values()) == {'success'}


def test_reuse_side_skips_its_branch(dag, simulate) -> None:
    states = _run(simulate, dag, {'clickhouse': False, 'opensearch': True})
    for t in ('create_disk_clickhouse_build', 'create_disk_clickhouse', 'run_backend_clickhouse',
              'snapshot_clickhouse'):  # fmt: skip
        assert states[t] == 'skipped', t
    for t in ('create_disk_opensearch', 'run_backend_opensearch', 'snapshot_opensearch'):
        assert states[t] == 'success', t
    assert (states['write_hand_off'], states['done']) == ('success', 'success')


def test_reuse_both_writes_hand_off_only(dag, simulate) -> None:
    states = _run(simulate, dag, dict.fromkeys(SIDES, False))
    for side in SIDES:
        assert states[f'run_backend_{side}'] == 'skipped'
        assert states[f'snapshot_{side}'] == 'skipped'
    for d in DISK_TASKS:
        assert states[f'create_disk_{d}'] == 'skipped'
    assert (states['write_hand_off'], states['done']) == ('success', 'success')


def test_reuse_side_cleanup_deletes_nothing(dag, simulate) -> None:
    states = _run(simulate, dag, {'clickhouse': False, 'opensearch': True})
    assert states['delete_vm_clickhouse'] == 'success'
    # old XComs from an earlier try of the creates remain
    with patch.object(release_candidate, 'delete_disk') as delete:
        for d in SIDE_DISKS['clickhouse']:
            assert states[f'delete_disk_{d}'] == 'success'
            release_candidate.cleanup_disk(_TI(CREATED, states), d)
    delete.assert_not_called()
    assert 'delete_snapshot' not in vars(release_candidate)


def test_one_side_fails_other_snapshot_kept(dag, simulate) -> None:
    states = _run(simulate, dag, failed={'run_backend_opensearch'})
    assert states['snapshot_clickhouse'] == 'success'
    assert states['snapshot_opensearch'] == 'upstream_failed'
    assert states['write_hand_off'] == 'upstream_failed'
    assert states['done'] == 'upstream_failed'
    assert states['delete_vm_opensearch'] == 'success'
    assert states['delete_disk_opensearch'] == 'success'
    with patch.object(release_candidate, 'delete_disk') as delete:
        release_candidate.cleanup_disk(_TI(CREATED, states), 'opensearch')
    assert [c.args[2] for c in delete.call_args_list] == [NAMES['os_disk']]


def test_plan_fails_nothing_is_created_or_deleted(dag, simulate) -> None:
    states = _run(simulate, dag, failed={'plan'})
    assert states['plan'] == 'failed'
    for t in dag.tasks:
        if t.task_id.startswith(('create_disk_', 'run_backend_', 'snapshot_')):
            assert states[t.task_id] == 'upstream_failed', t.task_id
    assert states['write_hand_off'] == states['done'] == 'upstream_failed'
    with patch.object(release_candidate, 'delete_disk') as delete:
        for d in DISK_TASKS:
            assert states[f'delete_disk_{d}'] == 'success'
            release_candidate.cleanup_disk(_TI(CREATED, states), d)
    delete.assert_not_called()


def test_plan_task_plans_in_the_platform_project(dag) -> None:
    func = _callable(dag, 'plan')
    plan = MagicMock(return_value=BUILD_BOTH)
    with patch.dict(func.__globals__, {'plan_candidate': plan}):
        assert func(NAMES) == BUILD_BOTH
    plan.assert_called_once_with(NAMES, GCP_PROJECT_PLATFORM, GCP_ZONE)


@pytest.mark.parametrize('action', ['build', 'reuse'])
def test_build_gate_opens_only_to_build(dag, action: str) -> None:
    for side in SIDES:
        plan = {**BUILD_BOTH, side: {'action': action, 'disk_size_gb': 1}}
        assert _callable(dag, f'build_{side}')(plan) is (action == 'build')


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
    assert _render(dag, 'run_backend_clickhouse', 'data_disks') == {
        'clickhouse': NAMES['ch_build_disk'],
        'clickhouse-compact': NAMES['ch_disk'],
    }
    assert _render(dag, 'run_backend_opensearch', 'data_disks') == {'opensearch': NAMES['os_disk']}


def test_snapshots_record_their_provenance(dag) -> None:
    for side in SIDES:
        assert _render(dag, f'snapshot_{side}', 'provenance') == {
            'pos_image': CAPTURED['pos_image'],
            'server_image': CAPTURED[f'{side}_image'],
            'airflow_run_id': RUN_ID,
        }


def test_trailing_slash_release_uri_finds_the_hand_off() -> None:
    names = release_candidate.candidate(f'{URI}/')
    assert names['release_uri'] == URI
    assert HandOff.uri(names['release_uri']) == f'{URI}/release_candidate.json'


def test_disks_are_sized_from_the_names_task(dag) -> None:
    assert str(_render(dag, 'create_disk_clickhouse_build', 'size_gb')) == '678'
    assert str(_render(dag, 'create_disk_clickhouse', 'size_gb')) == '123'
    assert str(_render(dag, 'create_disk_opensearch', 'size_gb')) == '45'


CREATED_OK = {
    'plan': 'success',
    'create_disk_clickhouse_build': 'success',
    'create_disk_clickhouse': 'success',
    'create_disk_opensearch': 'failed',
}


@pytest.mark.parametrize(
    ('states', 'deleted'),
    [
        # created this try, and a create whose wait failed after the insert
        (CREATED_OK, [NAMES['ch_build_disk'], NAMES['ch_disk'], NAMES['os_disk']]),
        # a refused plan: the creates never started, old XComs may remain
        ({'plan': 'failed', **dict.fromkeys(f'create_disk_{d}' for d in DISK_TASKS)}, []),
        ({'plan': 'upstream_failed'}, []),
        # plan passed, but a create was skipped or never ran and kept an old XCom
        ({'plan': 'success', 'create_disk_clickhouse_build': 'skipped', 'create_disk_clickhouse': 'upstream_failed',
          'create_disk_opensearch': None}, []),
    ],
)
def test_disk_deletes_remove_only_disks_this_try_created(states, deleted) -> None:
    with patch.object(release_candidate, 'delete_disk') as delete:
        for device in DISK_TASKS:
            release_candidate.cleanup_disk(_TI(CREATED, states), device)
        calls = [c.args[2] for c in delete.call_args_list]
    assert calls == deleted


def test_a_create_that_found_the_disk_pushed_nothing_and_deletes_nothing() -> None:
    states = {'plan': 'success', 'create_disk_clickhouse': 'failed'}
    with patch.object(release_candidate, 'delete_disk') as delete:
        release_candidate.cleanup_disk(_TI({}, states), 'clickhouse')
    delete.assert_not_called()


@pytest.mark.parametrize(
    ('side', 'machine', 'disks', 'compact'),
    [
        ('clickhouse', 'n2d-highmem-96', {'clickhouse', 'clickhouse-compact'}, {'clickhouse': 'clickhouse-compact'}),
        ('opensearch', 'n2d-highmem-32', {'opensearch'}, {}),
    ],
)
def test_backend_args(dag, side: str, machine: str, disks: set[str], compact: dict[str, str]) -> None:
    rc = ReleaseConfig()
    up = UnifiedPipelineConfig()
    run = cast(ComputeEngineRunBackendSensor, dag.get_task(f'run_backend_{side}'))
    assert isinstance(run, ComputeEngineRunBackendSensor)
    assert run.servers == {side}
    assert set(run.data_disks) == disks
    assert run.compact == compact
    assert run.read_only is False
    assert run.machine_type == machine
    assert run.work_disk_size_gb == 100
    assert run.max_run_duration == timedelta(hours=4)
    assert run.retries == 0
    assert run.vm_service_account is None
    assert run.boot_disk_size_gb == rc.boot_disk_gb
    assert run.source_image == up.cos_image
    assert run.deferrable is True
    assert run.labels['tool'] == 'pos'
    assert run.labels['step'] == f'backend_{side}'
    assert run.container_args is not None
    assert run.container_args[:4] == ['-s', f'backend_{side}', '-r', '{{ params.release_uri }}']
    args = ' '.join(_render(dag, f'run_backend_{side}', 'container_args'))
    assert args == f'-s backend_{side} -r {URI} -w /mnt/disks/work -p 8'
    assert _render(dag, f'run_backend_{side}', 'instance_name').startswith(f'up-release-candidate-{side}-')


def test_backends_and_hand_off_record_the_same_images(dag) -> None:
    rendered = {
        'pos_image': _render(dag, 'run_backend_clickhouse', 'container_image'),
        'clickhouse_image': _render(dag, 'run_backend_clickhouse', 'clickhouse_image'),
        'opensearch_image': _render(dag, 'run_backend_opensearch', 'opensearch_image'),
    }
    assert _render(dag, 'run_backend_opensearch', 'container_image') == rendered['pos_image']
    assert rendered == {k: CAPTURED[k] for k in rendered}
    body = _upload(NAMES)
    assert {k: body[k] for k in rendered} == rendered
    assert body['pos_version'] == CAPTURED['pos_version']


@pytest.mark.parametrize(('names', 'product'), [(NAMES, 'platform'), (PPP_NAMES, 'ppp')])
def test_labels_carry_the_product_and_side(dag, names: dict[str, Any], product: str) -> None:
    run = 'manual__2026-11-20t10-00-00-00-00'
    for side in SIDES:
        vm = _render(dag, f'run_backend_{side}', 'labels', names)
        assert (vm['tool'], vm['step'], vm['product']) == ('pos', f'backend_{side}', product)
        for d in SIDE_DISKS[side]:
            assert dict(_render(dag, f'create_disk_{d}', 'labels', names)) == {**vm, 'run': run}
        labels = dict(_render(dag, f'snapshot_{side}', 'labels', names))
        assert labels == {**vm, 'run': run, 'release': '2612'}


def test_vm_labels_are_cleaned_after_rendering(dag) -> None:
    vm = _render(dag, 'run_backend_clickhouse', 'labels', {**NAMES, 'product': 'Not Clean'})
    assert vm['product'] == 'not-clean'


def _upload(names: dict[str, Any], plan: dict[str, dict[str, Any]] = BUILD_BOTH) -> dict[str, Any]:
    now = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
    with patch.object(release_candidate, 'Client') as client:
        uri = release_candidate.upload_hand_off(names, plan, 'manual__run', now=now)
    assert uri == HandOff.uri(names['release_uri'])
    client.return_value.bucket.assert_called_once_with('open-targets-pipeline-runs')
    bucket = client.return_value.bucket.return_value
    bucket.blob.assert_called_once_with(f'{names["prefix"]}/{names["run"]}/release_candidate.json')
    upload = cast(MagicMock, bucket.blob.return_value.upload_from_string)
    assert upload.call_args.kwargs['if_generation_match'] == 0
    return HandOff.model_validate(json.loads(upload.call_args.args[0])).model_dump(mode='json')


def test_hand_off_uses_plan_disk_sizes() -> None:
    names = {**NAMES, 'clickhouse_disk_gb': 200}
    plan = {
        'clickhouse': {'action': 'reuse', 'disk_size_gb': 170},
        'opensearch': {'action': 'build', 'disk_size_gb': 80},
    }
    assert _upload(names, plan)['disk_size_gb'] == {'clickhouse': 170, 'opensearch': 80}


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
            assert release_candidate.upload_hand_off(NAMES, BUILD_BOTH, 'manual__run') == HandOff.uri(URI)
        else:
            with pytest.raises(PreconditionFailed):
                release_candidate.upload_hand_off(NAMES, BUILD_BOTH, 'manual__run')
