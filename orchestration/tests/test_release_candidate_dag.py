"""Tests for the release-candidate DAG and the unified pipeline's switch to it."""

import copy
import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from airflow.models import DagBag
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.task.trigger_rule import TriggerRule

from orchestration.dags import release_candidate
from orchestration.dags.config.release import HandOff, ReleaseConfig
from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig
from orchestration.operators.gce import DeleteInstanceOperator
from orchestration.operators.gce_backend import ComputeEngineRunBackendSensor
from orchestration.operators.gce_disks import DeleteDiskOperator
from orchestration.utils.release import candidate_names

URI = 'gs://open-targets-pipeline-runs/do/platform-2612-3'
PPP_URI = 'gs://open-targets-pipeline-runs/sz/ppp-2612-1'
# images and sizes as captured by `names`; distinct from the config so a test sees where they come from
CAPTURED = {
    'pos_image': 'europe-west1-docker.pkg.dev/p/r/pos:captured',
    'pos_version': 'captured',
    'clickhouse_image': 'clickhouse/clickhouse-server:captured',
    'opensearch_image': 'opensearchproject/opensearch:captured',
    'clickhouse_disk_gb': 123,
    'opensearch_disk_gb': 45,
}
NAMES = {**asdict(candidate_names(URI)), **CAPTURED}
PPP_NAMES = {**asdict(candidate_names(PPP_URI)), **CAPTURED}
RUN_ID = 'manual__2026-11-20T10:00:00+00:00'
DAGS = Path('src/orchestration/dags')


class _TI:
    """XComs keyed by task id (return value) or by `<task id>:<key>`."""

    def __init__(self, xcoms: dict[str, Any]) -> None:
        self.xcoms = xcoms

    def xcom_pull(self, task_ids: str, key: str = 'return_value', **_: Any) -> Any:
        return self.xcoms.get(task_ids if key == 'return_value' else f'{task_ids}:{key}')


CREATED = {'create_disk_clickhouse:disk': NAMES['ch_disk'], 'create_disk_opensearch:disk': NAMES['os_disk']}


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
    assert up['create_disk_clickhouse'] == {'guard'}
    assert up['create_disk_opensearch'] == {'guard'}
    assert up['run_backend'] == {'create_disk_clickhouse', 'create_disk_opensearch'}
    assert up['delete_vm_backend'] == {'run_backend'}
    for s in ('snapshot_clickhouse', 'snapshot_opensearch'):
        assert 'delete_vm_backend' in up[s]
        # a failed backend must not reach the snapshots through the ALL_DONE VM delete
        assert 'run_backend' in up[s]
    assert {'snapshot_clickhouse', 'snapshot_opensearch'} <= up['write_hand_off']
    for d in ('delete_disk_clickhouse', 'delete_disk_opensearch'):
        assert {'delete_vm_backend', 'snapshot_clickhouse', 'snapshot_opensearch'} <= up[d]
        assert dag.get_task(d).trigger_rule == TriggerRule.ALL_DONE
        assert isinstance(dag.get_task(d), DeleteDiskOperator)
    assert dag.get_task('delete_vm_backend').trigger_rule == TriggerRule.ALL_DONE
    assert isinstance(dag.get_task('delete_vm_backend'), DeleteInstanceOperator)
    assert dag.get_task('write_hand_off').trigger_rule == TriggerRule.ALL_SUCCESS
    dag.topological_sort()


def test_names_captures_the_images_and_sizes() -> None:
    rc = ReleaseConfig()
    up = UnifiedPipelineConfig()
    assert release_candidate.candidate(URI) == {
        **asdict(candidate_names(URI)),
        'pos_image': up.pos_image,
        'pos_version': up.pos_version,
        'clickhouse_image': rc.clickhouse_image,
        'opensearch_image': rc.opensearch_image,
        'clickhouse_disk_gb': rc.clickhouse_disk_gb,
        'opensearch_disk_gb': rc.opensearch_disk_gb,
    }


def test_names_are_templated_from_the_names_task(dag) -> None:
    assert _render(dag, 'create_disk_clickhouse', 'disk_name') == NAMES['ch_disk']
    assert _render(dag, 'create_disk_opensearch', 'disk_name') == NAMES['os_disk']
    assert _render(dag, 'snapshot_clickhouse', 'snapshot_name') == NAMES['ch_snapshot']
    assert _render(dag, 'snapshot_clickhouse', 'disk_name') == NAMES['ch_disk']
    assert _render(dag, 'snapshot_opensearch', 'snapshot_name') == NAMES['os_snapshot']
    assert _render(dag, 'snapshot_opensearch', 'disk_name') == NAMES['os_disk']
    assert _render(dag, 'guard', 'snapshot_names') == [NAMES['ch_snapshot'], NAMES['os_snapshot']]
    assert _render(dag, 'guard', 'disk_names') == [NAMES['ch_disk'], NAMES['os_disk']]
    assert _render(dag, 'guard', 'blob_uri') == HandOff.uri(URI)
    assert _render(dag, 'run_backend', 'data_disks') == {'clickhouse': NAMES['ch_disk'], 'opensearch': NAMES['os_disk']}


def test_disks_are_sized_from_the_names_task(dag) -> None:
    assert str(_render(dag, 'create_disk_clickhouse', 'size_gb')) == '123'
    assert str(_render(dag, 'create_disk_opensearch', 'size_gb')) == '45'


def test_disk_deletes_remove_only_disks_this_run_created(dag) -> None:
    disks = {'delete_disk_clickhouse': NAMES['ch_disk'], 'delete_disk_opensearch': NAMES['os_disk']}
    for task_id, disk in disks.items():
        assert _render(dag, task_id, 'disk_name', xcoms=CREATED) == disk
        # a refused guard, or a create that found the disk already there, records no disk
        assert _render(dag, task_id, 'disk_name') == ''
        with patch('orchestration.operators.gce_disks.compute_v1.DisksClient') as client:
            op = cast(DeleteDiskOperator, copy.deepcopy(dag.get_task(task_id)))
            op.disk_name = _render(dag, task_id, 'disk_name')
            op.execute({})
        client.return_value.delete.assert_not_called()


def test_run_backend(dag) -> None:
    rc = ReleaseConfig()
    up = UnifiedPipelineConfig()
    run = cast(ComputeEngineRunBackendSensor, dag.get_task('run_backend'))
    assert isinstance(run, ComputeEngineRunBackendSensor)
    assert run.servers is True
    assert run.read_only is False
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
    for task_id in ('create_disk_clickhouse', 'create_disk_opensearch'):
        assert dict(_render(dag, task_id, 'labels', names)) == dict(vm)
    for task_id in ('snapshot_clickhouse', 'snapshot_opensearch'):
        labels = dict(_render(dag, task_id, 'labels', names))
        assert labels['release'] == '2612'
        assert labels['run'] == RUN_ID
        assert {k: v for k, v in labels.items() if k not in ('release', 'run')} == dict(vm)
        assert dag.get_task(task_id).trigger_rule == TriggerRule.ALL_SUCCESS


def test_fail_cleanup_runs_only_on_failure_after_the_snapshots(dag) -> None:
    task = dag.get_task('fail_cleanup_snapshots')
    assert task.trigger_rule == TriggerRule.ONE_FAILED
    assert {'snapshot_clickhouse', 'snapshot_opensearch', 'write_hand_off'} <= task.upstream_task_ids
    assert not task.downstream_task_ids


def test_fail_cleanup_deletes_only_snapshots_this_run_created() -> None:
    with patch.object(release_candidate, '_delete_snapshot') as delete:
        release_candidate.cleanup_snapshots(
            _TI({'snapshot_clickhouse': NAMES['ch_snapshot'], 'snapshot_opensearch': NAMES['os_snapshot']})
        )
    assert [c.args[1] for c in delete.call_args_list] == [NAMES['ch_snapshot'], NAMES['os_snapshot']]

    # a refused guard leaves the snapshot tasks upstream_failed, with no XCom: an existing candidate stays
    with patch.object(release_candidate, '_delete_snapshot') as delete:
        release_candidate.cleanup_snapshots(_TI({}))
    delete.assert_not_called()


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
