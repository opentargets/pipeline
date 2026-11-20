"""Tests for the release-candidate DAG and the unified pipeline's switch to it."""

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

from orchestration.dags.config.release import HandOff, ReleaseConfig
from orchestration.dags.config.unified_pipeline import UnifiedPipelineConfig
from orchestration.operators.gce import DeleteInstanceOperator
from orchestration.operators.gce_backend import ComputeEngineRunBackendSensor
from orchestration.operators.gce_disks import CreateDiskOperator, CreateSnapshotOperator, DeleteDiskOperator
from orchestration.utils.release import candidate_names

URI = 'gs://open-targets-pipeline-runs/do/platform-2612-3'
NAMES = asdict(candidate_names(URI))
DAGS = Path('src/orchestration/dags')


class _TI:
    def __init__(self, xcoms: dict[str, Any]) -> None:
        self.xcoms = xcoms

    def xcom_pull(self, task_ids: str, **_: Any) -> Any:
        return self.xcoms.get(task_ids)


def _context(xcoms: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        'params': {'release_uri': URI},
        'run_id': 'manual__2026-11-20T10:00:00+00:00',
        'ti': _TI({'names': NAMES, **(xcoms or {})}),
    }


@pytest.fixture(scope='module')
def dag():
    bag = DagBag(dag_folder=str(DAGS / 'release_candidate.py'))
    assert not bag.import_errors
    return bag.dags['release_candidate']


def _render(dag, task_id: str, field: str) -> Any:
    task = dag.get_task(task_id)
    return task.render_template(getattr(task, field), _context(), dag.get_template_env())


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


def test_names_are_templated_from_the_names_task(dag) -> None:
    assert _render(dag, 'create_disk_clickhouse', 'disk_name') == NAMES['ch_disk']
    assert _render(dag, 'create_disk_opensearch', 'disk_name') == NAMES['os_disk']
    assert _render(dag, 'delete_disk_clickhouse', 'disk_name') == NAMES['ch_disk']
    assert _render(dag, 'delete_disk_opensearch', 'disk_name') == NAMES['os_disk']
    assert _render(dag, 'snapshot_clickhouse', 'snapshot_name') == NAMES['ch_snapshot']
    assert _render(dag, 'snapshot_clickhouse', 'disk_name') == NAMES['ch_disk']
    assert _render(dag, 'snapshot_opensearch', 'snapshot_name') == NAMES['os_snapshot']
    assert _render(dag, 'snapshot_opensearch', 'disk_name') == NAMES['os_disk']
    assert _render(dag, 'guard', 'snapshot_names') == [NAMES['ch_snapshot'], NAMES['os_snapshot']]
    assert _render(dag, 'guard', 'blob_uri') == HandOff.uri(URI)
    assert _render(dag, 'run_backend', 'data_disks') == {'clickhouse': NAMES['ch_disk'], 'opensearch': NAMES['os_disk']}


def test_disk_sizes_come_from_the_release_config(dag) -> None:
    rc = ReleaseConfig()
    assert int(cast(CreateDiskOperator, dag.get_task('create_disk_clickhouse')).size_gb) == rc.clickhouse_disk_gb
    assert int(cast(CreateDiskOperator, dag.get_task('create_disk_opensearch')).size_gb) == rc.opensearch_disk_gb


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
    assert run.container_image == up.pos_image
    assert run.clickhouse_image == rc.clickhouse_image
    assert run.opensearch_image == rc.opensearch_image
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


def test_snapshots_carry_the_vm_labels_and_the_release(dag) -> None:
    vm = dict(cast(ComputeEngineRunBackendSensor, dag.get_task('run_backend')).labels)
    for task_id in ('snapshot_clickhouse', 'snapshot_opensearch'):
        task = cast(CreateSnapshotOperator, dag.get_task(task_id))
        labels = _render(dag, task_id, 'labels')
        assert labels['release'] == '2612'
        assert labels['run'] == 'manual__2026-11-20T10:00:00+00:00'
        assert {k: v for k, v in labels.items() if k not in ('release', 'run')} == vm
        assert task.trigger_rule == TriggerRule.ALL_SUCCESS


def test_fail_cleanup_runs_only_on_failure_after_the_snapshots(dag) -> None:
    task = dag.get_task('fail_cleanup_snapshots')
    assert task.trigger_rule == TriggerRule.ONE_FAILED
    assert {'snapshot_clickhouse', 'snapshot_opensearch', 'write_hand_off'} <= task.upstream_task_ids
    assert not task.downstream_task_ids


def test_fail_cleanup_deletes_only_snapshots_this_run_created() -> None:
    from orchestration.dags import release_candidate

    with patch.object(release_candidate, '_delete_snapshot') as delete:
        release_candidate.cleanup_snapshots(
            _TI({'snapshot_clickhouse': NAMES['ch_snapshot'], 'snapshot_opensearch': NAMES['os_snapshot']})
        )
    assert [c.args[1] for c in delete.call_args_list] == [NAMES['ch_snapshot'], NAMES['os_snapshot']]

    # a refused guard leaves the snapshot tasks upstream_failed, with no XCom: an existing candidate stays
    with patch.object(release_candidate, '_delete_snapshot') as delete:
        release_candidate.cleanup_snapshots(_TI({}))
    delete.assert_not_called()


def test_hand_off_is_built_and_uploaded_without_overwrite() -> None:
    from orchestration.dags import release_candidate

    rc = ReleaseConfig()
    up = UnifiedPipelineConfig()
    now = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
    with patch.object(release_candidate, 'Client') as client:
        uri = release_candidate.upload_hand_off(NAMES, 'manual__run', now=now)
    assert uri == HandOff.uri(URI)
    client.return_value.bucket.assert_called_once_with('open-targets-pipeline-runs')
    bucket = client.return_value.bucket.return_value
    bucket.blob.assert_called_once_with('do/platform-2612-3/release_candidate.json')
    upload = cast(MagicMock, bucket.blob.return_value.upload_from_string)
    assert upload.call_args.kwargs['if_generation_match'] == 0
    body = json.loads(upload.call_args.args[0])
    assert HandOff.model_validate(body).model_dump(mode='json') == {
        'release_uri': URI,
        'run': 'do/platform-2612-3',
        'product': 'platform',
        'release': '26.12',
        'snapshots': {
            'clickhouse': 'projects/open-targets-eu-dev/global/snapshots/do-platform-2612-3-ch',
            'opensearch': 'projects/open-targets-eu-dev/global/snapshots/do-platform-2612-3-os',
        },
        'disk_size_gb': {'clickhouse': rc.clickhouse_disk_gb, 'opensearch': rc.opensearch_disk_gb},
        'pos_image': up.pos_image,
        'pos_version': up.pos_version,
        'clickhouse_image': rc.clickhouse_image,
        'opensearch_image': rc.opensearch_image,
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
