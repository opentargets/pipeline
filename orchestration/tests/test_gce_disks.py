"""Tests for the disk and snapshot operators."""

from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from airflow.sdk.exceptions import AirflowException
from google.api_core.exceptions import Conflict, NotFound

from orchestration.operators import gce_disks
from orchestration.operators.gce_disks import (
    CreateDiskOperator,
    CreateSnapshotOperator,
    DeleteDiskOperator,
    DeleteSnapshotOperator,
    SnapshotGuardOperator,
)
from orchestration.utils.labels import Labels, TemplatedLabels

DISK = 'pos-do-platform-2612-3-ch'
SNAP = 'do-platform-2612-3-ch'


@pytest.fixture
def disks():
    with patch.object(gce_disks.compute_v1, 'DisksClient') as c:
        yield c.return_value


@pytest.fixture
def snapshots():
    with patch.object(gce_disks.compute_v1, 'SnapshotsClient') as c:
        yield c.return_value


@pytest.fixture(autouse=True)
def waits():
    with patch.object(gce_disks, 'wait_for_extended_operation') as w:
        yield w


def test_create_disk_passes_size_snapshot_type_and_labels(disks) -> None:
    disks.get.side_effect = NotFound('x')
    labels = Labels({'step': 'x'})
    CreateDiskOperator(
        task_id='t', disk_name=DISK, size_gb='500', labels=labels, source_snapshot=SNAP
    ).execute({})
    req = disks.insert.call_args.kwargs['request']
    d = req.disk_resource
    assert d.name == DISK
    assert d.size_gb == 500
    assert d.source_snapshot.endswith(f'global/snapshots/{SNAP}')
    assert d.type_.endswith('diskTypes/pd-ssd')
    assert dict(d.labels) == dict(labels)
    assert req.zone == gce_disks.GCP_ZONE


def test_create_disk_fails_if_it_exists(disks) -> None:
    with pytest.raises(AirflowException, match=DISK):
        CreateDiskOperator(task_id='t', disk_name=DISK, size_gb=1, labels=Labels()).execute({})
    disks.insert.assert_not_called()


def test_create_disk_without_snapshot_has_no_source(disks) -> None:
    disks.get.side_effect = NotFound('x')
    CreateDiskOperator(task_id='t', disk_name=DISK, size_gb=1, labels=Labels()).execute({})
    disks.insert.assert_called_once()
    assert not disks.insert.call_args.kwargs['request'].disk_resource.source_snapshot


def test_delete_disk_treats_not_found_as_success(disks) -> None:
    disks.delete.side_effect = NotFound('x')
    DeleteDiskOperator(task_id='t', disk_name=DISK).execute({})


def test_delete_snapshot_treats_not_found_as_success(snapshots) -> None:
    snapshots.delete.side_effect = NotFound('x')
    DeleteSnapshotOperator(task_id='t', snapshot_name=SNAP).execute({})


def _guard(snapshots, existing=(), blob_exists=False):
    def get(**kw):
        if kw['snapshot'] in existing:
            return MagicMock()
        raise NotFound('x')

    snapshots.get.side_effect = get
    blob = MagicMock()
    blob.exists.return_value = blob_exists
    client = MagicMock()
    client.bucket.return_value.blob.return_value = blob
    return patch.object(gce_disks, 'Client', return_value=client)


def test_guard_passes_when_nothing_exists(snapshots) -> None:
    with _guard(snapshots):
        SnapshotGuardOperator(
            task_id='t', snapshot_names=[SNAP], blob_uri='gs://b/handoff.json'
        ).execute({})


def test_guard_names_existing_snapshot(snapshots) -> None:
    with _guard(snapshots, existing={SNAP}), pytest.raises(AirflowException, match=SNAP):
        SnapshotGuardOperator(
            task_id='t', snapshot_names=[SNAP, 'other'], blob_uri='gs://b/h.json'
        ).execute({})


def test_guard_names_existing_blob(snapshots) -> None:
    with _guard(snapshots, blob_exists=True), pytest.raises(AirflowException, match=r'gs://b/h\.json'):
        SnapshotGuardOperator(
            task_id='t', snapshot_names=[SNAP], blob_uri='gs://b/h.json'
        ).execute({})


def test_create_snapshot_sets_source_location_and_labels(snapshots) -> None:
    labels = Labels({'step': 'x'})
    CreateSnapshotOperator(
        task_id='t', disk_name=DISK, snapshot_name=SNAP, labels=labels
    ).execute({})
    req = snapshots.insert.call_args.kwargs['request']
    r = req.snapshot_resource
    assert r.name == SNAP
    assert r.source_disk == (
        f'projects/{gce_disks.GCP_PROJECT_PLATFORM}/zones/{gce_disks.GCP_ZONE}/disks/{DISK}'
    )
    assert list(r.storage_locations) == ['europe-west1']
    assert dict(r.labels) == dict(labels)
    assert req.project == gce_disks.GCP_PROJECT_PLATFORM


def test_failed_snapshot_is_deleted_and_error_reraised(snapshots, waits) -> None:
    waits.side_effect = [RuntimeError('boom'), None]
    snapshots.delete.side_effect = NotFound('x')
    with pytest.raises(RuntimeError, match='boom'):
        CreateSnapshotOperator(
            task_id='t', disk_name=DISK, snapshot_name=SNAP, labels=Labels()
        ).execute({})
    assert snapshots.delete.call_args.kwargs['snapshot'] == SNAP


def test_template_fields_cover_names_and_sizes() -> None:
    assert {'disk_name', 'size_gb', 'source_snapshot'} <= set(CreateDiskOperator.template_fields)
    assert {'snapshot_names', 'blob_uri'} <= set(SnapshotGuardOperator.template_fields)
    assert {'disk_name', 'snapshot_name'} <= set(CreateSnapshotOperator.template_fields)


def test_rejected_insert_does_not_delete_existing_snapshot(snapshots) -> None:
    snapshots.insert.side_effect = Conflict('exists')
    with pytest.raises(Conflict):
        CreateSnapshotOperator(
            task_id='t', disk_name=DISK, snapshot_name=SNAP, labels=Labels()
        ).execute({})
    snapshots.delete.assert_not_called()


def test_cleanup_failure_does_not_mask_the_original_error(snapshots, waits) -> None:
    waits.side_effect = [RuntimeError('boom'), None]
    snapshots.delete.side_effect = PermissionError('denied')
    with pytest.raises(RuntimeError, match='boom'):
        CreateSnapshotOperator(
            task_id='t', disk_name=DISK, snapshot_name=SNAP, labels=Labels()
        ).execute({})


def test_snapshot_labels_are_templated_and_cleaned(snapshots) -> None:
    op = CreateSnapshotOperator(
        task_id='t',
        disk_name=DISK,
        snapshot_name=SNAP,
        labels={**Labels({'step': 'x'}), 'release': '{{ yymm }}', 'run': '{{ run_id }}'},
    )
    assert 'labels' in CreateSnapshotOperator.template_fields
    context = cast(Any, {'yymm': '2612', 'run_id': 'manual__2026-11-20T10:00:00+00:00'})
    op.labels = op.render_template(op.labels, context)
    assert op.execute({}) == SNAP
    labels = dict(snapshots.insert.call_args.kwargs['request'].snapshot_resource.labels)
    assert labels['release'] == '2612'
    assert labels['run'] == 'manual__2026-11-20t10-00-00-00-00'
    assert labels['step'] == 'x'


def _disks_exist(disks, existing) -> None:
    def get(**kw):
        if kw['disk'] in existing:
            return MagicMock()
        raise NotFound('x')

    disks.get.side_effect = get


def test_guard_names_existing_disk(snapshots, disks) -> None:
    _disks_exist(disks, {DISK})
    with _guard(snapshots), pytest.raises(AirflowException, match=f'disk {DISK}'):
        SnapshotGuardOperator(
            task_id='t', snapshot_names=[SNAP], disk_names=[DISK, 'other'], blob_uri='gs://b/h.json'
        ).execute({})
    assert {c.kwargs['zone'] for c in disks.get.call_args_list} == {gce_disks.GCP_ZONE}


def test_guard_passes_when_no_disk_exists(snapshots, disks) -> None:
    _disks_exist(disks, set())
    with _guard(snapshots):
        SnapshotGuardOperator(
            task_id='t', snapshot_names=[SNAP], disk_names=[DISK], blob_uri='gs://b/h.json'
        ).execute({})
    assert 'disk_names' in SnapshotGuardOperator.template_fields


def test_create_disk_records_the_disk_once_inserted(disks, waits) -> None:
    disks.get.side_effect = NotFound('x')
    ti = MagicMock()
    op = CreateDiskOperator(task_id='t', disk_name=DISK, size_gb=1, labels=Labels())
    assert op.execute(cast(Any, {'ti': ti})) == DISK
    ti.xcom_push.assert_called_once_with(key='disk', value=DISK)

    # a failed wait still records the disk, so the delete removes it
    ti.reset_mock()
    waits.side_effect = RuntimeError('boom')
    with pytest.raises(RuntimeError):
        op.execute(cast(Any, {'ti': ti}))
    ti.xcom_push.assert_called_once_with(key='disk', value=DISK)


def test_create_disk_that_exists_records_nothing(disks) -> None:
    ti = MagicMock()
    with pytest.raises(AirflowException):
        CreateDiskOperator(task_id='t', disk_name=DISK, size_gb=1, labels=Labels()).execute(cast(Any, {'ti': ti}))
    ti.xcom_push.assert_not_called()


def test_delete_disk_without_a_name_deletes_nothing(disks) -> None:
    DeleteDiskOperator(task_id='t', disk_name='').execute({})
    disks.delete.assert_not_called()


def test_disk_labels_are_cleaned(disks) -> None:
    disks.get.side_effect = NotFound('x')
    CreateDiskOperator(
        task_id='t', disk_name=DISK, size_gb=1, labels=cast(Any, {'run': 'manual__2026-11-20T10:00'})
    ).execute({})
    labels = dict(disks.insert.call_args.kwargs['request'].disk_resource.labels)
    assert labels == {'run': 'manual__2026-11-20t10-00'}


def test_templated_labels_render_from_the_context() -> None:
    labels = TemplatedLabels({'tool': 'pos'}, templated={'product': '{{ product }}'})
    assert labels['product'] == '{{ product }}'
    op = DeleteDiskOperator(task_id='t', disk_name='')
    rendered = op.render_template(labels, cast(Any, {'product': 'ppp'}))
    assert rendered['product'] == 'ppp'
    assert rendered['tool'] == 'pos'
