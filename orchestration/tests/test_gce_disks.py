"""Tests for the disk and snapshot operators."""

from unittest.mock import MagicMock, patch

import pytest
from airflow.sdk.exceptions import AirflowException
from google.api_core.exceptions import NotFound

from orchestration.operators import gce_disks
from orchestration.operators.gce_disks import (
    CreateDiskOperator,
    CreateSnapshotOperator,
    DeleteDiskOperator,
    DeleteSnapshotOperator,
    SnapshotGuardOperator,
)
from orchestration.utils.labels import Labels

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
