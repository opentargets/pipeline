"""Operators for the persistent disks and snapshots that hold release candidates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from airflow.sdk import BaseOperator
from airflow.sdk.exceptions import AirflowException
from google.api_core.exceptions import NotFound
from google.cloud import compute_v1
from google.cloud.storage import Client

from orchestration.operators.gce import wait_for_extended_operation
from orchestration.utils import GCSPath
from orchestration.utils.common import GCP_PROJECT_PLATFORM, GCP_ZONE
from orchestration.utils.labels import Labels

SNAPSHOT_LOCATION = 'europe-west1'
DISK_TIMEOUT = 1800
SNAPSHOT_TIMEOUT = 7200


class _ComputeOperator(BaseOperator):
    """Shared project and zone settings."""

    def __init__(
        self,
        *args: Any,
        project_id: str = GCP_PROJECT_PLATFORM,
        zone: str = GCP_ZONE,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.project_id = project_id
        self.zone = zone


class CreateDiskOperator(_ComputeOperator):
    """Create a persistent disk, optionally from a snapshot. Fails if it exists."""

    template_fields: Sequence[str] = ('disk_name', 'size_gb', 'source_snapshot', 'disk_type')

    def __init__(
        self,
        *args: Any,
        disk_name: str,
        size_gb: int | str,
        labels: Labels,
        source_snapshot: str | None = None,
        disk_type: str = 'pd-ssd',
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.disk_name = disk_name
        self.size_gb = size_gb
        self.labels = labels
        self.source_snapshot = source_snapshot
        self.disk_type = disk_type

    def execute(self, context: Any) -> None:
        client = compute_v1.DisksClient()
        try:
            client.get(project=self.project_id, zone=self.zone, disk=self.disk_name)
        except NotFound:
            pass
        else:
            raise AirflowException(f'disk {self.disk_name} already exists')

        disk = compute_v1.Disk(
            name=self.disk_name,
            size_gb=int(self.size_gb),
            type_=f'zones/{self.zone}/diskTypes/{self.disk_type}',
            labels=dict(self.labels),
        )
        if self.source_snapshot:
            disk.source_snapshot = f'projects/{self.project_id}/global/snapshots/{self.source_snapshot}'
        op = client.insert(
            request=compute_v1.InsertDiskRequest(
                project=self.project_id, zone=self.zone, disk_resource=disk
            )
        )
        wait_for_extended_operation(op, f'create disk {self.disk_name}', DISK_TIMEOUT, self.log)


def _delete_snapshot(project_id: str, name: str, log: Any) -> None:
    client = compute_v1.SnapshotsClient()
    try:
        op = client.delete(project=project_id, snapshot=name)
        wait_for_extended_operation(op, f'delete snapshot {name}', DISK_TIMEOUT, log)
    except NotFound:
        log.info('snapshot %s not found, nothing to delete', name)


class DeleteDiskOperator(_ComputeOperator):
    """Delete a persistent disk. A missing disk is success."""

    template_fields: Sequence[str] = ('disk_name',)

    def __init__(self, *args: Any, disk_name: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.disk_name = disk_name

    def execute(self, context: Any) -> None:
        client = compute_v1.DisksClient()
        try:
            op = client.delete(project=self.project_id, zone=self.zone, disk=self.disk_name)
            wait_for_extended_operation(op, f'delete disk {self.disk_name}', DISK_TIMEOUT, self.log)
        except NotFound:
            self.log.info('disk %s not found, nothing to delete', self.disk_name)


class DeleteSnapshotOperator(_ComputeOperator):
    """Delete a snapshot. A missing snapshot is success."""

    template_fields: Sequence[str] = ('snapshot_name',)

    def __init__(self, *args: Any, snapshot_name: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.snapshot_name = snapshot_name

    def execute(self, context: Any) -> None:
        _delete_snapshot(self.project_id, self.snapshot_name, self.log)


class SnapshotGuardOperator(_ComputeOperator):
    """Fail if any snapshot or the hand-off blob already exists."""

    template_fields: Sequence[str] = ('snapshot_names', 'blob_uri')

    def __init__(
        self, *args: Any, snapshot_names: list[str], blob_uri: str, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.snapshot_names = snapshot_names
        self.blob_uri = blob_uri

    def execute(self, context: Any) -> None:
        client = compute_v1.SnapshotsClient()
        found: list[str] = []
        for name in self.snapshot_names:
            try:
                client.get(project=self.project_id, snapshot=name)
            except NotFound:
                continue
            found.append(f'snapshot {name}')

        bucket, path = GCSPath(self.blob_uri).split()
        if Client(project=self.project_id).bucket(bucket).blob(path).exists():
            found.append(f'blob {self.blob_uri}')

        if found:
            raise AirflowException(f'already exists: {", ".join(found)}')


class CreateSnapshotOperator(_ComputeOperator):
    """Snapshot a disk; delete the partial snapshot if creation fails."""

    template_fields: Sequence[str] = ('disk_name', 'snapshot_name')

    def __init__(
        self, *args: Any, disk_name: str, snapshot_name: str, labels: Labels, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.disk_name = disk_name
        self.snapshot_name = snapshot_name
        self.labels = labels

    def execute(self, context: Any) -> None:
        client = compute_v1.SnapshotsClient()
        snapshot = compute_v1.Snapshot(
            name=self.snapshot_name,
            source_disk=f'projects/{self.project_id}/zones/{self.zone}/disks/{self.disk_name}',
            storage_locations=[SNAPSHOT_LOCATION],
            labels=dict(self.labels),
        )
        try:
            op = client.insert(
                request=compute_v1.InsertSnapshotRequest(
                    project=self.project_id, snapshot_resource=snapshot
                )
            )
            wait_for_extended_operation(
                op, f'create snapshot {self.snapshot_name}', SNAPSHOT_TIMEOUT, self.log
            )
        except Exception:
            self.log.exception('snapshot %s failed, deleting it', self.snapshot_name)
            _delete_snapshot(self.project_id, self.snapshot_name, self.log)
            raise
