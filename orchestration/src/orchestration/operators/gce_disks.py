"""Operators for the persistent disks and snapshots that hold release candidates."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from airflow.sdk import BaseOperator
from airflow.sdk.exceptions import AirflowException
from google.api_core.exceptions import NotFound
from google.cloud import compute_v1

from orchestration.operators.gce import wait_for_extended_operation
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
    """Create a persistent disk, optionally from a snapshot (a name or a path). Fails if it exists."""

    template_fields: Sequence[str] = ('disk_name', 'size_gb', 'source_snapshot', 'disk_type', 'labels')

    def __init__(
        self,
        *args: Any,
        disk_name: str,
        size_gb: int | str,
        labels: Labels | dict[str, str],
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

    def execute(self, context: Any) -> str:
        """Return the disk name; XCom key `disk` records it as soon as the insert is accepted."""
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
            labels=_clean(self.labels),
        )
        if self.source_snapshot:
            # a bare name is in this project; a path is used as is
            disk.source_snapshot = (
                self.source_snapshot
                if '/' in self.source_snapshot
                else f'projects/{self.project_id}/global/snapshots/{self.source_snapshot}'
            )
        op = client.insert(
            request=compute_v1.InsertDiskRequest(
                project=self.project_id, zone=self.zone, disk_resource=disk
            )
        )
        if ti := context.get('ti'):
            ti.xcom_push(key='disk', value=self.disk_name)
        wait_for_extended_operation(op, f'create disk {self.disk_name}', DISK_TIMEOUT, self.log)
        return self.disk_name


def _clean(labels: Labels | dict[str, str]) -> dict[str, str]:
    return {k: Labels.clean_label(str(v)) for k, v in labels.items()}


def delete_snapshot(project_id: str, name: str, log: Any) -> None:
    """Delete a snapshot; a missing one is success."""
    client = compute_v1.SnapshotsClient()
    try:
        op = client.delete(project=project_id, snapshot=name)
        wait_for_extended_operation(op, f'delete snapshot {name}', DISK_TIMEOUT, log)
    except NotFound:
        log.info('snapshot %s not found, nothing to delete', name)


def delete_disk(project_id: str, zone: str, name: str, log: Any) -> None:
    """Delete a disk; a missing one is success."""
    client = compute_v1.DisksClient()
    try:
        op = client.delete(project=project_id, zone=zone, disk=name)
        wait_for_extended_operation(op, f'delete disk {name}', DISK_TIMEOUT, log)
    except NotFound:
        log.info('disk %s not found, nothing to delete', name)


class CreateSnapshotOperator(_ComputeOperator):
    """Snapshot a disk; delete the partial snapshot if creation fails."""

    template_fields: Sequence[str] = ('disk_name', 'snapshot_name', 'labels', 'provenance')

    def __init__(
        self,
        *args: Any,
        disk_name: str,
        snapshot_name: str,
        labels: Labels | dict[str, str],
        provenance: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.disk_name = disk_name
        self.snapshot_name = snapshot_name
        self.labels = labels
        self.provenance = provenance

    def _is_ours(self, client: Any) -> bool:
        """False if the snapshot records another run, which must keep it."""
        if self.provenance is None:
            return True
        try:
            existing = client.get(project=self.project_id, snapshot=self.snapshot_name)
        except NotFound:
            return True
        try:
            recorded = json.loads(existing.description).get('airflow_run_id')
        except (ValueError, AttributeError):
            recorded = None
        if recorded != self.provenance.get('airflow_run_id'):
            self.log.warning('snapshot %s belongs to run %s, leaving it', self.snapshot_name, recorded)
            return False
        return True

    def execute(self, context: Any) -> str:
        """Return the snapshot name, so XCom records the snapshots this run created."""
        client = compute_v1.SnapshotsClient()
        snapshot = compute_v1.Snapshot(
            name=self.snapshot_name,
            source_disk=f'projects/{self.project_id}/zones/{self.zone}/disks/{self.disk_name}',
            storage_locations=[SNAPSHOT_LOCATION],
            labels=_clean(self.labels),
        )
        if self.provenance is not None:
            snapshot.description = json.dumps(self.provenance, sort_keys=True)
        op = client.insert(
            request=compute_v1.InsertSnapshotRequest(
                project=self.project_id, snapshot_resource=snapshot
            )
        )
        try:
            wait_for_extended_operation(
                op, f'create snapshot {self.snapshot_name}', SNAPSHOT_TIMEOUT, self.log
            )
        except Exception:
            self.log.exception('snapshot %s failed, deleting it', self.snapshot_name)
            try:
                if self._is_ours(client):
                    delete_snapshot(self.project_id, self.snapshot_name, self.log)
            except Exception:
                self.log.exception('could not delete snapshot %s', self.snapshot_name)
            raise
        return self.snapshot_name
