"""Release-candidate configuration and hand-off model."""

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

_CONFIG_PATH = Path(__file__).with_name('release.yaml')


class ReleaseConfig(BaseModel):
    """Machines, disks and pins for building and publishing a release, from ``release.yaml``."""

    model_config = ConfigDict(extra='forbid')

    backend_machine_type: str
    backend_max_run_hours: int
    clickhouse_build_disk_gb: int
    """The disk ClickHouse runs on during the build."""
    clickhouse_disk_gb: int
    """The disk ClickHouse is copied onto, snapshotted and restored for serving."""
    opensearch_disk_gb: int
    work_disk_gb: int
    boot_disk_gb: int
    tarballs_machine_type: str
    small_step_machine_type: str
    small_step_work_disk_gb: int
    clickhouse_image: str
    opensearch_image: str
    pos_release_service_account: str
    aws_secret: str
    aws_credentials_path: str
    scratch_min_pos_version: str

    def __init__(self, **data: Any) -> None:
        """Load ``release.yaml``; keyword arguments override its values."""
        super().__init__(**{**yaml.safe_load(_CONFIG_PATH.read_text()), **data})


class HandOff(BaseModel):
    """The release candidate a build run hands to the publish run (``release_candidate.json``)."""

    model_config = ConfigDict(extra='forbid')

    release_uri: str
    run: str
    """The ``<prefix>/<run>`` path, e.g. ``do/platform-2612-3``."""
    product: Literal['platform', 'ppp']
    release: str
    snapshots: dict[Literal['clickhouse', 'opensearch'], str]
    disk_size_gb: dict[Literal['clickhouse', 'opensearch'], int]
    pos_image: str
    pos_version: str
    clickhouse_image: str
    opensearch_image: str
    created: datetime
    airflow_run_id: str = Field(min_length=1)

    @staticmethod
    def uri(release_uri: str) -> str:
        """Return the URI of a run's hand-off file."""
        return f'{release_uri.rstrip("/")}/release_candidate.json'
