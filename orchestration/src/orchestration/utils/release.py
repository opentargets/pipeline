"""Release-candidate resource names from a run's release URI, and clean-up helpers for the release DAGs."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from airflow.sdk.exceptions import AirflowException
from google.api_core.exceptions import NotFound
from google.cloud import compute_v1
from google.cloud.storage import Client

from orchestration.dags.config.release import HandOff
from orchestration.utils import GCSPath

_RUN_RE = re.compile(r'^(platform|ppp)-(\d{2})(\d{2})-(\d+)(-[a-z0-9]+)*$')
_PREFIX_RE = re.compile(r'^[a-z][a-z0-9]*$')
_NAME_RE = re.compile(r'^[a-z]([-a-z0-9]{0,61}[a-z0-9])?$')
# states in which a task ran in this try, so any XCom it holds is from this try
RAN = {'success', 'failed'}


@dataclass(frozen=True)
class CandidateNames:
    """Names of one run's release candidate."""

    release_uri: str
    prefix: str
    run: str
    product: Literal['platform', 'ppp']
    release: str
    """Release as 'YY.MM'."""
    yymm: str
    """Release as 'YYMM'."""
    ch_snapshot: str
    os_snapshot: str
    ch_disk: str
    os_disk: str
    ch_build_disk: str
    """The disk ClickHouse runs on during the build, copied onto `ch_disk`."""


def candidate_names(release_uri: str) -> CandidateNames:
    """Derive the snapshot and disk names from a run's release URI.

    Args:
        release_uri: ``gs://<bucket>/.../<prefix>/<flavor>-YYMM-N[-suffix]``.

    Raises:
        ValueError: if the URI has no prefix or a malformed run, or a name is not a valid GCE name.
    """
    segments = release_uri.rstrip('/').split('/')
    if segments[0] == 'gs:':
        segments = segments[2:]
    if len(segments) < 3:
        raise ValueError(f"'{release_uri}' needs a '<prefix>/<run>' path after the bucket")
    prefix, run = segments[-2], segments[-1]
    if not _PREFIX_RE.fullmatch(prefix):
        raise ValueError(f"prefix '{prefix}' must match '{_PREFIX_RE.pattern}'")
    match = _RUN_RE.fullmatch(run)
    if match is None:
        raise ValueError(f"run '{run}' must match '(platform|ppp)-YYMM-N[-suffix]'")
    if int(match.group(4)) < 1:
        raise ValueError('run revision must be a positive integer starting from 1')

    base = f'{prefix}-{run}'
    names = {
        'ch_snapshot': f'{base}-ch',
        'os_snapshot': f'{base}-os',
        'ch_disk': f'pos-{base}-ch',
        'os_disk': f'pos-{base}-os',
        'ch_build_disk': f'pos-{base}-ch-build',
    }
    for name in names.values():
        if not _NAME_RE.fullmatch(name):
            raise ValueError(f"'{name}' is not a valid resource name")
    return CandidateNames(
        release_uri=release_uri,
        prefix=prefix,
        run=run,
        product=cast("Literal['platform', 'ppp']", match.group(1)),
        release=f'{match.group(2)}.{match.group(3)}',
        yymm=f'{match.group(2)}{match.group(3)}',
        **names,
    )


def task_states(ti: Any, task_ids: list[str]) -> dict[str, str | None]:
    """States of `task_ids` in this DAG run."""
    states = ti.get_task_states(dag_id=ti.dag_id, task_ids=task_ids, run_ids=[ti.run_id])
    return states.get(ti.run_id, {})


def created_this_try(
    ti: Any, producers: Mapping[str, str], states: Mapping[str, str | None], guard: str = 'guard'
) -> list[str]:
    """Resource names this try created, from the producers' XComs (task id to XCom key).

    Nothing unless `guard` passed. A producer's XCom counts only if it ran in this try: a cleared
    run keeps the XComs of tasks that never started again.
    """
    if states.get(guard) != 'success':
        return []
    names = []
    for task_id, key in producers.items():
        if states.get(task_id) in RAN and (name := ti.xcom_pull(task_ids=task_id, key=key)):
            names.append(name)
    return names


# side -> (snapshot short name, working-disk short names)
SIDES = {'clickhouse': ('ch', ['ch_build', 'ch']), 'opensearch': ('os', ['os'])}


def _provenance(description: str) -> dict[str, Any] | None:
    """The snapshot's recorded images, or None if it records none."""
    try:
        provenance = json.loads(description)
    except ValueError:
        return None
    if not isinstance(provenance, dict) or not {'pos_image', 'server_image'} <= provenance.keys():
        return None
    return provenance


def plan_candidate(names: Mapping[str, Any], project_id: str, zone: str) -> dict[str, dict[str, Any]]:
    """Decide per side whether to build the snapshot or reuse an existing one.

    A snapshot is reused only if it is READY and was built by this run's pos and server images.

    Returns:
        ``{side: {'action': 'build' | 'reuse', 'disk_size_gb': int}}``; a reused side keeps the
        snapshot's own size.

    Raises:
        AirflowException: if the hand-off exists, a snapshot is not READY or was built by other
            images, or a side to build has a leftover disk. Nothing is deleted.
    """
    problems: list[str] = []
    plan: dict[str, dict[str, Any]] = {}

    handoff_uri = HandOff.uri(names['release_uri'])
    bucket, path = GCSPath(handoff_uri).split()
    if Client(project=project_id).bucket(bucket).blob(path).exists():
        problems.append(f'blob {handoff_uri} already exists')

    snapshots = compute_v1.SnapshotsClient()
    disks = compute_v1.DisksClient()
    for side, (short, disk_shorts) in SIDES.items():
        snapshot_name = names[f'{short}_snapshot']
        try:
            snap = snapshots.get(project=project_id, snapshot=snapshot_name)
        except NotFound:
            snap = None

        if snap is None:
            plan[side] = {'action': 'build', 'disk_size_gb': names[f'{side}_disk_gb']}
            for disk_short in disk_shorts:
                disk_name = names[f'{disk_short}_disk']
                try:
                    disks.get(project=project_id, zone=zone, disk=disk_name)
                except NotFound:
                    continue
                problems.append(f'disk {disk_name} already exists')
        elif snap.status in ('CREATING', 'UPLOADING'):
            problems.append(
                f'snapshot {snapshot_name} is {snap.status}; another run may still be creating it, '
                'wait for it to finish'
            )
        elif snap.status != 'READY':
            problems.append(f'snapshot {snapshot_name} is {snap.status}, not READY; delete it to rebuild')
        elif (provenance := _provenance(snap.description)) is None:
            problems.append(f'snapshot {snapshot_name} has no provenance; delete it to rebuild')
        elif (
            provenance['pos_image'] != names['pos_image']
            or provenance['server_image'] != names[f'{side}_image']
        ):
            problems.append(
                f'snapshot {snapshot_name} was built by other images ({snap.description}); delete it to rebuild'
            )
        else:
            plan[side] = {'action': 'reuse', 'disk_size_gb': snap.disk_size_gb}

    if problems:
        raise AirflowException('; '.join(problems))
    return plan
