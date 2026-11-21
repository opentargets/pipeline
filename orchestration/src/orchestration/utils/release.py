"""Release-candidate resource names from a run's release URI, and clean-up helpers for the release DAGs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

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
