"""Names of the release-candidate resources derived from a run's release URI."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, cast

_RUN_RE = re.compile(r'^(platform|ppp)-(\d{2})(\d{2})-(\d+)(-[a-z0-9]+)*$')
_PREFIX_RE = re.compile(r'^[a-z][a-z0-9]*$')
_NAME_RE = re.compile(r'^[a-z]([-a-z0-9]{0,61}[a-z0-9])?$')


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
