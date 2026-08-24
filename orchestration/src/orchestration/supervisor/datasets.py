"""Which release datasets a pipeline step produces, per its stage config.

A task's `destination:` can be a scalar string, a list of strings, or a mapping whose
values are paths. `pts/config.yaml` and `pis/config.yaml` use two of those shapes
today — 206 scalar and 51 mapping declarations, no list. The list branch is handled
anyway, defensively, in case the yaml grows one; it is not exercised by either config
as of 2026-08-24.

`foreach:` fans a task out at run time by templating one nested task under `do:`; the
nested task carries the real `destination`, one level below the step's own task list,
so a walk that only reads the step's own list misses it entirely.

Only `output/` and `view/` paths are release datasets. `intermediate/` is scratch
written and consumed between steps within a single run, never published, and diffing
it against a baseline would bury real findings in noise that resets every run.

A templated destination (containing `${...}`) only resolves at run time inside
`foreach:`; as of 2026-08-24 no release dataset is templated in either config, so this
filter is defensive rather than load-bearing today. It is kept because a templated path
emitted verbatim would match nothing in GCS and read as a missing dataset — the one
failure mode that looks exactly like a real finding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from orchestration.supervisor.step_identity import identify

RELEASE_NAMESPACES = ('output/', 'view/')
"""The only `destination:` prefixes that name a published release dataset."""

_SRC_ROOT = Path(__file__).resolve().parents[1]
"""`src/orchestration`, two directories above this file.

This holds in both a source checkout and the Airflow container: `compose.yaml`
mounts the whole `src/` directory at `/opt/airflow/dags`, so this file lives at
`.../orchestration/supervisor/datasets.py` in the checkout and at
`/opt/airflow/dags/orchestration/supervisor/datasets.py` in the container, and
either way `parents[1]` is the `orchestration` directory that holds `dags/`. A path
built under it, like `_UNIFIED_PIPELINE_YAML` below, therefore resolves correctly
in both layouts without needing to know which one it is running in.
"""

_UNIFIED_PIPELINE_YAML = _SRC_ROOT / 'dags' / 'config' / 'unified_pipeline.yaml'

_STAGE_ROOT = _SRC_ROOT.parents[2]
"""Where `pis/` and `pts/` live: the repo root in a source checkout, `/opt` in the
container.

Unlike `_SRC_ROOT`, this crosses out of `src/` entirely, so the two layouts are not
the same directory by construction — they only happen to hold the same two files at
this depth. `compose.yaml` mounts `../pis/config.yaml` and `../pts/config.yaml` to
`/opt/pis/config.yaml` and `/opt/pts/config.yaml` specifically so that this offset —
three levels above `_SRC_ROOT`, matching `dags/config/unified_pipeline.py`'s own
`config_path.parents[4]` from one directory deeper — lands on them in the container
too. Getting this offset wrong does not raise: a wrong depth still resolves to some
existing directory and silently reads whatever config lives there instead.
"""


def _raw_destinations(tasks: list[dict[str, Any]]) -> list[Any]:
    """Collect every task's raw `destination:` value, recursing into `foreach`/`do`.

    Args:
        tasks: A stage config's task list for one step.

    Returns:
        Each task's `destination` value in its original shape (scalar, list, or
        mapping), in task order. Tasks without a `destination` are skipped.
    """
    raw: list[Any] = []
    for task in tasks:
        destination = task.get('destination')
        if destination is not None:
            raw.append(destination)
        do = task.get('do')
        if do:
            raw.extend(_raw_destinations(do))
    return raw


def destinations_for(step: str, stage_config: dict[str, Any]) -> list[str]:
    """Derive the release datasets a step produces from its stage config.

    `stage_config` is assumed well-formed. It is a repo-tracked YAML file that pis/pts
    themselves parse and run against, not an externally-written, eventually-consistent
    record like the run journal — contrast `stall.baseline_from_journal`, which must
    degrade gracefully because it cannot make that assumption. A malformed
    `stage_config` (a non-string destination, a `steps:` entry that is not a mapping)
    is a bug in the caller and is left to raise rather than silently producing a
    partial or wrong answer.

    Args:
        step: The `unified_pipeline.yaml` step name, e.g. `pts_disease`.
        stage_config: The parsed config for the step's stage (`pis/config.yaml` or
            `pts/config.yaml`).

    Returns:
        The step's `output/`/`view/` destinations, in first-seen order with
        duplicates removed. Empty if `step` resolves but has no matching entry in
        `stage_config` — most steps produce no release dataset at all, which is
        expected, not an error.

    Raises:
        ValueError: If `step` itself is malformed — carries no stage or no config key,
            per `identify`. This is distinct from a step that resolves but has no
            config *entry*, which is not an error and returns `[]`.
    """
    tasks = stage_config.get('steps', {}).get(identify(step).config_key)
    if not tasks:
        return []

    destinations: list[str] = []
    for raw in _raw_destinations(tasks):
        if isinstance(raw, dict):
            paths = raw.values()
        elif isinstance(raw, list):
            paths = raw
        else:
            paths = [raw]
        for path in paths:
            if path.startswith(RELEASE_NAMESPACES) and '${' not in path and path not in destinations:
                destinations.append(path)
    return destinations


def stage_configs() -> dict[str, Any]:
    """Load `pis` and `pts`'s own configs, the only two `destinations_for` needs.

    Gentropy's steps are deliberately not read here: their destinations live in
    `dags/config/gentropy.yaml`, and callers walking `unified_pipeline_steps()`
    record them as having no stage config rather than needing a third config this
    does not parse the same way.

    Returns:
        Each stage's parsed config, keyed by stage name.
    """
    return {stage: yaml.safe_load((_STAGE_ROOT / stage / 'config.yaml').read_text()) for stage in ('pis', 'pts')}


def unified_pipeline_steps() -> list[str]:
    """Load the step list a full comparison walks, from `unified_pipeline.yaml`.

    Returns:
        Every step name declared under `steps:`, in file order.
    """
    up = yaml.safe_load(_UNIFIED_PIPELINE_YAML.read_text())
    return list(up['steps'])


def run_name() -> str | None:
    """Read `run_name` straight out of `unified_pipeline.yaml`, the same file the DAG parsed.

    `dags/config/unified_pipeline.py`'s own `UnifiedPipelineConfig.run_name` falls back
    to a fresh timestamp (`datetime.now().strftime(...)`) when the field is unset. This
    function does not reproduce that fallback: the timestamp is generated once, at
    DAG-parse time, on the Airflow side, and there is no way to recover that exact value
    by re-reading the file afterwards — a second timestamp minted here would not be the
    DAG's `run_name`, it would just look like it.

    Whether this field is in fact the run's own prefix in the runs bucket is unverified
    for any particular run — see `cli.py`'s module docstring on why `observe`'s dataset
    diff does not auto-derive `--reference` from it. This function only reads whatever
    is in the config the observer can see; it is a config snapshot, not confirmed
    provenance.

    That snapshot is also not necessarily the observed run's own: `unified_pipeline.yaml`
    is read fresh at wakeup time from the observer's own checkout, never from the run
    being watched. A run discovered on the first wakeup after the cron is enabled may
    have started well before that wakeup, against a yaml that has since been edited for
    the next dev run — so the value returned here can describe a different run than the
    one it gets journalled against, with nothing to signal the mismatch.

    Returns:
        The `run_name` field from the config the observer can currently see, or None if
        the field is unset in the yaml (in which case the DAG itself picked a timestamp
        this function cannot see).
    """
    up = yaml.safe_load(_UNIFIED_PIPELINE_YAML.read_text())
    return up.get('run_name')
