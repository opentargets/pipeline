"""Which release datasets a pipeline step produces, per its stage config.

A task's `destination:` can be a scalar string, a list of strings, or a mapping whose
values are paths — `pts/config.yaml` and `pis/config.yaml` use all three shapes today.
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

from typing import Any

from orchestration.supervisor.step_identity import identify

RELEASE_NAMESPACES = ('output/', 'view/')
"""The only `destination:` prefixes that name a published release dataset."""


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

    Args:
        step: The `unified_pipeline.yaml` step name, e.g. `pts_disease`.
        stage_config: The parsed config for the step's stage (`pis/config.yaml` or
            `pts/config.yaml`).

    Returns:
        The step's `output/`/`view/` destinations, in first-seen order with
        duplicates removed. Empty if the step has no matching config entry — most
        steps produce no release dataset at all, which is expected, not an error.
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
