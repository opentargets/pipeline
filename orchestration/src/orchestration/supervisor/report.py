"""Rendering a wakeup as markdown for the run's GitHub issue.

`render_comment` turns an `Observation` — what `observer.observe` decided is new since
the last wakeup — into the body of one issue comment. It never reads the journal or the
snapshot for anything the `Observation` has already filtered; the whole point of
`observe` doing that filtering is that this module does not have to re-decide what is
worth saying.

That restraint is the actual requirement, not a style preference. This comment lands on
a GitHub issue every ten minutes for potentially more than a day. A comment that
carries nothing new trains the reader to skip the thread, and the one wakeup that
matters gets skipped along with it — which is worse than never having commented, because
it looks like the thread is being watched when it is not. So an `Observation` with
nothing new, and no dataset comparison to report, renders **no comment at all**:
`render_comment` returns `None`, not an empty or placeholder string, and callers must
treat `None` as "post nothing" rather than degrade to posting a heartbeat.

`Snapshot` is consulted for identity only — `dag_id`/`run_id`/`taken_at`, to stamp which
run and which wakeup a comment belongs to — never for its counts or its currently-running
tasks. Those describe the whole world as of this wakeup, and dumping them here would be
exactly the noise this module exists to avoid; `snapshot.render_snapshot` is where that
whole-world view belongs, for a human pulling a snapshot on demand.
"""

from __future__ import annotations

from orchestration.supervisor.diff import DatasetDiff, is_material
from orchestration.supervisor.observer import Observation, StepCompletion, StepFailure, StepStall
from orchestration.supervisor.snapshot import Snapshot

_NO_MATERIAL_DIFF = 'No material differences against the reference release.'
"""Shown when a dataset comparison ran and found nothing worth reporting — stated
plainly rather than left as an absent section, which would read as the comparison
never having run at all."""


def _format_duration(seconds: float) -> str:
    """Render a duration for a human, in the coarsest unit that keeps it readable.

    Existing renderers in this package (`snapshot.render_snapshot`) show stall
    durations as a bare `{hours:.1f}h`, which is fine when every duration is hours-scale.
    A step completion is routinely minutes-scale, where that format would print `0.1h`
    and leave the reader to convert it back — the exact arithmetic this exists to save
    them. One format handles both scales without forcing a unit choice on the caller.

    Args:
        seconds: The duration, in seconds. Never negative in practice (elapsed time and
            recorded durations both are), but a negative input is clamped to zero rather
            than rendered as a negative duration, which would not be a duration at all.

    Returns:
        `'5h'` for a duration that lands on an exact hour, `'1h01m'` when it does not
        (seconds are dropped at hour scale — precision a reader does not need there),
        `'4m12s'` at minute scale, `'12s'` below a minute.
    """
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f'{hours}h{minutes:02d}m' if minutes else f'{hours}h'
    if minutes:
        return f'{minutes}m{secs:02d}s'
    return f'{secs}s'


def _label(step: str, map_index: int) -> str:
    """Render a step name, qualified with its shard index when it names one of many.

    The two Google Batch steps expand into many task instances under one `task_id`
    (`gentropy_variant_annotation`, `gentropy_l2g_prediction`); reporting the bare step
    name for one shard reads as the whole step, which is wrong when 39 other shards are
    unaffected. `map_index=-1` is Airflow's value for a task instance outside a mapped
    operator (see `TaskInstance.ref`), so it is the one case that must stay unqualified.

    Args:
        step: The bare `unified_pipeline.yaml` step name.
        map_index: The task instance's map_index, or -1 outside a mapped operator.

    Returns:
        `step` unchanged outside a mapped operator, `step[map_index]` for a shard.
    """
    return f'{step}[{map_index}]' if map_index != -1 else step


def _render_run_finished(state: str, snapshot: Snapshot) -> str:
    """Render the run reaching a terminal state.

    Deliberately its own heading rather than folded into `_render_completed`: "the run
    finished" and "a step finished" are different kinds of news for the reader, and
    collapsing them into one bulleted list would bury the run-level headline among
    however many steps happened to finish in the same wakeup.

    Args:
        state: `snapshot.run_state`'s terminal value, `'success'` or `'failed'`.
        snapshot: For `dag_id`/`run_id`, so the run is named even if the reader has
            several issue threads open.

    Returns:
        One rendered heading line.
    """
    verb = 'succeeded' if state == 'success' else 'FAILED'
    return f'### Run {verb} — `{snapshot.dag_id}` / `{snapshot.run_id}`'


def _render_failed(failures: list[StepFailure]) -> str:
    """Render newly failed task instances.

    Args:
        failures: As `Observation.failed`. Never empty when called.

    Returns:
        A heading followed by one bullet per failure.
    """
    lines = ['**Failed**']
    lines.extend(f'- `{_label(f.step, f.map_index)}`' for f in failures)
    return '\n'.join(lines)


def _render_stalled(stalls: list[StepStall]) -> str:
    """Render newly stalled task instances.

    Each line states elapsed time against the threshold it crossed and which rule
    fired. `basis='history'` is called out with its own explanation rather than printed
    as a bare word: `stall.py` documents it as reachable only when a step is cleared and
    re-run within the same DAG run, so it is unusual enough that a reader seeing it
    deserves to know why, not just that it happened.

    Args:
        stalls: As `Observation.stalled`. Never empty when called.

    Returns:
        A heading followed by one bullet per stall.
    """
    lines = ['**Stalled**']
    for stall in stalls:
        elapsed = _format_duration(stall.elapsed)
        threshold = _format_duration(stall.threshold)
        label = _label(stall.step, stall.map_index)
        if stall.basis == 'history':
            note = (
                f'past its own {threshold} history-based threshold — unusual: this step already '
                'completed once earlier in this run, and is now taking unusually long on a retry'
            )
        else:
            note = f'past the {threshold} ceiling (no completed run of this step yet this run)'
        lines.append(f'- `{label}` — running {elapsed}, {note}')
    return '\n'.join(lines)


def _render_completed(completions: list[StepCompletion]) -> str:
    """Render newly completed steps.

    Args:
        completions: As `Observation.completed`. Never empty when called.

    Returns:
        A heading followed by one bullet per completion.
    """
    lines = ['**Completed**']
    lines.extend(f'- `{_label(c.step, c.map_index)}` finished in {_format_duration(c.duration)}' for c in completions)
    return '\n'.join(lines)


def _render_diffs(diffs: list[DatasetDiff], threshold: float) -> str:
    """Render a dataset comparison against the reference release.

    Condensed relative to `cli.render_diff`: this lands inside a running issue thread,
    not a one-off report, so only material datasets are listed (schema changes are
    always material, per `is_material`, and always shown). A dataset present on only one
    side is called out by name rather than left to be inferred from a missing row.

    Args:
        diffs: Every dataset compared, as returned by `gcs.collect_diffs`.
        threshold: Fractional change past which a size or row move counts as material.

    Returns:
        A heading, a one-line summary of how many datasets were compared and how many
        were material, and one entry per material dataset.
    """
    material = [d for d in diffs if is_material(d, threshold)]
    lines = [
        f'**Dataset comparison** — {len(diffs)} dataset(s) compared against the reference release, '
        f'{len(material)} with material changes'
    ]
    if not material:
        lines.append(_NO_MATERIAL_DIFF)
        return '\n'.join(lines)

    for diff in material:
        if diff.side != 'both':
            where = 'the run only' if diff.side == 'run_only' else 'the reference only'
            lines.append(f'- `{diff.dataset}` — present in {where}')
        else:
            rows_before = '-' if diff.reference_rows is None else f'{diff.reference_rows:,}'
            rows_after = '-' if diff.run_rows is None else f'{diff.run_rows:,}'
            lines.append(
                f'- `{diff.dataset}` — rows {rows_before} -> {rows_after}, '
                f'bytes {diff.reference_bytes:,} -> {diff.run_bytes:,}'
            )
        for change in diff.columns:
            types = f'{change.reference_type or "-"} -> {change.run_type or "-"}'
            lines.append(f'    - {change.kind}: `{change.column}` ({types})')
    return '\n'.join(lines)


def render_comment(
    observation: Observation,
    snapshot: Snapshot,
    diffs: list[DatasetDiff] | None = None,
    diff_threshold: float = 0.05,
) -> str | None:
    """Render one wakeup as the body of a GitHub issue comment.

    Sections appear only when they have something new to say, in the order a reader
    should see them: the run reaching a terminal state first (the single biggest thing
    that can happen to a run), then failures and stalls (escalations), then completions,
    then a dataset comparison. This mirrors `snapshot.render_snapshot`'s rule that
    escalations are never folded into routine detail.

    Args:
        observation: What `observer.observe` decided is new since the journal was last
            read. See the module docstring for why nothing else is consulted for
            "what's new" — everything here is already filtered.
        snapshot: The run this wakeup belongs to, read only for `dag_id`/`run_id` (to
            name the run) and `taken_at` (to stamp the comment) — never for its counts
            or running tasks, which are the whole-world view this module deliberately
            does not repeat.
        diffs: The dataset comparison against the reference release, when one ran this
            wakeup — the caller runs it once, at the run's terminal state (see
            `cli.py`'s module docstring for why not every wakeup). `None` when no
            comparison ran this wakeup, which is the common case and renders no section
            at all; an empty list means a comparison ran and found zero datasets, which
            is itself worth a line rather than silence — the two are not the same and
            are rendered differently.
        diff_threshold: Fractional change past which a size or row move in `diffs`
            counts as material. Unused when `diffs` is None.

    Returns:
        The rendered markdown, or `None` when there is nothing new to report: this
        wakeup's `observation` was empty (`Observation.is_empty`) and no dataset
        comparison ran. `None` is the sentinel a caller checks before posting — never an
        empty string, which a broken renderer could also produce for the same case
        without anyone noticing the difference.
    """
    if observation.is_empty and diffs is None:
        return None

    sections: list[str] = []
    if observation.run_finished is not None:
        sections.append(_render_run_finished(observation.run_finished, snapshot))
    if observation.failed:
        sections.append(_render_failed(observation.failed))
    if observation.stalled:
        sections.append(_render_stalled(observation.stalled))
    if observation.completed:
        sections.append(_render_completed(observation.completed))
    if diffs is not None:
        sections.append(_render_diffs(diffs, diff_threshold))

    stamp = snapshot.taken_at.strftime('%Y-%m-%d %H:%M UTC')
    header = f'#### {snapshot.dag_id} / {snapshot.run_id} — {stamp}'
    return '\n\n'.join([header, *sections])
