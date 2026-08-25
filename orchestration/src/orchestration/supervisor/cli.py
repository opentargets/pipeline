"""Command line entry point for the pipeline supervisor.

Exposes billed cost for a run and for one step's history, both read from the GCP
billing export, a read-only `snapshot` of a run's state read from Airflow and the
run's journal, and `observe` — one wakeup of the stateless observer: discover the
active run, snapshot it, decide what is new since the journal, comment on the run's
GitHub issue when there is something worth saying, and only then journal it. Posting
before journalling is deliberate, not incidental — see the comment in `main` above
the `observe` branch's post/journal ordering for why reversing it silently loses
reports rather than merely duplicating one.

`--run` and `--step` are matched against GCP labels, which are normalised, so both
are passed through `clean_label` first. A run ID copied straight out of the Airflow
UI therefore works.

Every figure the `usage`/`history` commands report covers only the resources the
pipeline labelled per step, which is not everything it spends. The `usage` report
says so with a coverage line rather than leaving its total to be read as the cost of
the run. Cost only: the export is hourly-bucketed and cannot support a per-step
duration, which comes from Airflow task instances instead — see `snapshot`.

`observe` diffs the run against a reference release only when both `--run` and
`--reference` are given, and only once the run has reached a terminal state — never
on every wakeup, which is what makes the ten-minute cron cheap (see the module
docstring on `report.py`). Neither is auto-derived: `unified_pipeline.yaml` carries a
`run_name`/`release_name` pair that *look* like the right values, but confirming
they are the GCS prefixes those fields actually mean (as opposed to, say, the name
this dev run is working towards rather than a baseline to diff against) is a product
decision this module does not make silently. Until that is settled, the diff stays
opt-in via these two flags, which is also what lets `--dry-run` be exercised against
a real run today without it.

`observe` discovers the run to watch with `active_dag_run`, falling back to
`AirflowClient.most_recent_dag_run` when nothing is running: a run that finished
between two wakeups is not "running" any more, so `active_dag_run` alone can never
lead this command into its own terminal-state branch — the fallback is what lets a
finished run still be found, comment once more, and go quiet. Once found, the diff
itself runs at most once per run: the terminal branch checks the journal for a
`dataset_diff_completed` marker before running `collect_diffs`, and journals that
marker alongside the wakeup's other events (after the post, per the ordering note
below) — without it, every wakeup after the run finishes would re-run the comparison
and repost the same "Dataset comparison" section forever. A separate
`observation_started` event, carrying `run_name` from `unified_pipeline.yaml`, is
journalled on the first wakeup that finds this run (idempotent like every other
event here) — a durable "this run was discovered" marker letting a human map a
journal prefix back to a bucket path. It is not a liveness marker: because it is
journalled once per run and every later wakeup no-ops, a dead observer and a quiet
one produce byte-identical journals, so liveness is not addressed here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Literal

import requests
from google.api_core.exceptions import GoogleAPICallError
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery, dataproc_v1, secretmanager, storage

from orchestration.supervisor.airflow import AirflowClient
from orchestration.supervisor.compute import (
    ClusterCompute,
    StepCompute,
    cluster_compute_report,
    compute_report,
    unresolved_job_count,
)
from orchestration.supervisor.dataproc import TERMINAL_JOB_STATES, job_executions
from orchestration.supervisor.datasets import run_name, stage_configs, unified_pipeline_steps
from orchestration.supervisor.diff import DatasetDiff, is_material
from orchestration.supervisor.gcs import Skipped, collect_diffs, footer_reader
from orchestration.supervisor.github import GitHubApp, read_app_key
from orchestration.supervisor.journal import Journal, JournalEvent, heartbeat_event
from orchestration.supervisor.observer import Observation, observe
from orchestration.supervisor.report import render_comment
from orchestration.supervisor.snapshot import render_snapshot, take_snapshot
from orchestration.supervisor.usage import (
    BillingExport,
    StepUsage,
    WindowCoverage,
    total_cost,
    usage_window,
)
from orchestration.utils.common import (
    AIRFLOW_BASE_URL,
    BILLING_EXPORT_START,
    GCP_PROJECT_PLATFORM,
    GCP_REGION,
    GCS_PIPELINE_RUNS_BUCKET,
    GCS_PRE_RELEASES_BUCKET,
    clean_label,
)

RowKey = Literal['run', 'step']
"""Which label identifies a row of the rendered table."""

OptionalColumn = Literal['product', 'currency', 'shared']
"""Columns rendered only when they have something to say about this particular result.

Each marks a way the rows are less comparable than they look: a `platform` figure and a
`ppp` figure are different pipelines, a GBP figure and a USD figure are different money,
and a shared-cluster figure is not the row's own cost at all. When a column would say the
same thing about every row it is noise, so it is left out.
"""

_OPTIONAL_COLUMNS: tuple[OptionalColumn, ...] = ('product', 'currency', 'shared')

_TOTAL_KEYS: tuple[OptionalColumn, ...] = ('product', 'currency')
"""Columns a total may never be summed across, whether or not they are displayed.

Their values make two amounts incomparable: GBP plus USD is not an amount of money, and
platform plus ppp is not one pipeline's cost. `shared` is not among them — a flagged row
and an unflagged one in the same product and currency are the same money, mis-attributed
between steps, so their sum is still what the run spent.

Keyed on the values rather than on what is displayed, so the totals cannot disagree with
the rows above them if the display rule ever changes. When every row agrees this yields
exactly one group, which is the common case and looks like it always did.
"""

_MIN_KEY_WIDTH = 20
"""Floor for the identity column, so a short-labelled table is not cramped."""

_GITHUB_APP_ID = '4699938'
"""The pipeline supervisor App's numeric id (`iss` claim), verified 2026-08-24 against
the real App (`GET /app` returns slug `opentargets-pipeline-supervisor`)."""

_GITHUB_APP_KEY_NAME = 'supervisor-github-app-key'
"""Secret Manager id holding the App's PEM-encoded private key, in `GCP_PROJECT_PLATFORM`."""

_GITHUB_INSTALLATION_ID = 156145657
"""The App's installation on `_GITHUB_REPO`, verified 2026-08-24."""

_GITHUB_REPO = 'opentargets/pipeline'
"""The only repository the App's installation covers."""

_OBSERVE_TERMINAL_RUN_STATES = frozenset({'success', 'failed'})
"""Airflow DAG run states past which the run itself is over, one way or the other.

Mirrors `observer._TERMINAL_RUN_STATES`, kept as its own constant here rather than
imported so this module does not reach into another module's private name for a
two-element frozenset. `observe` (the CLI command) diffs a run against a reference
release only once its state lands in this set — see the module docstring for why not
on every wakeup."""

_DATASET_DIFF_COMPLETED_EVENT = 'dataset_diff_completed'
"""Journal event type marking that the terminal-state dataset comparison has run for
this run.

`observe` checks for this key before calling `collect_diffs`, and journals it once
the comparison has actually run — never speculatively. Without this gate, F1's fix
(discovering a finished run via the `most_recent_dag_run` fallback) would make things
worse, not better: a finished run would be rediscovered on every wakeup forever, and
each one would re-run the full comparison and repost the same "Dataset comparison"
section, re-reading every parquet footer on both sides each time when `--rows` is set.
Carries no `step`/`try_number`/`map_index` — one run has exactly one comparison to
mark, never per-step — so its `JournalEvent.key` is the bare event type."""

_OBSERVATION_STARTED_EVENT = 'observation_started'
"""Journal event type marking that this run has been discovered and is being watched.

Journalled once per run (idempotent, like every other event here), carrying
`run_name` (see `datasets.run_name`) in its payload so a human can map this journal's
GCS prefix — keyed on the Airflow `dag_run_id`, not `run_name`, see `journal.py`'s
module docstring — back to the run's own prefix in the runs bucket.

It is not a liveness marker, despite looking like one. `Journal.append` no-ops on
every wakeup after the first for a given event, which is correct for this event's
real purpose but means a dead observer (the cron stopped firing at 10:00) and a
quiet one (nothing new to report since 10:00) journal the same thing: one
`observation_started` entry and nothing else. The two are indistinguishable from
this event alone — `journal.heartbeat_event`, journalled alongside this one on every
wakeup below *while the run is not yet terminal*, is what closes that gap.

An earlier version of this docstring judged a per-wakeup heartbeat not worth its cost
(144 journal objects a day for liveness alone) and said not to add one. That judgment
is reversed here, because the cost turns out to buy a second thing at no extra price:
`stall.run_stalled`'s `'no_progress'` signature counts these same heartbeats to tell
"the run is quiet because nothing is happening" apart from "the run is quiet because
the observer itself stopped running" — a distinction wall-clock time cannot make,
since a cron down for three hours and a cron that ran every ten minutes through three
genuinely quiet hours both show three hours of wall-clock silence, but the former has
far fewer heartbeats to show for it. One mechanism closing two gaps is what makes the
cost worth paying now.

144 a day only holds while the run is actually running, though: `active_dag_run` falls
back to `most_recent_dag_run` once a run finishes (see the module docstring), so this
run is rediscovered and re-observed on every wakeup forever, with nothing left to say
past its `run_finished`/`dataset_diff_completed` markers. `run_stalled` itself already
returns before reading `events` at all once `run_state != 'running'` (`stall.py`), so a
heartbeat journalled after that point buys nothing even for its own stated purpose. The
heartbeat append below is gated on the run not being in `_OBSERVE_TERMINAL_RUN_STATES`
for exactly that reason — an idle pipeline must not accumulate one heartbeat object per
wakeup, unboundedly, for as long as the cron keeps running after the pipeline is done."""

_RUN_STALL_DETECTED_EVENT_PREFIX = 'run_stall_detected_'
"""Journal event type prefix for a run-level stall, completed with `RunStallVerdict.reason`.

Mirrors `observer.observe`'s own `f'run_stall_detected_{reason}'` string — kept as its
own constant here rather than imported, the same choice `_OBSERVE_TERMINAL_RUN_STATES`
above makes against `observer._TERMINAL_RUN_STATES`, so this module does not reach into
another module's private detail for a one-line format string. `'no_progress'` and
`'stuck_trigger'` are tracked as independent, idempotent-per-reason events this way —
the same "once and never again" idempotency `_DATASET_DIFF_COMPLETED_EVENT`/
`_OBSERVATION_STARTED_EVENT` get for a whole run, applied per reason instead."""

_MISSING = '-'
"""Shown for a row that carries no value for an optional column."""

_SINCE_HELP = """earliest partition date to scan, as YYYY-MM-DD. This is the date the
rows were ingested into the export, not the date of the usage they describe, so it
prunes the scan rather than selecting a period of pipeline activity"""

_EMPTY = """No billed usage found. It may not have billed yet, or the label may not match.
Labels are lowercased with everything outside [a-z0-9-_] replaced by '-', and
--run/--step are normalised the same way, so check the normalised form: an Airflow
run ID like manual__2026-07-21T15:07:47.545737+00:00 is stored as the label
manual__2026-07-21t15-07-47-545737-00-00."""

_CURRENCY_FOOTER = [
    'these rows are billed in more than one currency, so they are totalled',
    'separately. The per-currency totals must not be added together.',
]

_PRODUCT_FOOTER = [
    'product separates the platform pipeline from the partner preview one, which runs',
    'the same DAG and can share a run label. Their rows are never added together.',
]

_SHARED_FOOTER = [
    'shared marks a row where more than one step of this run billed against a single',
    'Dataproc cluster instance, which keeps the labels it was created with — so the row',
    "is charged that instance's whole cost rather than the step's own. No row in the",
    'export has ever been marked: this guards a code path (use_if_exists=True) that does',
    'not currently manifest. Repeated use of one cluster *name* is not this, and is fine.',
]

_COVERAGE_FOOTER = [
    'the denominator is everything the pipeline billed in that window, including any',
    'other run that overlapped it. The remainder is real pipeline spend the steps above',
    'do not account for: Google Batch jobs carry no step labels, and some Dataproc disk',
    'and licensing rows fall outside the labels of the step that caused them.',
]

_IMPOSSIBLE_SHARE_FOOTER = [
    'a share above 100% is not good coverage, it is a broken measurement: the rows above',
    'were counted in the numerator but not in the denominator, which means something the',
    'report shows is not labelled as created by the pipeline. Treat the figure as wrong.',
]

_EMPTY_COMPUTE = """No step joined any cost, execution or wall-time data for this run. It may not
have billed yet, produced no Dataproc job, and never reached the journal -- or the run
label may not match. Labels are lowercased with everything outside [a-z0-9-_] replaced
by '-', and --run is normalised the same way, so check the normalised form: an Airflow
run ID like manual__2026-07-21T15:07:47.545737+00:00 is stored as the label
manual__2026-07-21t15-07-47-545737-00-00."""

_WALL_UNAVAILABLE_FOOTER = [
    'task wall time (queueing under cluster contention) is not shown: no step in this',
    'report has one. That does not mean no step queued -- it means the measurement was',
    "never taken. Wall time comes from the pipeline supervisor's journal, which only",
    'exists for a run the observer watched; Airflow keeps no history of its own once the',
    'VM that ran it is torn down. It arrives with the first run the observer watches.',
]


def _airflow_credentials() -> tuple[str, str]:
    """Read Airflow FAB credentials from the environment.

    Passed as environment variables rather than a flag: a flag's value is visible in
    `ps` output to every user on the machine, and would land in shell history and in
    any cron entry that ran this command. Phase 4 moves these to GCP Secret Manager
    per the design spec's identity section; env vars are the phase 1 answer, not the
    final one.

    Not `_AIRFLOW_WWW_USER_USERNAME` / `_AIRFLOW_WWW_USER_PASSWORD`: those are
    compose-level variables consumed by the `airflow-init` service, with their
    defaults supplied inline by `compose.yaml` — they are never exported into the
    environment of a process running outside Docker, which is exactly where this
    CLI runs.

    Returns:
        Username and password.

    Raises:
        RuntimeError: If either is unset. There is no default: the dev VM's
            credentials happen to be `airflow`/`airflow` today, but baking a
            known-weak default into shipped code is how it ends up somewhere it
            matters.
    """
    username = os.environ.get('AIRFLOW_USERNAME')
    password = os.environ.get('AIRFLOW_PASSWORD')
    if username and password:
        return username, password
    missing = [name for name, value in (('AIRFLOW_USERNAME', username), ('AIRFLOW_PASSWORD', password)) if not value]
    raise RuntimeError(
        f'{" and ".join(missing)} not set. The snapshot command needs Airflow credentials; on '
        "the dev VM these are compose.yaml's _AIRFLOW_WWW_USER_USERNAME / "
        '_AIRFLOW_WWW_USER_PASSWORD defaults, exported into this process.'
    )


def totals_by_group(usages: list[StepUsage]) -> dict[tuple[str, ...], float]:
    """Total net cost per group of rows that may legitimately be added together.

    Args:
        usages: The usages to total.

    Returns:
        Net cost keyed by the `_TOTAL_KEYS` values of the group, in `_TOTAL_KEYS` order,
        sorted. Two currencies are not one amount of money, and two products are not one
        pipeline, so a single total across either is a number describing nothing.
    """
    groups = sorted({tuple(_cell(u, c) for c in _TOTAL_KEYS) for u in usages})
    return {
        group: total_cost(u for u in usages if tuple(_cell(u, c) for c in _TOTAL_KEYS) == group)
        for group in groups
    }


def _cell(usage: StepUsage, column: OptionalColumn) -> str:
    """The rendered value of an optional column for one row."""
    if column == 'shared':
        return 'yes' if usage.shared_cluster else _MISSING
    return getattr(usage, column) or _MISSING


def _is_worth_showing(column: OptionalColumn, usages: list[StepUsage]) -> bool:
    """Whether a column tells the reader something about this particular result.

    `product` and `currency` matter when the rows disagree: one value everywhere is the
    normal case and naming it is noise. `shared` is not symmetric — a table where every
    row is shared is the worst case, not a uniform one — so it shows whenever any row is
    flagged and disappears only when none is.
    """
    if column == 'shared':
        return any(u.shared_cluster for u in usages)
    return len({_cell(u, column) for u in usages}) > 1


def optional_columns(usages: list[StepUsage]) -> list[OptionalColumn]:
    """Which optional columns this result has to show.

    Args:
        usages: The usages to be rendered.

    Returns:
        The columns that have something to say about this result, in table order. A column
        that would say the same thing about every row tells the reader nothing; one that
        would not is the difference between comparable numbers and incomparable ones.
    """
    return [column for column in _OPTIONAL_COLUMNS if _is_worth_showing(column, usages)]


def render_table(usages: list[StepUsage], key: RowKey = 'step') -> str:
    """Render usages as a fixed-width table.

    Args:
        usages: The usages to render.
        key: Which label identifies a row, and so gets the first column. `step` for
            the steps of one run; `run` for one step's history, where every row
            shares the step the user asked for and only the run tells them apart.

    Returns:
        The rendered table, or a message when there is nothing to show.
    """
    if not usages:
        return _EMPTY

    keys = [getattr(u, key) for u in usages]
    width = max(_MIN_KEY_WIDTH, len(key), *(len(k) for k in keys))
    columns = optional_columns(usages)
    widths = {c: max(len(c), *(len(_cell(u, c)) for u in usages)) for c in columns}
    extra_header = ''.join(f' {c:<{widths[c]}}' for c in columns)

    header = f'{key:<{width}} {"tool":<9}{extra_header} {"net cost":>10}'
    lines = [header, '-' * len(header)]
    lines.extend(
        f'{k:<{width}} {u.tool:<9}'
        + ''.join(f' {_cell(u, c):<{widths[c]}}' for c in columns)
        + f' {u.net_cost:>10.2f}'
        for k, u in zip(keys, usages, strict=True)
    )
    lines.append('-' * len(header))

    totals = totals_by_group(usages)
    for group, amount in totals.items():
        values = dict(zip(_TOTAL_KEYS, group, strict=True))
        cells = ''.join(f' {values.get(c, ""):<{widths[c]}}' for c in columns)
        currency = values['currency']
        lines.append(f'{"total":<{width}} {"":<9}{cells} {amount:>10.2f} {currency}')

    blocks = []
    if len({u.currency for u in usages}) > 1:
        blocks.append(_CURRENCY_FOOTER)
    if 'product' in columns:
        blocks.append(_PRODUCT_FOOTER)
    if 'shared' in columns:
        blocks.append(_SHARED_FOOTER)
    if not blocks:
        return '\n'.join(lines)
    return '\n'.join([*lines, '', '\n\n'.join('\n'.join(block) for block in blocks)])


def _stamp(moment: datetime) -> str:
    """Render a window boundary to the minute, in the export's own UTC."""
    return moment.strftime('%Y-%m-%d %H:%M')


def render_coverage(window: tuple[datetime, datetime] | None, coverage: list[WindowCoverage]) -> str:
    """Render how much of the pipeline's spend the table above accounts for.

    Args:
        window: The period measured, or `None` when the run billed nothing and there is
            therefore no window.
        coverage: One entry per currency, as returned by `BillingExport.window_coverage`.

    Returns:
        The rendered coverage note. Never a percentage the data does not support: an
        unknown window and an empty window each say so instead.
    """
    if window is None:
        return (
            'coverage: unknown. No billed usage was found at or after --since, so there is no\n'
            'window over which to compare labelled cost against total pipeline spend. This may\n'
            'mean the run has not billed yet, or that --since prunes the scan past when it did --\n'
            'a --since narrower than the run produces exactly this, not a claim the run is free.'
        )
    period = f'{_stamp(window[0])} to {_stamp(window[1])} UTC'
    if not coverage:
        return f'coverage: no pipeline spend found from {period}, which should not happen.'

    lines = [f'coverage: from {period},']
    for entry in coverage:
        share = 'share undefined' if entry.labelled_share is None else f'{entry.labelled_share:.1%}'
        lines.append(
            f'  the steps above account for {entry.labelled_cost:.2f} of the {entry.pipeline_cost:.2f} '
            f'{entry.currency} the pipeline billed ({share})'
        )
    blocks = [_COVERAGE_FOOTER]
    if any(entry.exceeds_pipeline_cost for entry in coverage):
        blocks.append(_IMPOSSIBLE_SHARE_FOOTER)
    return '\n'.join([*lines, '', '\n\n'.join('\n'.join(block) for block in blocks)])


def _money(value: float | None) -> str:
    return f'{value:.2f}' if value is not None else _MISSING


def _hours(value: int | None) -> str:
    return f'{value}h' if value is not None else _MISSING


def _seconds(value: float | None) -> str:
    return f'{value:,.0f}s' if value is not None else _MISSING


def _core_hours(value: float | None) -> str:
    return f'{value:.1f}' if value is not None else _MISSING


def _share(value: float | None) -> str:
    return f'{value:.0%}' if value is not None else _MISSING


def _families(values: list[str]) -> str:
    return ','.join(values) if values else _MISSING


def _compute_sort_key(step: StepCompute) -> tuple[int, float, int, float, str]:
    """Waste first, worst to least; steps this cannot be judged on sort after, by cost.

    Sorting by cost puts the run's most expensive step first, which is not always the
    step wasting the most cluster time and is not the number a reader can act on -- see
    `compute.py`'s module docstring. A step with no `billed_execution_gap_seconds` (no
    Dataproc job, or no billing) cannot be placed on that axis at all, so it sorts after
    every step that can be, rather than at either extreme of a mixed ranking that would
    misread a `None` as either the worst or the best case.
    """
    gap = step.billed_execution_gap_seconds
    cost = step.net_cost
    return (
        0 if gap is not None else 1,
        -(gap if gap is not None else 0.0),
        0 if cost is not None else 1,
        -(cost if cost is not None else 0.0),
        step.step,
    )


def _compute_table_columns(
    steps: list[StepCompute],
) -> list[tuple[str, Callable[[StepCompute], str]]]:
    """Which columns this result has to show, in table order, and how to render each cell.

    `cost`/`billed`/`exec`/`waste` are always shown -- they are this report's whole
    point. `ccy`/`shared`/`core-hrs`/`spot`/`machines`/`wall`/`queue-gap` are shown only
    when at least one row carries a value for them; a column of nothing but `-` is noise,
    and for `wall`/`queue-gap` specifically it is worse than noise -- see
    `_WALL_UNAVAILABLE_FOOTER`, appended instead whenever this leaves them out.

    Args:
        steps: The rows to be rendered.

    Returns:
        `(header, cell)` pairs, in table order.
    """
    columns: list[tuple[str, Callable[[StepCompute], str]]] = [
        ('cost', lambda s: _money(s.net_cost)),
    ]
    if len({s.currency for s in steps if s.currency is not None}) > 1:
        columns.append(('ccy', lambda s: s.currency or _MISSING))
    columns.append(('billed', lambda s: _hours(s.billed_hours)))
    columns.append(('exec', lambda s: _seconds(s.execution_seconds)))
    columns.append(('waste', lambda s: _seconds(s.billed_execution_gap_seconds)))
    if any(s.shared_cluster for s in steps):
        # Explains every `-` in the `waste` column above that is not a "no Dataproc job"
        # row: `waste` reads `-` for a shared step too, and without this column that looks
        # identical to a step that simply never ran on Dataproc -- see F1 in the review
        # ledger and `compute.py`'s `shared_cluster` docstring.
        columns.append(('shared', lambda s: 'yes' if s.shared_cluster else _MISSING))
    if any(s.core_seconds is not None for s in steps):
        columns.append(('core-hrs', lambda s: _core_hours(s.core_hours)))
        columns.append(('spot', lambda s: _share(s.spot_share)))
        columns.append(('machines', lambda s: _families(s.machine_families)))
    if any(s.wall_seconds is not None for s in steps):
        columns.append(('wall', lambda s: _seconds(s.wall_seconds)))
        columns.append(('queue-gap', lambda s: _seconds(s.wall_execution_gap_seconds)))
    return columns


_FAILED_JOB_STATES = TERMINAL_JOB_STATES - {'DONE'}
"""`{'ERROR', 'CANCELLED'}` -- a job that reached a terminal state without succeeding.

Distinct from a job still in progress (`RUNNING`, `PENDING`, `SETUP_DONE`, ...), which
has not failed, it simply has not finished yet -- see `render_compute`'s F3 fix and
`dataproc.TERMINAL_JOB_STATES`."""

def _unresolved_job_block(count: int) -> list[str]:
    """The footer paragraph naming how many Dataproc jobs joined no step, or []."""
    if not count:
        return []
    return [
        f'{count} Dataproc job(s) in this run matched no known step (no step label, and the job',
        "id matched no known step name either) and are not counted in any step's row above --",
        'see compute.unresolved_job_count.',
    ]


def _journal_read_block(journal_prefix: str | None, journal_event_count: int | None) -> list[str]:
    """F4: name which journal prefix was actually read, so a wrong `--run` form is visible.

    `compute --run` accepts both the raw Airflow run id (what the journal is keyed on)
    and the cleaned billing label (what `usage`'s own output, and this flag's help text,
    show) -- billing and Dataproc clean either form to the same value, so both appear to
    work, but the journal never exists under the cleaned form's prefix. Silent today
    because no journal exists for any run yet; stating the prefix and whether it was
    empty is what lets a reader tell "this run predates the observer" apart from "I
    passed --run in the wrong form" once one does.
    """
    if journal_prefix is None:
        return []
    return [
        f'journal prefix read for this report: {journal_prefix} ({journal_event_count} event(s) found).',
        'If --run was given in its cleaned label form (as usage/history print it) rather',
        'than the raw Airflow run id, this is the wrong prefix and will always read empty --',
        'pass the raw id instead.',
    ]


def _cluster_block(clusters: list[ClusterCompute]) -> list[str]:
    """The cluster-idle breakdown for every *shared* instance, or [] when none is shared.

    An unshared instance is already fully represented by its one step's own row above --
    repeating it here would be noise. See F1 in the review ledger and `compute.py`'s
    `cluster_compute_report`.
    """
    shared = sorted((c for c in clusters if c.shared), key=lambda c: -(c.idle_seconds or 0.0))
    if not shared:
        return []

    id_width = max(len('cluster instance'), *(len(c.cluster_instance) for c in shared))
    step_width = max(len('billing step'), *(len(c.billing_step or _MISSING) for c in shared))
    header = (
        f'{"cluster instance":<{id_width}} {"billing step":<{step_width}} {"steps":>5} '
        f'{"billed":>8} {"exec":>10} {"idle":>10}'
    )
    lines = [header, '-' * len(header)]
    lines.extend(
        f'{c.cluster_instance:<{id_width}} {(c.billing_step or _MISSING):<{step_width}} {len(c.steps):>5} '
        f'{_seconds(c.billed_seconds):>8} {_seconds(c.execution_seconds):>10} {_seconds(c.idle_seconds):>10}'
        for c in shared
    )
    lines.append('-' * len(header))
    return [
        'cluster idle time -- billed and not computing, pooled across every step that ran a Dataproc',
        'job on the instance. Every step above showing waste as - because it shares one of these',
        "instances (see the shared column) has its true idle time here instead, at the instance's",
        'own row, not attributed to any one step:',
        '',
        *lines,
    ]


def render_compute(
    steps: list[StepCompute],
    unresolved_jobs: int = 0,
    clusters: list[ClusterCompute] | None = None,
    journal_prefix: str | None = None,
    journal_event_count: int | None = None,
) -> str:
    """Render a per-step compute report: cost, execution time, and the gaps between them.

    Rows are sorted by `billed_execution_gap_seconds`, worst first -- see
    `_compute_sort_key`. That is deliberately not the same as sorting by cost: the
    run's most expensive step and its most wasteful one are not always the same step,
    and the wasteful one is the one a reader can act on. The report does not say why
    the gap is there -- a cluster held open across a step group, a late delete, and
    unlabelled work on the same cluster all look identical here -- only that it is.

    Args:
        steps: One row per step, normally `compute.compute_report`'s result.
        unresolved_jobs: Count of Dataproc jobs that could not be joined to any step,
            normally `compute.unresolved_job_count` on the same executions passed to
            `compute_report`. Defaults to 0 -- most callers (including every existing
            test of this function) have nothing to report here, and the footer this
            adds is silent at 0, so it is additive rather than a required migration.
        clusters: One row per Dataproc cluster instance, normally
            `compute.cluster_compute_report`'s result on the same inputs passed to
            `compute_report`. Only the *shared* instances are rendered -- see
            `_cluster_block` -- so this is what lets a shared step's `-` waste cell be
            followed by the real number, at the instance level, instead of just
            vanishing. Defaults to `None`, rendered the same as `[]`: additive, like
            `unresolved_jobs`.
        journal_prefix: The GCS journal prefix this report actually read, or `None` to
            omit the line entirely (existing callers with nothing to report here are
            unaffected). See `_journal_read_block` and F4 in the review ledger.
        journal_event_count: How many events that prefix returned. Shown alongside
            `journal_prefix`; meaningless without it.

    Returns:
        The rendered report, or a message when there is nothing to show.
    """
    clusters = clusters or []
    if not steps:
        empty_blocks = [
            _unresolved_job_block(unresolved_jobs),
            _journal_read_block(journal_prefix, journal_event_count),
        ]
        blocks = [block for block in empty_blocks if block]
        if blocks:
            return f'{_EMPTY_COMPUTE}\n\n' + '\n\n'.join('\n'.join(block) for block in blocks)
        return _EMPTY_COMPUTE

    ordered = sorted(steps, key=_compute_sort_key)
    columns = _compute_table_columns(ordered)

    key_width = max(_MIN_KEY_WIDTH, len('step'), *(len(s.step) for s in ordered))
    widths = {name: max(len(name), *(len(cell(s)) for s in ordered)) for name, cell in columns}

    header = f'{"step":<{key_width}}' + ''.join(f' {name:>{widths[name]}}' for name, _ in columns)
    lines = [header, '-' * len(header)]
    lines.extend(
        f'{s.step:<{key_width}}' + ''.join(f' {cell(s):>{widths[name]}}' for name, cell in columns)
        for s in ordered
    )
    lines.append('-' * len(header))

    no_job = sum(1 for s in ordered if not s.dataproc_job_states)
    shared_job = sum(1 for s in ordered if s.shared_cluster)
    failed_job = sum(1 for s in ordered if set(s.dataproc_job_states) & _FAILED_JOB_STATES)
    in_flight_job = sum(1 for s in ordered if set(s.dataproc_job_states) - TERMINAL_JOB_STATES)
    step_measured = [s.billed_execution_gap_seconds for s in ordered if s.billed_execution_gap_seconds is not None]
    # Only *shared* clusters, not every cluster `cluster_compute_report` returned: an
    # unshared instance's idle time is already counted once, above, as its one step's own
    # `billed_execution_gap_seconds` -- adding `idle_seconds` for that same instance here
    # too would double it. Only a shared instance's idle time is absent from `step_measured`
    # (its billing step's gap is suppressed to `None`), so only shared instances belong here.
    cluster_measured = [c.idle_seconds for c in clusters if c.shared and c.idle_seconds is not None]

    blocks = [[
        f'{len(ordered)} steps. {no_job} had no Dataproc job at all -- normal for a step that runs on a',
        'plain GCE VM rather than Dataproc, not evidence of a problem.',
    ]]
    if step_measured or cluster_measured:
        total = sum(step_measured) + sum(cluster_measured)
        blocks.append([
            f'billed time paid for and not computing: {total:,.0f}s ({total / 3600:.1f}h), a floor covering',
            f'{len(step_measured)} step(s) with an exclusive Dataproc cluster and {len(cluster_measured)} shared',
            'cluster instance(s) (see the table below). Steps with no Dataproc job, and any step whose',
            'usage could not be attributed to exactly one cluster instance, are not counted here at all --',
            'not counted as zero waste.',
        ])
    if shared_job:
        blocks.append([
            f'{shared_job} step(s) share a Dataproc cluster instance with other steps in this run -- their',
            'own waste could not be isolated from the instance (waste reads - above); see the cluster',
            "breakdown below for the instance's true idle time, pooled across every step that used it.",
        ])
    if failed_job:
        blocks.append([
            f'{failed_job} step(s) had a Dataproc job that reached a terminal state other than DONE',
            '(cancelled or errored) alongside or instead of a successful one -- see dataproc_job_states',
            'in --json for which.',
        ])
    if in_flight_job:
        blocks.append([
            f'{in_flight_job} step(s) had a Dataproc job still in progress (pending, running, or setting',
            'up) when this report ran -- expected for the currently active run, not evidence anything',
            'failed. Re-run the report once the run finishes for a settled figure.',
        ])
    unresolved_block = _unresolved_job_block(unresolved_jobs)
    if unresolved_block:
        blocks.append(unresolved_block)
    cluster_block = _cluster_block(clusters)
    if cluster_block:
        blocks.append(cluster_block)
    if not any(s.wall_seconds is not None for s in ordered):
        journal_block = list(_WALL_UNAVAILABLE_FOOTER)
        journal_block.extend(_journal_read_block(journal_prefix, journal_event_count))
        blocks.append(journal_block)

    return '\n'.join([*lines, '', '\n\n'.join('\n'.join(block) for block in blocks)])


_UNCOUNTABLE = 'n/a'
"""Shown for a row count the dataset's format has no footer to have supplied, as
distinct from `_MISSING`, which marks a side that is absent altogether."""

def _count(value: int | None, countable: bool) -> str:
    """Render a row count, distinguishing unavailable from zero.

    Args:
        value: The count, or None.
        countable: Whether the dataset's format has row counts at all.

    Returns:
        The number, `n/a` when the format has no footer, or `-` when the side is absent.
    """
    if value is not None:
        return f'{value:,}'
    return _UNCOUNTABLE if not countable else _MISSING


def render_diff(diffs: list[DatasetDiff], skipped: Skipped, threshold: float, rows_skipped: bool = False) -> str:
    """Render a dataset comparison as text.

    Schema changes are listed for every dataset that has one, never hidden behind the
    threshold. One-sided datasets are called out explicitly rather than left to be
    inferred from an absent row. The footer states what the comparison did not cover,
    because a report that silently omits things reads as a clean run.

    Args:
        diffs: Every dataset compared.
        skipped: What was not covered.
        threshold: Fractional change past which a size or row move is reported.
        rows_skipped: True when footer reads were skipped for this run, which is the
            default (`--rows` opts into them). Without this, every row count prints as
            `-` (`_count`'s "absent" symbol, since `countable` is still True for
            parquet), which reads as every dataset's counterpart being missing rather
            than as rows simply not having been read.

    Returns:
        The rendered report.
    """
    material = [d for d in diffs if is_material(d, threshold)]
    lines = [f'{len(diffs)} datasets compared, {len(material)} with material changes']
    lines.append('')

    if not material:
        lines.append('No material differences.')
    for diff in material:
        if diff.side != 'both':
            where = 'the run only' if diff.side == 'run_only' else 'the reference only'
            lines.append(f'{diff.dataset}  PRESENT IN {where.upper()}')
        else:
            rows = f'{_count(diff.reference_rows, diff.countable)} -> {_count(diff.run_rows, diff.countable)}'
            lines.append(
                f'{diff.dataset}  rows {rows}  bytes {diff.reference_bytes:,} -> {diff.run_bytes:,}'
                f'  files {diff.reference_files} -> {diff.run_files}'
            )
        for change in diff.columns:
            types = f'{change.reference_type or _MISSING} -> {change.run_type or _MISSING}'
            lines.append(f'    {change.kind:8} {change.column}  {types}')

    footer = [
        '',
        f'Threshold: {threshold:.0%} on rows and bytes. Schema changes are always reported.',
        'Not compared: intermediate/ (scratch between steps), and templated destinations,',
        'which resolve only at run time.',
    ]
    if rows_skipped:
        footer.append(
            'Row counts were not read (pass --rows to include them, ~7min for a full release). '
            'Sizes, file counts and presence are compared; a row count of "-" above means not '
            'read, not absent.'
        )
    if skipped.stages_without_config:
        footer.append(
            f'{len(skipped.stages_without_config)} steps skipped: their stage has no local config '
            f'(gentropy declares destinations in dags/config/gentropy.yaml).'
        )
    if skipped.steps_without_datasets:
        footer.append(
            f'{len(skipped.steps_without_datasets)} steps declare no release dataset. '
            f'That is normal, not an anomaly.'
        )
    if skipped.datasets_absent_from_both:
        footer.append(
            f'{len(skipped.datasets_absent_from_both)} datasets absent from both buckets, '
            f'usually a step that has not run: {", ".join(sorted(skipped.datasets_absent_from_both)[:5])}'
        )
    if skipped.undeclared_in_buckets:
        footer.append(
            f'{len(skipped.undeclared_in_buckets)} datasets present in a bucket but declared by no '
            f'step, so NOT compared: {", ".join(skipped.undeclared_in_buckets[:5])}. '
            f'A dataset here that exists only in the reference has been dropped from the pipeline.'
        )
    return '\n'.join(lines + footer)


def _observation_events(observation: Observation, at: datetime) -> list[JournalEvent]:
    """Build the journal events one wakeup's `Observation` implies.

    One event per new item, so `Journal.append`'s per-key idempotency covers each of
    them individually — a wakeup that journals three completions and then dies before
    reaching the fourth leaves the first three recorded, not none of them. The payload
    on each carries enough for a human reading the raw journal, and, for
    `step_completed`, the one thing another reader depends on: `stall.
    baseline_from_journal` reads `payload['duration']` back out to build the stall
    baseline, so that key is not optional decoration. A run-level stall is journalled
    under `_RUN_STALL_DETECTED_EVENT_PREFIX` + its `reason`, so `'no_progress'` and
    `'stuck_trigger'` are independently idempotent (see that constant's docstring); the
    heartbeat itself is not built here — it is not derived from `observation` at all,
    so it is appended directly in `main`'s `observe` branch instead.

    Args:
        observation: What `observer.observe` decided is new this wakeup.
        at: When this wakeup ran. The same instant for every event this call produces
            — they were all learned in the same wakeup, even if what they describe
            happened at different times.

    Returns:
        The events to append, in no particular order (each carries a distinct key, so
        `Journal.append` order does not matter here the way `Journal.read`'s
        chronological order does for a reader).
    """
    events = [
        JournalEvent(
            event_type='step_failed', step=f.step, map_index=f.map_index, try_number=f.try_number,
            at=at, payload={'ref': f.ref},
        )
        for f in observation.failed
    ]
    events.extend(
        JournalEvent(
            event_type='stall_detected', step=s.step, map_index=s.map_index, try_number=s.try_number,
            at=at, payload={'ref': s.ref, 'elapsed': s.elapsed, 'threshold': s.threshold, 'basis': s.basis},
        )
        for s in observation.stalled
    )
    events.extend(
        JournalEvent(
            event_type='step_completed', step=c.step, map_index=c.map_index, try_number=c.try_number,
            at=at, payload={'ref': c.ref, 'duration': c.duration},
        )
        for c in observation.completed
    )
    if observation.run_finished is not None:
        events.append(JournalEvent(event_type='run_finished', at=at, payload={'state': observation.run_finished}))
    if observation.run_stall is not None:
        events.append(JournalEvent(
            event_type=f'{_RUN_STALL_DETECTED_EVENT_PREFIX}{observation.run_stall.reason}', at=at,
            payload=observation.run_stall.model_dump(exclude_none=True),
        ))
    return events


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(prog='pipeline-supervisor')
    sub = parser.add_subparsers(dest='command', required=True)

    since_help = ' '.join(_SINCE_HELP.split())

    usage = sub.add_parser('usage', help='billed usage for every step in one run')
    usage.add_argument('--run', required=True, help='the run label to report on')
    usage.add_argument('--since', type=date.fromisoformat, default=BILLING_EXPORT_START, help=since_help)
    usage.add_argument('--json', action='store_true', help='emit JSON instead of a table')

    history = sub.add_parser('history', help="one step's billed usage across runs")
    history.add_argument('--step', required=True, help='the step label to report on')
    history.add_argument('--since', type=date.fromisoformat, default=BILLING_EXPORT_START, help=since_help)
    history.add_argument('--json', action='store_true', help='emit JSON instead of a table')

    compute = sub.add_parser(
        'compute', help='billed cost, Dataproc execution time and task wall time per step, joined'
    )
    compute.add_argument('--run', required=True, help='the run label to report on, as `usage --run`')
    compute.add_argument('--since', type=date.fromisoformat, default=BILLING_EXPORT_START, help=since_help)
    compute.add_argument('--json', action='store_true', help='emit JSON instead of a table')

    snapshot = sub.add_parser('snapshot', help='read-only view of a pipeline run')
    snapshot.add_argument('--run', required=True, help='the Airflow DAG run id')
    snapshot.add_argument('--dag', default='unified_pipeline', help='the DAG to read')
    snapshot.add_argument('--journal-bucket', default=GCS_PIPELINE_RUNS_BUCKET, help="the run's journal bucket")
    snapshot.add_argument('--json', action='store_true', help='emit JSON instead of text')

    diff = sub.add_parser('diff', help="compare a run's datasets against a reference release")
    diff.add_argument('--run', required=True, help='the run name, a prefix in the runs bucket')
    diff.add_argument('--reference', required=True, help='the reference release name')
    diff.add_argument('--threshold', type=float, default=0.05, help='fractional change to report')
    diff.add_argument('--run-bucket', default=GCS_PIPELINE_RUNS_BUCKET, help='bucket holding the run')
    diff.add_argument('--reference-bucket', default=GCS_PRE_RELEASES_BUCKET, help='bucket holding the release')
    diff.add_argument(
        '--rows',
        action='store_true',
        help=(
            'also read row counts from parquet footers. Sizes, file counts and presence are '
            'always compared and take ~10s for a full release; row counts add ~7min (2,602 '
            'footers across both sides, measured 2026-08-24)'
        ),
    )
    diff.add_argument('--json', action='store_true', help='emit JSON instead of text')

    observe = sub.add_parser(
        'observe', help='one wakeup: journal what changed since last time and comment on the GitHub issue'
    )
    observe.add_argument('--dag', default='unified_pipeline', help='the DAG to watch')
    observe.add_argument('--issue', required=True, type=int, help='the GitHub issue to comment on')
    observe.add_argument(
        '--dry-run',
        action='store_true',
        help='render the comment to stdout instead of posting it; write nothing to the journal either',
    )
    observe.add_argument(
        '--run-bucket', default=GCS_PIPELINE_RUNS_BUCKET, help="bucket holding the run's journal and output data"
    )
    observe.add_argument(
        '--run',
        default=None,
        help=(
            "the run's own prefix in --run-bucket, as `diff --run` (not the Airflow run id — see the "
            'module docstring). Required alongside --reference to run the terminal-state dataset '
            'comparison; omitted, the comparison is skipped'
        ),
    )
    observe.add_argument(
        '--reference',
        default=None,
        help=(
            'reference release name to diff the run against once it reaches a terminal state, as '
            '`diff --reference`. Required alongside --run to run the comparison; omitted, it is skipped'
        ),
    )
    observe.add_argument('--reference-bucket', default=GCS_PRE_RELEASES_BUCKET, help='bucket holding the release')
    observe.add_argument('--threshold', type=float, default=0.05, help='fractional change the diff reports')
    observe.add_argument(
        '--rows',
        action='store_true',
        help=(
            'also read row counts on the terminal-state diff. Sizes, file counts and presence are '
            'always compared and take ~10s for a full release; row counts add ~7min, which is why '
            'this defaults off for a cron that wakes every ten minutes — see `diff --rows`'
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument vector, defaulting to `sys.argv[1:]`.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)

    key: RowKey = 'step'
    coverage: list[WindowCoverage] = []
    window: tuple[datetime, datetime] | None = None
    try:
        if args.command == 'usage':
            export = BillingExport(client=bigquery.Client(project=GCP_PROJECT_PLATFORM))
            run = clean_label(args.run)
            usages = export.run_usage(run=run, since=args.since)
            window = usage_window(usages)
            if window is not None and not args.json:
                coverage = export.window_coverage(run=run, window=window, since=args.since)
        elif args.command == 'history':
            key = 'run'
            export = BillingExport(client=bigquery.Client(project=GCP_PROJECT_PLATFORM))
            usages = export.step_history(step=clean_label(args.step), since=args.since)
        elif args.command == 'compute':
            run = clean_label(args.run)
            # `compute` crosses three separate Google APIs (BigQuery, Dataproc, GCS), each of
            # which raises its own `GoogleAPICallError` subclass. A single `except
            # GoogleAPICallError` around the whole branch (as this used to be) cannot tell
            # them apart, so a Dataproc `PermissionDenied` or a GCS read failure would print
            # as "billing export query failed" -- true of none of the three -- and send
            # whoever reads it hunting in the wrong system. Each stage below is wrapped
            # separately and re-raised as a `RuntimeError` carrying which API actually failed;
            # `main`'s `except RuntimeError` below renders that message as-is, unprefixed.
            try:
                export = BillingExport(client=bigquery.Client(project=GCP_PROJECT_PLATFORM))
                usages = export.run_usage(run=run, since=args.since)
                window = usage_window(usages)
                if window is not None and not args.json:
                    coverage = export.window_coverage(run=run, window=window, since=args.since)
            except GoogleAPICallError as exc:
                raise RuntimeError(f'billing export query failed: {" ".join(str(exc).split())}') from exc
            try:
                dataproc_client = dataproc_v1.JobControllerClient(
                    client_options={'api_endpoint': f'{GCP_REGION}-dataproc.googleapis.com:443'}
                )
                executions = job_executions(dataproc_client, project=GCP_PROJECT_PLATFORM, region=GCP_REGION, run=run)
            except GoogleAPICallError as exc:
                raise RuntimeError(f'Dataproc job query failed: {" ".join(str(exc).split())}') from exc
            try:
                bucket = storage.Client().bucket(GCS_PIPELINE_RUNS_BUCKET)
                # Unlike `run`/`step` above, the journal is keyed on the raw Airflow `dag_run_id`,
                # never the cleaned billing label -- see journal.py's module docstring, and
                # `snapshot`/`observe` below, which key it the same way. F4: `--run` accepts
                # either form (its own help text points at the cleaned one, which `usage` also
                # prints), but only the raw form ever finds a real journal here -- see
                # `_journal_read_block`, which is what surfaces this prefix and its event count
                # to the reader instead of a silent, permanently-empty read.
                journal_prefix = f'_agent/unified_pipeline/{args.run}/journal'
                journal = Journal(bucket=bucket, prefix=journal_prefix)
                journal_events = journal.read()
            except GoogleAPICallError as exc:
                raise RuntimeError(f'GCS journal read failed: {" ".join(str(exc).split())}') from exc
            steps = compute_report(usages, executions, journal_events)
            clusters = cluster_compute_report(usages, executions)
            unresolved_jobs = unresolved_job_count(executions)
            if args.json:
                payload = {
                    'steps': [s.model_dump(mode='json') for s in steps],
                    'unresolved_jobs': unresolved_jobs,
                    'clusters': [c.model_dump(mode='json') for c in clusters],
                }
                sys.stdout.write(json.dumps(payload, indent=2) + '\n')
            else:
                report = render_compute(
                    steps,
                    unresolved_jobs,
                    clusters=clusters,
                    journal_prefix=journal_prefix,
                    journal_event_count=len(journal_events),
                )
                report += '\n\n' + render_coverage(window, coverage)
                sys.stdout.write(report + '\n')
            return 0
        elif args.command == 'snapshot':
            username, password = _airflow_credentials()
            client = AirflowClient(
                session=requests.Session(), base_url=AIRFLOW_BASE_URL, username=username, password=password
            )
            bucket = storage.Client().bucket(args.journal_bucket)
            journal = Journal(bucket=bucket, prefix=f'_agent/{args.dag}/{args.run}/journal')
            snapshot = take_snapshot(client, journal, args.dag, args.run, datetime.now(tz=UTC))
            if args.json:
                sys.stdout.write(snapshot.model_dump_json(indent=2) + '\n')
            else:
                sys.stdout.write(render_snapshot(snapshot) + '\n')
            return 0
        elif args.command == 'diff':
            storage_client = storage.Client()
            run_bucket = storage_client.bucket(args.run_bucket)
            reference_bucket = storage_client.bucket(args.reference_bucket)
            run_read_footer = footer_reader(args.run_bucket) if args.rows else None
            reference_read_footer = footer_reader(args.reference_bucket) if args.rows else None
            diffs, skipped = collect_diffs(
                run_bucket,
                args.run,
                reference_bucket,
                args.reference,
                unified_pipeline_steps(),
                stage_configs(),
                run_read_footer,
                reference_read_footer,
            )
            if args.json:
                payload = {
                    'diffs': [d.model_dump(mode='json') for d in diffs],
                    'skipped': skipped.model_dump(mode='json'),
                }
                sys.stdout.write(json.dumps(payload, indent=2) + '\n')
            else:
                sys.stdout.write(render_diff(diffs, skipped, args.threshold, rows_skipped=not args.rows) + '\n')
            return 0
        elif args.command == 'observe':
            now = datetime.now(tz=UTC)
            username, password = _airflow_credentials()
            client = AirflowClient(
                session=requests.Session(), base_url=AIRFLOW_BASE_URL, username=username, password=password
            )
            # `active_dag_run` alone would only ever find a run that is still running —
            # a run that finished between two wakeups has stopped being "active" by the
            # time this runs, so it falls out of that discovery forever, and the
            # terminal-state handling below (a run_finished comment, the dataset diff)
            # would never fire in production. `most_recent_dag_run` is the fallback for
            # exactly that: "what did we most recently start watching", in any state.
            run = client.active_dag_run(args.dag) or client.most_recent_dag_run(args.dag)
            if run is None:
                sys.stdout.write(f'no run of {args.dag} has ever started\n')
                return 0

            bucket = storage.Client().bucket(args.run_bucket)
            journal = Journal(bucket=bucket, prefix=f'_agent/{args.dag}/{run.dag_run_id}/journal')
            snapshot = take_snapshot(client, journal, args.dag, run.dag_run_id, now)
            observation = observe(snapshot, journal.read())
            events = _observation_events(observation, now)
            events.append(
                JournalEvent(event_type=_OBSERVATION_STARTED_EVENT, at=now, payload={'run_name': run_name()})
            )
            # Gated on the run not being terminal, unlike the observation-started marker and
            # the diff-completed marker above, which are gated on their own idempotency keys
            # instead. A finished run is rediscovered by `most_recent_dag_run` on every wakeup
            # forever (see the module docstring), so an unconditional heartbeat would write one
            # object per wakeup into a journal that has nothing left to say — 144 objects a day,
            # forever, for a pipeline that finished once. `run_stalled` already returns before
            # reading `events` at all once `run_state != 'running'`, so a post-terminal heartbeat
            # buys no liveness signal either: nothing downstream ever counts it.
            if snapshot.run_state not in _OBSERVE_TERMINAL_RUN_STATES:
                events.append(heartbeat_event(now))

            diffs = None
            if (
                snapshot.run_state in _OBSERVE_TERMINAL_RUN_STATES
                and args.run
                and args.reference
                and not journal.has(_DATASET_DIFF_COMPLETED_EVENT)
            ):
                storage_client = storage.Client()
                run_bucket = storage_client.bucket(args.run_bucket)
                reference_bucket = storage_client.bucket(args.reference_bucket)
                run_read_footer = footer_reader(args.run_bucket) if args.rows else None
                reference_read_footer = footer_reader(args.reference_bucket) if args.rows else None
                diffs, _skipped = collect_diffs(
                    run_bucket,
                    args.run,
                    reference_bucket,
                    args.reference,
                    unified_pipeline_steps(),
                    stage_configs(),
                    run_read_footer,
                    reference_read_footer,
                )
                events.append(JournalEvent(event_type=_DATASET_DIFF_COMPLETED_EVENT, at=now))

            body = render_comment(observation, snapshot, diffs=diffs, diff_threshold=args.threshold)

            if args.dry_run:
                sys.stdout.write((body if body is not None else '(nothing new to report)') + '\n')
                return 0

            # Post BEFORE journalling, deliberately. `github_app.comment` raises RuntimeError
            # on any failure (a 5xx, a network blip, a bad Secret Manager read), which — since
            # nothing here catches it — unwinds straight out of this `elif` to `main`'s own
            # `except RuntimeError` below, skipping the journal-append loop entirely. That
            # means a failed post leaves these events unjournalled, so the next wakeup's
            # `observe()` sees them as new again and retries the comment. The alternative
            # (journal first) is the one bug this ordering exists to prevent: a transient
            # failure would journal the events as reported anyway, and the next wakeup's
            # idempotency check would then filter them out forever — the failure or stall
            # would never reach the issue at all, silently, with no second chance. The cost
            # of getting this right is a possible duplicate comment, on the rarer path where
            # the post succeeds and the following `journal.append` itself then fails or the
            # process dies before it runs — for a monitoring tool, duplicate beats silent, so
            # that cost is accepted. Do not reorder this back to journal-then-post.
            if body is not None:
                private_key = read_app_key(
                    secretmanager.SecretManagerServiceClient(), GCP_PROJECT_PLATFORM, _GITHUB_APP_KEY_NAME
                )
                github_app = GitHubApp(
                    session=requests.Session(),
                    app_id=_GITHUB_APP_ID,
                    private_key=private_key,
                    installation_id=_GITHUB_INSTALLATION_ID,
                    repo=_GITHUB_REPO,
                )
                github_app.comment(args.issue, body)

            # Unconditional on `body`, not nested under the `if` above. Unlike before F3/F7,
            # `body is None` no longer implies `events` is empty: `_OBSERVATION_STARTED_EVENT`
            # is always in `events`, and once journalled once for this run `journal.append`'s
            # own idempotency (via `has`) makes writing it again a no-op — so this loop still
            # has to run on the "nothing new to report" wakeup that follows, not only on one
            # with a comment to post. More generally: `_observation_events` journals exactly
            # `observation`'s failed/stalled/completed/run_finished, and `render_comment`
            # returns `None` only when `observation.is_empty` and no diff ran — the same
            # events, empty, plus whatever this call always appends. An `Observation` that
            # somehow carried an event `render_comment` did not turn into a section must still
            # be journalled, not silently dropped because nothing was posted for it.
            for event in events:
                journal.append(event)
            return 0
        else:
            raise ValueError(f'unknown subcommand: {args.command}')
    except DefaultCredentialsError as exc:
        sys.stderr.write(
            f'no Google Cloud credentials found: {" ".join(str(exc).split())}\n'
            'run `gcloud auth application-default login` first\n'
        )
        return 1
    except GoogleAPICallError as exc:
        sys.stderr.write(f'billing export query failed: {" ".join(str(exc).split())}\n')
        return 1
    except requests.RequestException as exc:
        sys.stderr.write(f'could not reach Airflow: {" ".join(str(exc).split())}\n')
        return 1
    except RuntimeError as exc:
        sys.stderr.write(f'{" ".join(str(exc).split())}\n')
        return 1
    except ValueError as exc:
        sys.stderr.write(f'{" ".join(str(exc).split())}\n')
        return 1

    if args.json:
        sys.stdout.write(json.dumps([u.model_dump(mode='json') for u in usages], indent=2) + '\n')
    else:
        report = render_table(usages, key)
        if args.command == 'usage':
            report += '\n\n' + render_coverage(window, coverage)
        sys.stdout.write(report + '\n')
    return 0
