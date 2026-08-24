"""Command line entry point for the pipeline supervisor.

Exposes billed cost for a run and for one step's history, both read from the GCP
billing export, and a read-only `snapshot` of a run's state read from Airflow and
the run's journal.

`--run` and `--step` are matched against GCP labels, which are normalised, so both
are passed through `clean_label` first. A run ID copied straight out of the Airflow
UI therefore works.

Every figure the `usage`/`history` commands report covers only the resources the
pipeline labelled per step, which is not everything it spends. The `usage` report
says so with a coverage line rather than leaving its total to be read as the cost of
the run. Cost only: the export is hourly-bucketed and cannot support a per-step
duration, which comes from Airflow task instances instead — see `snapshot`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import requests
import yaml
from google.api_core.exceptions import GoogleAPICallError
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery, storage

from orchestration.supervisor.airflow import AirflowClient
from orchestration.supervisor.diff import DatasetDiff, is_material
from orchestration.supervisor.gcs import Footer, Skipped, collect_diffs, footer_reader
from orchestration.supervisor.journal import Journal
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
            'coverage: unknown. This run has billed nothing yet, so there is no window\n'
            'over which to compare its labelled cost against total pipeline spend.'
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


_UNCOUNTABLE = 'n/a'
"""Shown for a row count the dataset's format has no footer to have supplied, as
distinct from `_MISSING`, which marks a side that is absent altogether."""

_REPO = Path(__file__).resolve().parents[4]
"""Repo root, four levels above this file, holding `pis/`, `pts/` and `orchestration/`."""

_UNIFIED_PIPELINE_YAML = _REPO / 'orchestration/src/orchestration/dags/config/unified_pipeline.yaml'


def _stage_configs() -> dict[str, Any]:
    """Load `pis` and `pts`'s own configs, the only two `collect_diffs` needs.

    Gentropy's steps are deliberately not read here: their destinations live in
    `dags/config/gentropy.yaml`, and `collect_diffs` records them under
    `stages_without_config` rather than needing a third config it cannot parse the
    same way.

    Returns:
        Each stage's parsed config, keyed by stage name.
    """
    return {stage: yaml.safe_load((_REPO / stage / 'config.yaml').read_text()) for stage in ('pis', 'pts')}


def _unified_pipeline_steps() -> list[str]:
    """Load the step list `collect_diffs` walks, from `unified_pipeline.yaml`.

    Returns:
        Every step name declared under `steps:`, in file order.
    """
    up = yaml.safe_load(_UNIFIED_PIPELINE_YAML.read_text())
    return list(up['steps'])


def _dispatching_footer_reader(
    run_bucket_name: str, run_prefix: str, reference_bucket_name: str, reference_prefix: str
) -> Callable[[str], Footer]:
    """Build one footer reader spanning both sides of a diff, which may be different buckets.

    `collect_diffs` shares a single `read_footer` between both sides of the walk, but
    `gcs.footer_reader` is pinned to one bucket name — the path it builds for `pyarrow`
    is `f'{bucket_name}/{object_name}'`, which is wrong for an object read from the
    other bucket. Routing on the object name's prefix is safe because `run_prefix` and
    `reference_prefix` are normally a run name and a release name, always distinct
    strings, so a blob from one side never starts with the other side's root — a
    prefix that is a strict prefix of the other (`26.03` vs `26.03-ppp`) is still safe,
    since the trailing `/` this appends makes `26.03-ppp/...` fail to start with
    `26.03/`. What is not safe is the two normalising to the *same* root, which
    `--run` and `--reference` naming the same release would do; that is checked for
    explicitly rather than left to silently route every object to the run side.

    Args:
        run_bucket_name: The bucket holding the run.
        run_prefix: The run's root prefix within it.
        reference_bucket_name: The bucket holding the reference release.
        reference_prefix: The release's root prefix within it.

    Returns:
        A callable reading a parquet footer from whichever bucket the object belongs to.

    Raises:
        ValueError: If `run_prefix` and `reference_prefix` normalise to the same root.
            Routing cannot tell the two sides apart in that case, and reading every
            object through the run side's reader — silently comparing a release
            against itself, or 404ing on every reference object — is worse than
            refusing outright.
    """
    run_root = run_prefix.rstrip('/') + '/'
    reference_root = reference_prefix.rstrip('/') + '/'
    if run_root == reference_root:
        raise ValueError(
            f'--run {run_prefix!r} and --reference {reference_prefix!r} are the same root, so a diff '
            'cannot tell which bucket an object belongs to'
        )
    read_run = footer_reader(run_bucket_name)
    read_reference = footer_reader(reference_bucket_name)

    def read(name: str) -> Footer:
        return read_run(name) if name.startswith(run_root) else read_reference(name)

    return read


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
        rows_skipped: True when `--no-rows` skipped footer reads for this run. Without
            this, every row count prints as `-` (`_count`'s "absent" symbol, since
            `countable` is still True for parquet), which reads as every dataset's
            counterpart being missing rather than as rows simply not having been read.

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
            'Row counts were not read (--no-rows). Sizes, file counts and presence are compared; '
            'a row count of "-" above means not read, not absent.'
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
        '--no-rows',
        action='store_true',
        help='skip row counts (sizes, file counts and presence only); seconds instead of minutes',
    )
    diff.add_argument('--json', action='store_true', help='emit JSON instead of text')

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
            read_footer = (
                None
                if args.no_rows
                else _dispatching_footer_reader(args.run_bucket, args.run, args.reference_bucket, args.reference)
            )
            diffs, skipped = collect_diffs(
                run_bucket,
                args.run,
                reference_bucket,
                args.reference,
                _unified_pipeline_steps(),
                _stage_configs(),
                read_footer,
            )
            if args.json:
                payload = {
                    'diffs': [d.model_dump(mode='json') for d in diffs],
                    'skipped': skipped.model_dump(mode='json'),
                }
                sys.stdout.write(json.dumps(payload, indent=2) + '\n')
            else:
                sys.stdout.write(render_diff(diffs, skipped, args.threshold, rows_skipped=args.no_rows) + '\n')
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
