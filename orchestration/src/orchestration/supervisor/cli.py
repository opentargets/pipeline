"""Command line entry point for the pipeline supervisor.

Currently exposes billed cost for a run and for one step's history. Both read the
GCP billing export, which is hourly-bucketed, so the reported span is an upper
bound on wall-clock time rather than a measurement of it.

`--run` and `--step` are matched against GCP labels, which are normalised, so both
are passed through `clean_label` first. A run ID copied straight out of the Airflow
UI therefore works.

Every figure here covers only the resources the pipeline labelled per step, which is
not everything it spends. The `usage` report says so with a coverage line rather than
leaving its total to be read as the cost of the run.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from typing import Literal

from google.api_core.exceptions import GoogleAPICallError
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery

from orchestration.supervisor.usage import (
    BillingExport,
    StepUsage,
    WindowCoverage,
    total_cost,
    usage_window,
)
from orchestration.utils.common import BILLING_EXPORT_START, GCP_PROJECT_PLATFORM, clean_label

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

_SPAN_FOOTER = [
    "span is the envelope of a step's billed rows: from the start of its first billed",
    'hour to the end of its last, gaps included. It is not billed hours and not a',
    'duration. One step labels several billed resources at once (a GCE step labels its',
    'instance and both disks, a Dataproc step every node), so the hours actually billed',
    'are a multiple of this.',
]

_CURRENCY_FOOTER = [
    'these rows are billed in more than one currency, so they are totalled',
    'separately. The per-currency totals must not be added together.',
]

_PRODUCT_FOOTER = [
    'product separates the platform pipeline from the partner preview one, which runs',
    'the same DAG and can share a run label. Their rows are never added together.',
]

_SHARED_FOOTER = [
    'shared marks a step whose Dataproc cluster also served other steps of the same run.',
    'A cluster keeps the labels of the step that created it, so a marked row is charged',
    "the whole cluster's hours, including the other steps' — it is not that step's own",
    'cost. Splitting it up needs the Dataproc Jobs API and is not attempted here.',
]

_COVERAGE_FOOTER = [
    'the remainder is real pipeline spend that no step above accounts for: Google Batch',
    'jobs carry no step labels, and some Dataproc disk and licensing rows fall outside',
    'the labels of the step that caused them.',
]


def _totals_by_currency(usages: list[StepUsage]) -> dict[str, float]:
    """Total net cost per currency.

    Args:
        usages: The usages to total.

    Returns:
        Net cost keyed by currency code. Costs in different currencies are never
        added together, because their sum is not a quantity of money.
    """
    return {
        currency: total_cost(u for u in usages if u.currency == currency)
        for currency in sorted({u.currency for u in usages})
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
    blanks = ''.join(f' {"":<{widths[c]}}' for c in columns)

    header = f'{key:<{width}} {"tool":<9}{extra_header} {"span (h)":>9} {"net cost":>10}'
    lines = [header, '-' * len(header)]
    lines.extend(
        f'{k:<{width}} {u.tool:<9}'
        + ''.join(f' {_cell(u, c):<{widths[c]}}' for c in columns)
        + f' {u.span_hours:>9.2f} {u.net_cost:>10.2f}'
        for k, u in zip(keys, usages, strict=True)
    )
    lines.append('-' * len(header))

    totals = _totals_by_currency(usages)
    lines.extend(
        f'{"total":<{width}} {"":<9}{blanks} {"":>9} {amount:>10.2f} {currency}'
        for currency, amount in totals.items()
    )
    blocks = []
    if len(totals) > 1:
        blocks.append(_CURRENCY_FOOTER)
    if 'product' in columns:
        blocks.append(_PRODUCT_FOOTER)
    if 'shared' in columns:
        blocks.append(_SHARED_FOOTER)
    blocks.append(_SPAN_FOOTER)
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
    lines.append('')
    lines.extend(_COVERAGE_FOOTER)
    return '\n'.join(lines)


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
        export = BillingExport(client=bigquery.Client(project=GCP_PROJECT_PLATFORM))
        if args.command == 'usage':
            usages = export.run_usage(run=clean_label(args.run), since=args.since)
            window = usage_window(usages)
            if window is not None and not args.json:
                coverage = export.window_coverage(window=window, since=args.since)
        elif args.command == 'history':
            key = 'run'
            usages = export.step_history(step=clean_label(args.step), since=args.since)
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

    if args.json:
        sys.stdout.write(json.dumps([u.model_dump(mode='json') for u in usages], indent=2) + '\n')
    else:
        report = render_table(usages, key)
        if args.command == 'usage':
            report += '\n\n' + render_coverage(window, coverage)
        sys.stdout.write(report + '\n')
    return 0
