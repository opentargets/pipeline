"""Command line entry point for the pipeline supervisor.

Currently exposes billed cost for a run and for one step's history. Both read the
GCP billing export, which is hourly-bucketed, so the reported span is an upper
bound on wall-clock time rather than a measurement of it.

`--run` and `--step` are matched against GCP labels, which are normalised, so both
are passed through `clean_label` first. A run ID copied straight out of the Airflow
UI therefore works.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from typing import Literal

from google.api_core.exceptions import GoogleAPICallError
from google.cloud import bigquery

from orchestration.supervisor.usage import BillingExport, StepUsage, total_cost
from orchestration.utils.common import BILLING_EXPORT_START, GCP_PROJECT_PLATFORM, clean_label

RowKey = Literal['run', 'step']
"""Which label identifies a row of the rendered table."""

_MIN_KEY_WIDTH = 20
"""Floor for the identity column, so a short-labelled table is not cramped."""

_EMPTY = """No billed usage found. It may not have billed yet, or the label may not match.
Labels are lowercased with everything outside [a-z0-9-_] replaced by '-', and
--run/--step are normalised the same way, so check the normalised form: an Airflow
run ID like manual__2026-07-21T15:07:47.545737+00:00 is stored as the label
manual__2026-07-21t15-07-47-545737-00-00."""


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

    header = f'{key:<{width}} {"tool":<9} {"span (h)":>9} {"net cost":>10}'
    lines = [header, '-' * len(header)]
    lines.extend(
        f'{k:<{width}} {u.tool:<9} {u.span_hours:>9.2f} {u.net_cost:>10.2f}'
        for k, u in zip(keys, usages, strict=True)
    )
    lines.append('-' * len(header))

    totals = _totals_by_currency(usages)
    lines.extend(
        f'{"total":<{width}} {"":<9} {"":>9} {amount:>10.2f} {currency}'
        for currency, amount in totals.items()
    )
    lines.append('')
    if len(totals) > 1:
        lines.append('these rows are billed in more than one currency, so they are totalled')
        lines.append('separately. The per-currency totals must not be added together.')
    lines.append('span is billed hours, quantised up to the hour by the export. It is an')
    lines.append('upper bound on wall-clock time, not a duration.')
    return '\n'.join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(prog='pipeline-supervisor')
    sub = parser.add_subparsers(dest='command', required=True)

    usage = sub.add_parser('usage', help='billed usage for every step in one run')
    usage.add_argument('--run', required=True, help='the run label to report on')
    usage.add_argument('--since', type=date.fromisoformat, default=BILLING_EXPORT_START)
    usage.add_argument('--json', action='store_true', help='emit JSON instead of a table')

    history = sub.add_parser('history', help="one step's billed usage across runs")
    history.add_argument('--step', required=True, help='the step label to report on')
    history.add_argument('--since', type=date.fromisoformat, default=BILLING_EXPORT_START)
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
    export = BillingExport(client=bigquery.Client(project=GCP_PROJECT_PLATFORM))

    key: RowKey = 'step'
    try:
        if args.command == 'usage':
            usages = export.run_usage(run=clean_label(args.run), since=args.since)
        elif args.command == 'history':
            key = 'run'
            usages = export.step_history(step=clean_label(args.step), since=args.since)
        else:
            raise ValueError(f'unknown subcommand: {args.command}')
    except GoogleAPICallError as exc:
        sys.stderr.write(f'billing export query failed: {" ".join(str(exc).split())}\n')
        return 1

    if args.json:
        sys.stdout.write(json.dumps([u.model_dump(mode='json') for u in usages], indent=2) + '\n')
    else:
        sys.stdout.write(render_table(usages, key) + '\n')
    return 0
