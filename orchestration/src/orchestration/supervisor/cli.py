"""Command line entry point for the pipeline supervisor.

Currently exposes billed cost for a run and for one step's history. Both read the
GCP billing export, which is hourly-bucketed, so the reported span is an upper
bound on wall-clock time rather than a measurement of it.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from google.cloud import bigquery

from orchestration.supervisor.usage import BillingExport, StepUsage, total_cost

EXPORT_START = date(2026, 5, 1)
"""Oldest partition in the billing export. Scanning earlier is wasted work."""


def render_table(usages: list[StepUsage]) -> str:
    """Render usages as a fixed-width table.

    Args:
        usages: The usages to render.

    Returns:
        The rendered table, or a message when there is nothing to show.
    """
    if not usages:
        return 'No billed usage found. The run may not have billed yet, or the label may be wrong.'

    header = f'{"step":<45} {"tool":<9} {"span (h)":>9} {"net cost":>10}'
    lines = [header, '-' * len(header)]
    lines.extend(
        f'{u.step:<45} {u.tool:<9} {u.span_hours:>9.2f} {u.net_cost:>10.2f}' for u in usages
    )
    lines.append('-' * len(header))
    currency = usages[0].currency
    lines.append(f'{"total":<45} {"":<9} {"":>9} {total_cost(usages):>10.2f} {currency}')
    lines.append('')
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
    usage.add_argument('--since', type=date.fromisoformat, default=EXPORT_START)
    usage.add_argument('--json', action='store_true', help='emit JSON instead of a table')

    history = sub.add_parser('history', help="one step's billed usage across runs")
    history.add_argument('--step', required=True, help='the step label to report on')
    history.add_argument('--since', type=date.fromisoformat, default=EXPORT_START)
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
    export = BillingExport(client=bigquery.Client())

    if args.command == 'usage':
        usages = export.run_usage(run=args.run, since=args.since)
    else:
        usages = export.step_history(step=args.step, since=args.since)

    if args.json:
        sys.stdout.write(json.dumps([u.model_dump(mode='json') for u in usages], indent=2) + '\n')
    else:
        sys.stdout.write(render_table(usages) + '\n')
    return 0
