"""Tests for the supervisor CLI."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from orchestration.supervisor.cli import build_parser, render_table
from orchestration.supervisor.usage import StepUsage


def _usage(step: str = 'pts_target', net_cost: float = 1.5) -> StepUsage:
    return StepUsage(
        run='r',
        step=step,
        tool='pts',
        started=datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
        ended=datetime(2026, 7, 21, 16, 0, tzinfo=UTC),
        span_hours=2.0,
        net_cost=net_cost,
        currency='GBP',
    )


class TestRenderTable:
    def test_includes_each_step_and_the_total(self) -> None:
        out = render_table([_usage('pts_target', 1.5), _usage('pis_disease', 2.5)])
        assert 'pts_target' in out
        assert 'pis_disease' in out
        assert '4.00' in out

    def test_column_header_says_span_not_duration(self) -> None:
        """The export is hourly-bucketed, so heading this column 'duration' would mislead.

        The explanatory footer may still use the word.
        """
        header = render_table([_usage()]).splitlines()[0]
        assert 'span' in header.lower()
        assert 'duration' not in header.lower()

    def test_empty_says_so_rather_than_printing_an_empty_table(self) -> None:
        assert 'no billed usage' in render_table([]).lower()


class TestParser:
    def test_run_is_required_for_usage(self) -> None:
        parser = build_parser()
        args = parser.parse_args(['usage', '--run', 'my-run'])
        assert args.run == 'my-run'

    def test_json_flag_defaults_off(self) -> None:
        parser = build_parser()
        assert parser.parse_args(['usage', '--run', 'r']).json is False
        assert parser.parse_args(['usage', '--run', 'r', '--json']).json is True

    def test_since_defaults_to_export_start(self) -> None:
        parser = build_parser()
        assert parser.parse_args(['usage', '--run', 'r']).since.isoformat() == '2026-05-01'

    def test_history_subcommand_takes_a_step(self) -> None:
        parser = build_parser()
        args = parser.parse_args(['history', '--step', 'pts_target'])
        assert args.step == 'pts_target'


class TestJsonOutput:
    def test_models_serialise(self) -> None:
        payload = json.loads(_usage().model_dump_json())
        assert payload['step'] == 'pts_target'
        assert payload['net_cost'] == 1.5
