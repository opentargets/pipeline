"""Tests for the supervisor CLI."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from orchestration.supervisor import cli
from orchestration.supervisor.cli import build_parser, main, render_table
from orchestration.supervisor.usage import StepUsage
from orchestration.utils.common import GCP_PROJECT_PLATFORM, clean_label

RAW_RUN_ID = 'manual__2026-07-21T15:07:47.545737+00:00'
CLEAN_RUN_ID = 'manual__2026-07-21t15-07-47-545737-00-00'


def _usage(
    step: str = 'pts_target',
    net_cost: float = 1.5,
    run: str = 'r',
    currency: str = 'GBP',
) -> StepUsage:
    return StepUsage(
        run=run,
        step=step,
        tool='pts',
        started=datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
        ended=datetime(2026, 7, 21, 16, 0, tzinfo=UTC),
        span_hours=2.0,
        net_cost=net_cost,
        currency=currency,
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

    def test_empty_message_explains_label_normalisation(self) -> None:
        """The likeliest reason for an empty result is an unnormalised run ID.

        Telling the user only that the run 'may not have billed yet' sends them
        looking for a billing delay that is not there.
        """
        out = render_table([])
        assert CLEAN_RUN_ID in out

    def test_history_rows_are_distinguishable_by_run(self) -> None:
        """Every row of a history shares one step, so the run has to identify the row."""
        out = render_table(
            [_usage(run='first-run', net_cost=1.5), _usage(run='second-run', net_cost=2.5)],
            key='run',
        )
        assert 'first-run' in out
        assert 'second-run' in out
        assert out.splitlines()[0].startswith('run')

    def test_key_column_width_follows_the_data(self) -> None:
        """A step name longer than the old hardcoded 45 must not go ragged."""
        long_step = 'pts_evidence_postprocess_clinical_precedence_and_then_some'
        out = render_table([_usage(step=long_step)])
        header, rule, row = out.splitlines()[:3]
        assert long_step in row
        assert len(row) == len(header) == len(rule)

    def test_key_column_has_a_floor(self) -> None:
        out = render_table([_usage(step='a')])
        assert out.splitlines()[0].startswith('step' + ' ' * 17)

    def test_mixed_currencies_are_totalled_separately(self) -> None:
        """Adding GBP to USD produces a number that is not an amount of money.

        The export is uniformly GBP today, so this is a guard, not a description.
        """
        out = render_table([_usage(net_cost=1.5), _usage(net_cost=2.5, currency='USD')])
        totals = [line for line in out.splitlines() if line.startswith('total')]
        assert '4.00' not in out
        assert [line.split()[-2:] for line in totals] == [['1.50', 'GBP'], ['2.50', 'USD']]

    def test_single_currency_still_has_one_total(self) -> None:
        out = render_table([_usage(net_cost=1.5), _usage(net_cost=2.5)])
        totals = [line for line in out.splitlines() if line.startswith('total')]
        assert len(totals) == 1
        assert totals[0].split()[-2:] == ['4.00', 'GBP']


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


class TestLabelNormalisation:
    def test_airflow_run_id_normalises_to_the_stored_label(self) -> None:
        """This is the form the CLI has to query for, and the form users never type."""
        assert clean_label(RAW_RUN_ID) == CLEAN_RUN_ID


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch out `bigquery.Client`, exposing the constructor as `client.constructor`."""
    fake = MagicMock()
    fake.query.return_value.result.return_value = []
    fake.constructor = MagicMock(return_value=fake)
    monkeypatch.setattr(cli.bigquery, 'Client', fake.constructor)
    return fake


def _bound(client: MagicMock) -> dict[str, Any]:
    job_config = client.query.call_args.kwargs['job_config']
    return {p.name: p.value for p in job_config.query_parameters}


class TestMain:
    def test_run_argument_is_normalised_before_querying(self, client: MagicMock) -> None:
        """A user copies the run ID out of the Airflow UI, unsanitised."""
        assert main(['usage', '--run', RAW_RUN_ID]) == 0
        assert _bound(client)['run'] == CLEAN_RUN_ID

    def test_step_argument_is_normalised_before_querying(self, client: MagicMock) -> None:
        assert main(['history', '--step', 'PTS_Target']) == 0
        assert _bound(client)['step'] == 'pts_target'

    def test_client_is_pinned_to_the_platform_project(self, client: MagicMock) -> None:
        """Unpinned, this picks up whatever project the machine happens to default to."""
        main(['usage', '--run', 'r'])
        assert client.constructor.call_args.kwargs['project'] == GCP_PROJECT_PLATFORM

    def test_history_renders_with_the_run_column(
        self, client: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        client.query.return_value.result.return_value = [_usage(run='older-run')]
        assert main(['history', '--step', 'pts_target']) == 0
        assert capsys.readouterr().out.splitlines()[0].startswith('run')

    def test_api_error_exits_non_zero_with_a_message(
        self, client: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A missing table or a denied project must not surface as a traceback."""
        client.query.side_effect = cli.GoogleAPICallError('403 Access Denied\non the export')
        assert main(['usage', '--run', 'r']) == 1
        err = capsys.readouterr().err
        assert 'billing export query failed' in err
        assert err.count('\n') == 1


class TestJsonOutput:
    def test_models_serialise(self) -> None:
        payload = json.loads(_usage().model_dump_json())
        assert payload['step'] == 'pts_target'
        assert payload['net_cost'] == 1.5
