"""Tests for the supervisor CLI."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from orchestration.supervisor import cli
from orchestration.supervisor.cli import (
    build_parser,
    main,
    optional_columns,
    render_coverage,
    render_table,
    totals_by_group,
)
from orchestration.supervisor.usage import StepUsage, WindowCoverage
from orchestration.utils.common import GCP_PROJECT_PLATFORM, clean_label

RAW_RUN_ID = 'manual__2026-07-21T15:07:47.545737+00:00'
CLEAN_RUN_ID = 'manual__2026-07-21t15-07-47-545737-00-00'
WINDOW = (datetime(2026, 7, 21, 14, 0, tzinfo=UTC), datetime(2026, 7, 22, 2, 0, tzinfo=UTC))


def _usage(
    step: str = 'pts_target',
    net_cost: float = 1.5,
    run: str = 'r',
    currency: str = 'GBP',
    product: str | None = 'platform',
    shared: bool = False,
) -> StepUsage:
    return StepUsage(
        run=run,
        step=step,
        tool='pts',
        product=product,
        started=datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
        ended=datetime(2026, 7, 21, 16, 0, tzinfo=UTC),
        net_cost=net_cost,
        currency=currency,
        shared_cluster=shared,
    )


class TestRenderTable:
    def test_includes_each_step_and_the_total(self) -> None:
        out = render_table([_usage('pts_target', 1.5), _usage('pis_disease', 2.5)])
        assert 'pts_target' in out
        assert 'pis_disease' in out
        assert '4.00' in out

    def test_no_time_column_is_offered(self) -> None:
        """The export cannot support a per-step duration, so this module reports none.

        The old span column was a whole number of hours in every row of the export, and
        an envelope including gaps, so it read 39.00 for a step that ran for an hour.
        Duration comes from Airflow task instances in a later phase.
        """
        out = render_table([_usage()])
        assert 'span' not in out.lower()
        assert 'duration' not in out.lower()
        assert 'hour' not in out.lower()

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


class TestTotalsNeverCrossAnIncomparableColumn:
    """The rows were split by product and the footer promised they are never added.

    A single total underneath them broke that promise in the one place a reader looks.
    """

    def test_mixed_products_are_totalled_separately(self) -> None:
        out = render_table([_usage(net_cost=100.0), _usage(net_cost=200.0, product='ppp')])
        totals = [line for line in out.splitlines() if line.startswith('total')]
        assert '300.00' not in out
        assert [line.split()[1:] for line in totals] == [
            ['platform', '100.00', 'GBP'],
            ['ppp', '200.00', 'GBP'],
        ]

    def test_the_total_names_the_product_it_belongs_to(self) -> None:
        """An unlabelled total line under a split table is not attributable to either."""
        out = render_table([_usage(net_cost=100.0), _usage(net_cost=200.0, product='ppp')])
        totals = [line for line in out.splitlines() if line.startswith('total')]
        assert all('platform' in totals[0] for _ in totals[:1])
        assert 'ppp' in totals[1]

    def test_product_and_currency_split_together(self) -> None:
        out = render_table([
            _usage(net_cost=1.0),
            _usage(net_cost=2.0, product='ppp'),
            _usage(net_cost=4.0, currency='USD'),
            _usage(net_cost=8.0, product='ppp', currency='USD'),
        ])
        totals = [line.split()[1:] for line in out.splitlines() if line.startswith('total')]
        assert totals == [
            ['platform', 'GBP', '1.00', 'GBP'],
            ['platform', 'USD', '4.00', 'USD'],
            ['ppp', 'GBP', '2.00', 'GBP'],
            ['ppp', 'USD', '8.00', 'USD'],
        ]

    def test_a_shared_flag_does_not_split_the_total(self) -> None:
        """Shared rows are the same money badly attributed, so they still add up."""
        out = render_table([_usage(net_cost=1.5), _usage(net_cost=2.5, shared=True)])
        totals = [line for line in out.splitlines() if line.startswith('total')]
        assert len(totals) == 1
        assert totals[0].split()[-2:] == ['4.00', 'GBP']

    def test_totals_by_group_keys_on_values_not_on_display(self) -> None:
        """The split must not depend on the rendering rule that decides what is shown."""
        assert totals_by_group([_usage(net_cost=1.0), _usage(net_cost=2.0, product='ppp')]) == {
            ('platform', 'GBP'): 1.0,
            ('ppp', 'GBP'): 2.0,
        }
        assert totals_by_group([_usage(net_cost=1.0), _usage(net_cost=2.0)]) == {
            ('platform', 'GBP'): 3.0
        }


class TestOptionalColumns:
    def test_nothing_extra_when_the_rows_agree(self) -> None:
        """The common case is one product and one currency, and naming them is noise."""
        assert optional_columns([_usage(net_cost=1.5), _usage(net_cost=2.5)]) == []

    def test_product_appears_when_the_rows_disagree(self) -> None:
        assert optional_columns([_usage(), _usage(product='ppp')]) == ['product']

    def test_currency_appears_when_the_rows_disagree(self) -> None:
        assert optional_columns([_usage(), _usage(currency='USD')]) == ['currency']

    def test_a_missing_product_counts_as_a_distinct_value(self) -> None:
        """An unlabelled row next to a labelled one is exactly the ambiguity to surface."""
        assert optional_columns([_usage(), _usage(product=None)]) == ['product']

    def test_shared_appears_as_soon_as_one_row_is_flagged(self) -> None:
        assert optional_columns([_usage(), _usage(shared=True)]) == ['shared']

    def test_shared_still_appears_when_every_row_is_flagged(self) -> None:
        """All rows shared is the worst case, not a uniform one to be hidden as noise."""
        assert optional_columns([_usage(shared=True), _usage(shared=True)]) == ['shared']

    def test_shared_stays_hidden_when_nothing_is_flagged(self) -> None:
        assert optional_columns([_usage(), _usage()]) == []


class TestIncomparableRowsAreLabelled:
    def test_rows_in_different_currencies_carry_their_currency(self) -> None:
        """Bare, 1.50 and 2.50 read as comparable amounts. They are not."""
        out = render_table([_usage(net_cost=1.5), _usage(net_cost=2.5, currency='USD')])
        rows = [line for line in out.splitlines() if line.startswith('pts_target')]
        assert [line.split()[2] for line in rows] == ['GBP', 'USD']

    def test_rows_from_different_products_carry_their_product(self) -> None:
        """PPP and platform runs can share a run label, and their costs are not one cost."""
        out = render_table([_usage(net_cost=1.5), _usage(net_cost=2.5, product='ppp')])
        rows = [line for line in out.splitlines() if line.startswith('pts_target')]
        assert [line.split()[2] for line in rows] == ['platform', 'ppp']
        assert 'never added together' in out

    def test_an_unlabelled_product_renders_as_a_dash(self) -> None:
        out = render_table([_usage(), _usage(product=None)])
        rows = [line for line in out.splitlines() if line.startswith('pts_target')]
        assert [line.split()[2] for line in rows] == ['platform', '-']

    def test_the_table_stays_square_with_extra_columns(self) -> None:
        out = render_table([_usage(product='ppp'), _usage(currency='USD')])
        header, rule, row = out.splitlines()[:3]
        assert len(row) == len(header) == len(rule)

    def test_a_shared_cluster_row_is_marked_and_explained(self) -> None:
        """The number on a shared row is the cluster instance's cost, not the step's."""
        out = render_table([_usage(step='pts_reactome'), _usage(step='pts_ontoma', shared=True)])
        rows = {line.split()[0]: line.split()[2] for line in out.splitlines() if line.startswith('pts_')}
        assert rows == {'pts_reactome': '-', 'pts_ontoma': 'yes'}
        assert 'shared' in out.splitlines()[0]
        assert "rather than the step's own" in out

    def test_the_shared_footer_does_not_claim_this_happens(self) -> None:
        """A footer implying this happens describes billing that does not exist.

        Which invites the reader to distrust correct per-step numbers.
        """
        out = render_table([_usage(shared=True)])
        assert 'has ever been marked' in out
        assert 'not currently manifest' in out
        assert 'name' in out

    def test_an_unshared_table_is_unchanged(self) -> None:
        """The common case must not grow a column of dashes."""
        out = render_table([_usage(net_cost=1.5), _usage(net_cost=2.5)])
        assert 'shared' not in out
        assert out.splitlines()[0] == f'{"step":<20} {"tool":<9} {"net cost":>10}'

    def test_no_currency_column_when_they_all_match(self) -> None:
        out = render_table([_usage(net_cost=1.5), _usage(net_cost=2.5)])
        assert 'GBP' not in out.splitlines()[0]
        rows = [line for line in out.splitlines() if line.startswith('pts_target')]
        assert all('GBP' not in line for line in rows)


class TestRenderCoverage:
    def test_states_the_labelled_share_of_pipeline_spend(self) -> None:
        out = render_coverage(
            WINDOW, [WindowCoverage(currency='GBP', labelled_cost=3170.71, pipeline_cost=4232.82)]
        )
        assert '3170.71 of the 4232.82 GBP' in out
        assert '74.9%' in out
        assert '2026-07-21 14:00 to 2026-07-22 02:00 UTC' in out

    def test_explains_what_the_remainder_is(self) -> None:
        """Otherwise the gap reads as a bug in this tool rather than unlabelled spend."""
        out = render_coverage(
            WINDOW, [WindowCoverage(currency='GBP', labelled_cost=1.0, pipeline_cost=2.0)]
        )
        assert 'Batch' in out
        assert 'Dataproc' in out

    def test_an_unknown_window_says_so_rather_than_claiming_full_coverage(self) -> None:
        """A run with no rows has no window, and 100% would be the most misleading answer."""
        out = render_coverage(None, [])
        assert 'unknown' in out
        assert '%' not in out

    def test_an_empty_window_says_so(self) -> None:
        out = render_coverage(WINDOW, [])
        assert 'no pipeline spend' in out
        assert '%' not in out

    def test_zero_spend_does_not_render_a_percentage(self) -> None:
        """Zero of zero is not full coverage."""
        out = render_coverage(
            WINDOW, [WindowCoverage(currency='GBP', labelled_cost=0.0, pipeline_cost=0.0)]
        )
        assert 'share undefined' in out
        assert '%' not in out

    def test_an_impossible_share_reads_as_broken_not_as_excellent(self) -> None:
        """Above 100% the numerator counted rows the denominator did not."""
        out = render_coverage(
            WINDOW, [WindowCoverage(currency='GBP', labelled_cost=120.0, pipeline_cost=100.0)]
        )
        assert '120.0%' in out
        assert 'broken measurement' in out

    def test_a_normal_share_carries_no_alarm(self) -> None:
        out = render_coverage(
            WINDOW, [WindowCoverage(currency='GBP', labelled_cost=75.0, pipeline_cost=100.0)]
        )
        assert 'broken measurement' not in out

    def test_the_denominator_is_declared_run_agnostic(self) -> None:
        """An overlapping run's spend is in the denominator, and the reader must know."""
        out = render_coverage(
            WINDOW, [WindowCoverage(currency='GBP', labelled_cost=1.0, pipeline_cost=2.0)]
        )
        assert 'including any' in out
        assert 'other run that overlapped it' in out

    def test_each_currency_gets_its_own_line(self) -> None:
        out = render_coverage(
            WINDOW,
            [
                WindowCoverage(currency='GBP', labelled_cost=1.0, pipeline_cost=2.0),
                WindowCoverage(currency='USD', labelled_cost=3.0, pipeline_cost=4.0),
            ],
        )
        assert '1.00 of the 2.00 GBP' in out
        assert '3.00 of the 4.00 USD' in out


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

    @pytest.mark.parametrize('command', ['usage', 'history'])
    def test_since_help_says_it_filters_ingestion_date(
        self, command: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`--since 2026-07-01` reads as 'runs since July' and means 'rows ingested since July'.

        It was also the only flag with no help text at all.
        """
        with pytest.raises(SystemExit):
            build_parser().parse_args([command, '--help'])
        help_text = ' '.join(capsys.readouterr().out.split())
        assert '--since' in help_text
        assert 'ingested' in help_text
        assert 'not the date of the usage' in help_text


class TestLabelNormalisation:
    def test_airflow_run_id_normalises_to_the_stored_label(self) -> None:
        """This is the form the CLI has to query for, and the form users never type."""
        assert clean_label(RAW_RUN_ID) == CLEAN_RUN_ID


class TestSnapshotParser:
    def test_snapshot_takes_a_run(self) -> None:
        args = build_parser().parse_args(['snapshot', '--run', 'r'])
        assert args.run == 'r'

    def test_snapshot_defaults_to_the_unified_pipeline_dag(self) -> None:
        assert build_parser().parse_args(['snapshot', '--run', 'r']).dag == 'unified_pipeline'

    def test_snapshot_supports_json(self) -> None:
        assert build_parser().parse_args(['snapshot', '--run', 'r', '--json']).json is True


class TestSnapshotCommand:
    def test_a_client_error_exits_non_zero_with_a_one_line_message(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A client-level `RuntimeError` must reach the user as a message, not a traceback.

        A 404 from Airflow (e.g. a typo'd run id) surfaces as `RuntimeError` from the
        client, not a `ValidationError`. This confirms `main`'s existing `RuntimeError`
        handler turns that into a clean exit rather than a traceback.
        """
        monkeypatch.setenv('AIRFLOW_USERNAME', 'u')
        monkeypatch.setenv('AIRFLOW_PASSWORD', 'p')
        fake_client = MagicMock()
        fake_client.dag_run.side_effect = RuntimeError(
            'airflow API returned HTTP 404 for /api/v2/dags/unified_pipeline/dagRuns/typo'
        )
        monkeypatch.setattr(cli, 'AirflowClient', MagicMock(return_value=fake_client))
        monkeypatch.setattr(cli, 'storage', MagicMock())

        assert main(['snapshot', '--run', 'typo']) == 1
        err = capsys.readouterr().err
        assert err.count('\n') == 1
        assert '404' in err

    def test_missing_credentials_exit_non_zero_naming_the_variable(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The whole reason credentials are env vars is to fail closed and say which are missing."""
        monkeypatch.delenv('AIRFLOW_USERNAME', raising=False)
        monkeypatch.delenv('AIRFLOW_PASSWORD', raising=False)
        monkeypatch.setattr(cli, 'AirflowClient', MagicMock())
        monkeypatch.setattr(cli, 'storage', MagicMock())

        assert main(['snapshot', '--run', 'r']) == 1
        err = capsys.readouterr().err
        assert 'AIRFLOW_USERNAME' in err
        assert 'AIRFLOW_PASSWORD' in err

    def test_an_empty_string_credential_is_treated_as_missing(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The subtler half of the fail-closed check: `''` is falsy, not a valid credential."""
        monkeypatch.setenv('AIRFLOW_USERNAME', '')
        monkeypatch.setenv('AIRFLOW_PASSWORD', 'p')
        monkeypatch.setattr(cli, 'AirflowClient', MagicMock())
        monkeypatch.setattr(cli, 'storage', MagicMock())

        assert main(['snapshot', '--run', 'r']) == 1
        err = capsys.readouterr().err
        assert 'AIRFLOW_USERNAME' in err
        assert 'AIRFLOW_PASSWORD' not in err


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch out `bigquery.Client`, exposing the constructor as `client.constructor`."""
    fake = MagicMock()
    fake.query.return_value.result.return_value = []
    fake.constructor = MagicMock(return_value=fake)
    monkeypatch.setattr(cli.bigquery, 'Client', fake.constructor)
    return fake


def _results(*result_sets: list[Any]) -> list[MagicMock]:
    """One fake query job per query the command is expected to run, in order."""
    jobs = []
    for rows in result_sets:
        job = MagicMock()
        job.result.return_value = rows
        jobs.append(job)
    return jobs


def _usage_row(**kw: Any) -> SimpleNamespace:
    """Stand in for a bigquery.Row of the aggregate."""
    base = {
        'run': 'r',
        'step': 'pts_target',
        'tool': 'pts',
        'product': 'platform',
        'started': datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
        'ended': datetime(2026, 7, 22, 2, 0, tzinfo=UTC),
        'net_cost': 21.60,
        'currency': 'GBP',
        'shared_cluster': False,
    }
    base.update(kw)
    return SimpleNamespace(**base)


def _bound(client: MagicMock, call: int = 0) -> dict[str, Any]:
    job_config = client.query.call_args_list[call].kwargs['job_config']
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
        client.query.return_value.result.return_value = [_usage_row(run='older-run')]
        assert main(['history', '--step', 'pts_target']) == 0
        assert capsys.readouterr().out.splitlines()[0].startswith('run')

    def test_usage_reports_coverage_over_the_runs_window(
        self, client: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The table's total is a subset of pipeline spend, and must not read as all of it."""
        coverage_row = SimpleNamespace(currency='GBP', labelled_cost=21.60, pipeline_cost=28.80)
        client.query.side_effect = _results([_usage_row()], [coverage_row])
        assert main(['usage', '--run', 'r']) == 0
        out = capsys.readouterr().out
        assert 'coverage:' in out
        assert '21.60 of the 28.80 GBP the pipeline billed (75.0%)' in out
        assert _bound(client, call=1)['window_start'] == datetime(2026, 7, 21, 14, 0, tzinfo=UTC)
        assert _bound(client, call=1)['window_end'] == datetime(2026, 7, 22, 2, 0, tzinfo=UTC)
        assert _bound(client, call=1)['run'] == _bound(client, call=0)['run']

    def test_coverage_counts_the_run_the_table_shows(
        self, client: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Unbound, the numerator is every run in the window and is not the table's total."""
        coverage_row = SimpleNamespace(currency='GBP', labelled_cost=21.60, pipeline_cost=28.80)
        client.query.side_effect = _results([_usage_row()], [coverage_row])
        assert main(['usage', '--run', RAW_RUN_ID]) == 0
        assert _bound(client, call=1)['run'] == CLEAN_RUN_ID

    def test_an_empty_run_reports_coverage_as_unknown(
        self, client: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """With no rows there is no window, so there is nothing to query and nothing to claim."""
        assert main(['usage', '--run', 'r']) == 0
        assert 'coverage: unknown' in capsys.readouterr().out
        assert client.query.call_count == 1

    def test_history_does_not_report_coverage(self, client: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        """A history spans many runs, so a single window would not describe any of them."""
        client.query.return_value.result.return_value = [_usage_row()]
        assert main(['history', '--step', 'pts_target']) == 0
        assert 'coverage' not in capsys.readouterr().out
        assert client.query.call_count == 1

    def test_api_error_exits_non_zero_with_a_message(
        self, client: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A missing table or a denied project must not surface as a traceback."""
        client.query.side_effect = cli.GoogleAPICallError('403 Access Denied\non the export')
        assert main(['usage', '--run', 'r']) == 1
        err = capsys.readouterr().err
        assert 'billing export query failed' in err
        assert err.count('\n') == 1

    def test_missing_credentials_exit_non_zero_with_a_message(
        self, client: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A machine with no ADC is the first thing a new user hits, and it is not a crash."""
        client.constructor.side_effect = cli.DefaultCredentialsError('could not automatically determine')
        assert main(['usage', '--run', 'r']) == 1
        err = capsys.readouterr().err
        assert 'no Google Cloud credentials found' in err
        assert 'gcloud auth application-default login' in err


class TestJsonOutput:
    def test_models_serialise(self) -> None:
        payload = json.loads(_usage().model_dump_json())
        assert payload['step'] == 'pts_target'
        assert payload['net_cost'] == 1.5
        assert payload['shared_cluster'] is False

    def test_json_does_not_pay_for_the_coverage_query(
        self, client: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """JSON renders no coverage line, and every query here scans a billed export."""
        client.query.return_value.result.return_value = [_usage_row()]
        assert main(['usage', '--run', 'r', '--json']) == 0
        assert capsys.readouterr().out.startswith('[')
        assert client.query.call_count == 1

    def test_json_carries_the_product(self, client: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        """Machine readers need the discriminator as much as the table does."""
        coverage_row = SimpleNamespace(currency='GBP', labelled_cost=1.0, pipeline_cost=2.0)
        client.query.side_effect = _results([_usage_row(product='ppp')], [coverage_row])
        assert main(['usage', '--run', 'r', '--json']) == 0
        assert json.loads(capsys.readouterr().out)[0]['product'] == 'ppp'
