"""Tests for the supervisor CLI."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from orchestration.supervisor import cli
from orchestration.supervisor.airflow import DagRun
from orchestration.supervisor.cli import (
    build_parser,
    main,
    optional_columns,
    render_coverage,
    render_diff,
    render_table,
    totals_by_group,
)
from orchestration.supervisor.datasets import run_name, stage_configs, unified_pipeline_steps
from orchestration.supervisor.diff import ColumnChange, DatasetDiff
from orchestration.supervisor.gcs import Footer, Skipped, collect_diffs
from orchestration.supervisor.journal import JournalEvent, is_heartbeat
from orchestration.supervisor.snapshot import Snapshot
from orchestration.supervisor.stall import RunStallVerdict
from orchestration.supervisor.usage import StepUsage, WindowCoverage
from orchestration.utils.common import (
    GCP_PROJECT_PLATFORM,
    GCS_PIPELINE_RUNS_BUCKET,
    GCS_PRE_RELEASES_BUCKET,
    clean_label,
)

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
        billed_hours=2,
        net_cost=net_cost,
        currency=currency,
        shared_cluster=shared,
        core_seconds=None,
        spot_core_seconds=None,
        machine_families=[],
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


class TestDiffParser:
    def test_run_and_reference_are_required(self) -> None:
        args = build_parser().parse_args(['diff', '--run', 'r', '--reference', 'rel'])
        assert args.run == 'r'
        assert args.reference == 'rel'

    def test_threshold_defaults_to_five_percent(self) -> None:
        args = build_parser().parse_args(['diff', '--run', 'r', '--reference', 'rel'])
        assert args.threshold == 0.05

    def test_threshold_is_overridable(self) -> None:
        args = build_parser().parse_args(['diff', '--run', 'r', '--reference', 'rel', '--threshold', '0.1'])
        assert args.threshold == 0.1

    def test_bucket_flags_default_to_the_standard_buckets(self) -> None:
        args = build_parser().parse_args(['diff', '--run', 'r', '--reference', 'rel'])
        assert args.run_bucket == GCS_PIPELINE_RUNS_BUCKET
        assert args.reference_bucket == GCS_PRE_RELEASES_BUCKET

    def test_bucket_flags_are_overridable(self) -> None:
        args = build_parser().parse_args(
            ['diff', '--run', 'r', '--reference', 'rel', '--run-bucket', 'x', '--reference-bucket', 'y']
        )
        assert args.run_bucket == 'x'
        assert args.reference_bucket == 'y'

    def test_json_flag_defaults_off(self) -> None:
        args = build_parser().parse_args(['diff', '--run', 'r', '--reference', 'rel'])
        assert args.json is False

    def test_rows_flag_defaults_off(self) -> None:
        """Row counts are opt-in: a full release costs ~10s without them, ~7min with."""
        args = build_parser().parse_args(['diff', '--run', 'r', '--reference', 'rel'])
        assert args.rows is False

    def test_rows_flag_is_settable(self) -> None:
        args = build_parser().parse_args(['diff', '--run', 'r', '--reference', 'rel', '--rows'])
        assert args.rows is True


def _diff(**overrides: Any) -> DatasetDiff:
    base: dict[str, Any] = {
        'dataset': 'output/disease',
        'side': 'both',
        'run_rows': 1000,
        'reference_rows': 1000,
        'run_bytes': 1000,
        'reference_bytes': 1000,
        'run_files': 2,
        'reference_files': 2,
        'columns': [],
        'countable': True,
    }
    base.update(overrides)
    return DatasetDiff(**base)


class TestRenderDiff:
    def test_no_material_differences_says_so(self) -> None:
        out = render_diff([_diff()], Skipped(), threshold=0.05)
        assert '1 datasets compared, 0 with material changes' in out
        assert 'No material differences.' in out

    def test_schema_changes_are_shown_even_when_the_size_move_is_below_threshold(self) -> None:
        """A schema-only diff has an unmoved row and byte count.

        With `run_bytes == reference_bytes` and `run_rows == reference_rows`, only
        `diff.columns` being non-empty can be responsible for this diff being material
        or for the column line appearing at all.
        """
        diff = _diff(columns=[ColumnChange(column='new_col', kind='added', run_type='string')])
        out = render_diff([diff], Skipped(), threshold=0.05)
        assert '1 with material changes' in out
        assert 'added' in out
        assert 'new_col' in out

    def test_a_run_only_dataset_is_named_as_such(self) -> None:
        diff = _diff(side='run_only', reference_rows=None, reference_bytes=0, reference_files=0)
        out = render_diff([diff], Skipped(), threshold=0.05)
        assert 'PRESENT IN THE RUN ONLY' in out

    def test_a_reference_only_dataset_is_named_as_such(self) -> None:
        diff = _diff(side='reference_only', run_rows=None, run_bytes=0, run_files=0)
        out = render_diff([diff], Skipped(), threshold=0.05)
        assert 'PRESENT IN THE REFERENCE ONLY' in out

    def test_an_uncountable_dataset_renders_n_a_not_a_blank_or_zero(self) -> None:
        """`countable=False` means the format has no row footer at all.

        Both sides carry `rows=None`. The report must read `n/a`, not an empty cell and
        not `0`, either of which would misread as "this dataset really has zero rows".
        The byte side is moved past the threshold so the diff is material without the
        (skipped, both-None) row comparison being what makes it so.
        """
        diff = _diff(run_rows=None, reference_rows=None, countable=False, run_bytes=2000)
        out = render_diff([diff], Skipped(), threshold=0.05)
        assert 'n/a -> n/a' in out
        assert '0 -> 0' not in out

    def test_a_missing_row_count_on_a_countable_dataset_renders_a_dash_not_n_a(self) -> None:
        """Distinguishes `_MISSING` ('-') from `_UNCOUNTABLE` ('n/a').

        `countable=True` here (the format does have footers); `reference_rows` is
        simply unset. `_count` must read this as `-`, not fall through to the
        `n/a` branch that a format with no footer at all gets.
        """
        diff = _diff(reference_rows=None, run_rows=5000, run_bytes=2000)
        out = render_diff([diff], Skipped(), threshold=0.05)
        assert 'rows - -> 5,000' in out
        assert 'n/a' not in out

    def test_row_counts_are_thousands_separated(self) -> None:
        diff = _diff(run_rows=1234567, reference_rows=1000, run_bytes=2000)
        out = render_diff([diff], Skipped(), threshold=0.05)
        assert '1,234,567' in out

    def test_threshold_is_shown_as_a_percentage(self) -> None:
        out = render_diff([], Skipped(), threshold=0.1)
        assert '10%' in out

    def test_the_footer_states_what_is_not_compared(self) -> None:
        out = render_diff([], Skipped(), threshold=0.05)
        assert 'intermediate/' in out
        assert 'templated' in out

    def test_stages_without_config_are_reported_with_a_count_and_gentropy_named(self) -> None:
        skipped = Skipped(stages_without_config=['gentropy_l2g', 'gentropy_variant_annotation'])
        out = render_diff([], skipped, threshold=0.05)
        assert '2 steps skipped' in out
        assert 'gentropy' in out

    def test_steps_without_datasets_are_reported_as_normal_not_an_anomaly(self) -> None:
        skipped = Skipped(steps_without_datasets=['pts_association'])
        out = render_diff([], skipped, threshold=0.05)
        assert '1 steps declare no release dataset' in out
        assert 'not an anomaly' in out

    def test_datasets_absent_from_both_are_named(self) -> None:
        skipped = Skipped(datasets_absent_from_both=['output/orphan'])
        out = render_diff([], skipped, threshold=0.05)
        assert '1 datasets absent from both buckets' in out
        assert 'output/orphan' in out

    def test_undeclared_datasets_are_flagged_as_a_possible_pipeline_drop(self) -> None:
        skipped = Skipped(undeclared_in_buckets=['output/retired'])
        out = render_diff([], skipped, threshold=0.05)
        assert 'output/retired' in out
        assert 'dropped from the pipeline' in out

    def test_no_skip_footer_lines_appear_when_nothing_was_skipped(self) -> None:
        out = render_diff([], Skipped(), threshold=0.05)
        assert 'steps skipped' not in out
        assert 'declare no release dataset' not in out
        assert 'absent from both buckets' not in out
        assert 'declared by no' not in out

    def test_rows_skipped_notes_that_rows_were_not_read(self) -> None:
        out = render_diff([], Skipped(), threshold=0.05, rows_skipped=True)
        assert '--rows' in out
        assert 'not read' in out

    def test_rows_skipped_defaults_to_false_and_adds_no_note(self) -> None:
        """`render_diff` itself still defaults `rows_skipped` to False.

        The CLI is what makes skipping rows the actual default, by calling this with
        `rows_skipped=not args.rows`; this pins that `render_diff`'s own default is
        unchanged, so a caller that does not pass the flag gets no note.
        """
        out = render_diff([], Skipped(), threshold=0.05)
        assert '--rows' not in out
        assert 'not read' not in out


class TestDiffLoadersAgainstTheRealConfig:
    """Measured against both configs and `unified_pipeline.yaml` on 2026-08-24."""

    def test_loads_pis_and_pts_only(self) -> None:
        """Gentropy is deliberately excluded; its config is not `{stage}/config.yaml`-shaped."""
        assert set(stage_configs()) == {'pis', 'pts'}

    def test_loads_every_declared_step(self) -> None:
        steps = unified_pipeline_steps()
        assert len(steps) == 132
        assert 'pts_disease' in steps

    def test_loads_the_configured_run_name(self) -> None:
        """Checks the field is read at all, not its live value.

        `run_name` is edited to configure each dev run — unlike the step count above,
        which changes deliberately and rarely — so pinning today's value would red-fail
        CI on the next dev run that sets a different one, teaching people to edit this
        test to match config. A non-None return is enough to catch what this test
        actually guards against: a rename of the *field* in `unified_pipeline.yaml`
        that would silently turn `datasets.run_name`'s `up.get('run_name')` into a
        permanent `None`.
        """
        assert run_name() is not None

    def test_collect_diffs_reconciles_against_the_measured_release_inventory(self) -> None:
        """With both buckets empty, every declared dataset is absent from both sides.

        Pins the branch's own measured totals (see `test_supervisor_datasets.py`): 71
        release datasets total, 12 gentropy steps with no local config, 63 steps
        producing none of the remaining 120 pis/pts steps.
        """

        class _EmptyBucket:
            def list_blobs(self, prefix: str) -> list[Any]:
                return []

        diffs, skipped = collect_diffs(
            _EmptyBucket(), 'run', _EmptyBucket(), 'release', unified_pipeline_steps(), stage_configs()
        )
        assert diffs == []
        assert len(skipped.stages_without_config) == 12
        assert len(skipped.steps_without_datasets) == 63
        assert len(skipped.datasets_absent_from_both) == 71
        assert skipped.undeclared_in_buckets == []


@pytest.fixture
def diff_command(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch `storage.Client`, `collect_diffs`, and the two config loaders.

    The loaders are stubbed to a single tiny step so tests are isolated from the real
    yaml files and from each other; `TestDiffLoadersAgainstTheRealConfig` above already
    covers the loaders against the real repo. `collect_diffs` is stubbed with an empty
    result by default; tests override `.return_value` to check rendering.
    """
    buckets: dict[str, MagicMock] = {}

    def fake_bucket(name: str) -> MagicMock:
        return buckets.setdefault(name, MagicMock())

    storage_client = MagicMock()
    storage_client.bucket.side_effect = fake_bucket
    fake_storage = MagicMock()
    fake_storage.Client.return_value = storage_client
    monkeypatch.setattr(cli, 'storage', fake_storage)
    monkeypatch.setattr(cli, 'unified_pipeline_steps', MagicMock(return_value=['pts_disease']))
    monkeypatch.setattr(cli, 'stage_configs', MagicMock(return_value={'pts': {'steps': {}}}))
    collect = MagicMock(return_value=([], Skipped()))
    monkeypatch.setattr(cli, 'collect_diffs', collect)
    collect.storage_client = storage_client
    collect.buckets = buckets
    return collect


class TestDiffCommand:
    def test_buckets_are_constructed_from_the_default_bucket_names(self, diff_command: MagicMock) -> None:
        assert main(['diff', '--run', 'myrun', '--reference', 'rel1']) == 0
        diff_command.storage_client.bucket.assert_any_call(GCS_PIPELINE_RUNS_BUCKET)
        diff_command.storage_client.bucket.assert_any_call(GCS_PRE_RELEASES_BUCKET)

    def test_the_run_bucket_and_reference_bucket_are_not_swapped(self, diff_command: MagicMock) -> None:
        """Each bucket object must reach `collect_diffs` in its own slot, not the other's.

        Both bucket names are always called regardless of which variable holds which
        object, so a test only checking `assert_any_call` on each name would not catch
        the two being swapped; this checks the actual objects `collect_diffs` receives.
        """
        assert main(['diff', '--run', 'myrun', '--reference', 'rel1']) == 0
        args = diff_command.call_args.args
        assert args[0] is diff_command.buckets[GCS_PIPELINE_RUNS_BUCKET]
        assert args[2] is diff_command.buckets[GCS_PRE_RELEASES_BUCKET]

    def test_bucket_flags_override_the_defaults(self, diff_command: MagicMock) -> None:
        assert (
            main(
                [
                    'diff',
                    '--run',
                    'myrun',
                    '--reference',
                    'rel1',
                    '--run-bucket',
                    'x',
                    '--reference-bucket',
                    'y',
                ]
            )
            == 0
        )
        diff_command.storage_client.bucket.assert_any_call('x')
        diff_command.storage_client.bucket.assert_any_call('y')

    def test_collect_diffs_is_called_with_the_run_and_reference_prefixes(self, diff_command: MagicMock) -> None:
        assert main(['diff', '--run', 'myrun', '--reference', 'rel1']) == 0
        args = diff_command.call_args.args
        assert args[1] == 'myrun'
        assert args[3] == 'rel1'

    def test_collect_diffs_receives_the_loaded_steps_and_stage_configs(self, diff_command: MagicMock) -> None:
        assert main(['diff', '--run', 'myrun', '--reference', 'rel1']) == 0
        args = diff_command.call_args.args
        assert args[4] == ['pts_disease']
        assert args[5] == {'pts': {'steps': {}}}

    def test_text_output_renders_the_report(
        self, diff_command: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        diff_command.return_value = ([_diff()], Skipped())
        assert main(['diff', '--run', 'myrun', '--reference', 'rel1']) == 0
        out = capsys.readouterr().out
        assert 'datasets compared' in out

    def test_json_output_carries_diffs_and_skipped(
        self, diff_command: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        diff_command.return_value = ([_diff()], Skipped(stages_without_config=['gentropy_l2g']))
        assert main(['diff', '--run', 'myrun', '--reference', 'rel1', '--json']) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload['diffs'][0]['dataset'] == 'output/disease'
        assert payload['skipped']['stages_without_config'] == ['gentropy_l2g']

    def test_the_threshold_flag_reaches_rendering(
        self, diff_command: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A +30% byte move is material at the default 5% threshold but not at 50%."""
        diff_command.return_value = ([_diff(run_bytes=1300, reference_bytes=1000)], Skipped())
        assert main(['diff', '--run', 'myrun', '--reference', 'rel1', '--threshold', '0.5']) == 0
        out = capsys.readouterr().out
        assert '0 with material changes' in out

    def test_a_footer_reader_is_passed_to_collect_diffs_for_each_side_with_rows(
        self, diff_command: MagicMock
    ) -> None:
        assert main(['diff', '--run', 'myrun', '--reference', 'rel1', '--rows']) == 0
        args = diff_command.call_args.args
        assert callable(args[6])
        assert callable(args[7])

    def test_default_passes_none_as_both_footer_readers(self, diff_command: MagicMock) -> None:
        """Row counts are opt-in, so the default must skip building `footer_reader` for either side.

        Building it for real would still construct two `pyarrow` GCS filesystems for
        no reason; passing `None` through for both is what makes the default fast.
        """
        assert main(['diff', '--run', 'myrun', '--reference', 'rel1']) == 0
        args = diff_command.call_args.args
        assert args[6] is None
        assert args[7] is None

    def test_default_reaches_the_rendered_footer_note(
        self, diff_command: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Row counts being skipped by default must be visible in the report, not silent."""
        assert main(['diff', '--run', 'myrun', '--reference', 'rel1']) == 0
        out = capsys.readouterr().out
        assert '--rows' in out

    def test_with_the_rows_flag_the_footer_carries_no_note(
        self, diff_command: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(['diff', '--run', 'myrun', '--reference', 'rel1', '--rows']) == 0
        out = capsys.readouterr().out
        assert '--rows' not in out

    def test_the_same_run_and_reference_name_across_different_buckets_is_a_real_comparison(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`--run 26.09 --reference 26.09` against two different buckets is ordinary CLI input.

        A run and the release of the same name is the most natural pre-publication
        sanity check there is, and must not be rejected just because the two `--run`/
        `--reference` values happen to match. `collect_diffs` is left un-mocked here
        (unlike `diff_command`'s other tests) so this exercises the real per-side
        routing: each side's footer reader is built from its own `--*-bucket` name and
        reads distinguishable row counts (5 vs 9), so a reader swapped between the two
        sides would show up as a wrong number, not just a missing one.
        """
        run_bucket = MagicMock()
        run_bucket.list_blobs.return_value = [SimpleNamespace(name='same-name/output/disease/part-0.parquet', size=1)]
        reference_bucket = MagicMock()
        reference_bucket.list_blobs.return_value = [
            SimpleNamespace(name='same-name/output/disease/part-0.parquet', size=1)
        ]
        buckets = {'run-bucket-name': run_bucket, 'reference-bucket-name': reference_bucket}

        storage_client = MagicMock()
        storage_client.bucket.side_effect = lambda name: buckets[name]
        fake_storage = MagicMock()
        fake_storage.Client.return_value = storage_client
        monkeypatch.setattr(cli, 'storage', fake_storage)

        def fake_footer_reader(bucket_name: str) -> Callable[[str], Footer]:
            rows = 5 if bucket_name == 'run-bucket-name' else 9
            return lambda name: Footer(rows=rows)

        monkeypatch.setattr(cli, 'footer_reader', fake_footer_reader)
        monkeypatch.setattr(cli, 'unified_pipeline_steps', MagicMock(return_value=['pts_disease']))
        monkeypatch.setattr(
            cli,
            'stage_configs',
            MagicMock(
                return_value={'pts': {'steps': {'disease': [{'name': 't', 'destination': 'output/disease'}]}}}
            ),
        )

        assert (
            main([
                'diff',
                '--run',
                'same-name',
                '--reference',
                'same-name',
                '--run-bucket',
                'run-bucket-name',
                '--reference-bucket',
                'reference-bucket-name',
                '--rows',
            ])
            == 0
        )
        out = capsys.readouterr().out
        assert 'rows 9 -> 5' in out


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
        'billed_hours': 4,
        'net_cost': 21.60,
        'currency': 'GBP',
        'shared_cluster': False,
        'core_seconds': None,
        'spot_core_seconds': None,
        'machine_families': [],
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


class TestObserveParser:
    def test_dag_defaults_to_the_unified_pipeline(self) -> None:
        assert build_parser().parse_args(['observe', '--issue', '5']).dag == 'unified_pipeline'

    def test_issue_is_required_and_parsed_as_an_int(self) -> None:
        assert build_parser().parse_args(['observe', '--issue', '5']).issue == 5

    def test_dry_run_defaults_off(self) -> None:
        assert build_parser().parse_args(['observe', '--issue', '5']).dry_run is False

    def test_dry_run_is_settable(self) -> None:
        assert build_parser().parse_args(['observe', '--issue', '5', '--dry-run']).dry_run is True

    def test_rows_defaults_off(self) -> None:
        assert build_parser().parse_args(['observe', '--issue', '5']).rows is False

    def test_run_and_reference_default_to_none(self) -> None:
        """Diffing is opt-in on `observe`.

        See `cli.py`'s module docstring for why neither is derived automatically from
        `unified_pipeline.yaml`.
        """
        args = build_parser().parse_args(['observe', '--issue', '5'])
        assert args.run is None
        assert args.reference is None

    def test_bucket_flags_default_to_the_standard_buckets(self) -> None:
        args = build_parser().parse_args(['observe', '--issue', '5'])
        assert args.run_bucket == GCS_PIPELINE_RUNS_BUCKET
        assert args.reference_bucket == GCS_PRE_RELEASES_BUCKET

    def test_threshold_defaults_to_five_percent(self) -> None:
        assert build_parser().parse_args(['observe', '--issue', '5']).threshold == 0.05


def _snapshot(**overrides: Any) -> Snapshot:
    """A `Snapshot` for a running, otherwise idle wakeup, overridable field by field."""
    base: dict[str, Any] = {
        'dag_id': 'unified_pipeline',
        'run_id': 'run123',
        'taken_at': datetime(2026, 8, 24, 12, 0, tzinfo=UTC),
        'run_state': 'running',
        'counts': {'running': 1},
        'running': ['pts_target.run_pts_target'],
        'failed': [],
        'succeeded': [],
        'durations': {},
        'try_numbers': {},
        'stalls': [],
        'journal_events': 0,
    }
    base.update(overrides)
    return Snapshot(**base)


@pytest.fixture
def observe_command(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Patch every I/O boundary `observe` crosses.

    `observer.observe` and `render_comment` run for real, so a scenario test is
    exercising the actual wiring between them, not a second copy of logic those
    modules' own test files already cover.

    The default `take_snapshot` return is a running, otherwise empty snapshot — the
    common case per `cli.py`'s module docstring — so a test that never overrides it is
    exercising "a wakeup with nothing new" by construction, not by accident.
    """
    monkeypatch.setenv('AIRFLOW_USERNAME', 'u')
    monkeypatch.setenv('AIRFLOW_PASSWORD', 'p')

    fake_airflow_client = MagicMock()
    fake_airflow_client.active_dag_run.return_value = DagRun(dag_run_id='run123', state='running')
    # No run has ever finished by default: `most_recent_dag_run` is the F1 fallback,
    # consulted only when `active_dag_run` finds nothing. Most scenarios here short-circuit
    # past it (see `main`'s `or`), so this default only matters to a test that overrides
    # `active_dag_run` to None and wants the genuinely-idle case rather than the fallback.
    fake_airflow_client.most_recent_dag_run.return_value = None
    airflow_client_cls = MagicMock(return_value=fake_airflow_client)
    monkeypatch.setattr(cli, 'AirflowClient', airflow_client_cls)

    monkeypatch.setattr(cli, 'storage', MagicMock())

    fake_journal = MagicMock()
    fake_journal.read.return_value = []
    # No dataset diff has run for this journal yet, by default — see F3's once-only gate
    # in `main`. A test that wants to simulate an already-diffed run overrides this.
    fake_journal.has.return_value = False
    journal_cls = MagicMock(return_value=fake_journal)
    monkeypatch.setattr(cli, 'Journal', journal_cls)

    take_snapshot = MagicMock(return_value=_snapshot())
    monkeypatch.setattr(cli, 'take_snapshot', take_snapshot)

    collect_diffs = MagicMock(return_value=([], Skipped()))
    monkeypatch.setattr(cli, 'collect_diffs', collect_diffs)
    footer_reader = MagicMock()
    monkeypatch.setattr(cli, 'footer_reader', footer_reader)
    monkeypatch.setattr(cli, 'unified_pipeline_steps', MagicMock(return_value=['pts_disease']))
    monkeypatch.setattr(cli, 'stage_configs', MagicMock(return_value={'pts': {'steps': {}}}))
    fake_run_name = MagicMock(return_value='ds/some_run')
    monkeypatch.setattr(cli, 'run_name', fake_run_name)

    fake_github_app = MagicMock()
    github_app_cls = MagicMock(return_value=fake_github_app)
    monkeypatch.setattr(cli, 'GitHubApp', github_app_cls)
    read_app_key = MagicMock(return_value='pem')
    monkeypatch.setattr(cli, 'read_app_key', read_app_key)
    monkeypatch.setattr(cli, 'secretmanager', MagicMock())

    return SimpleNamespace(
        airflow_client=fake_airflow_client,
        airflow_client_cls=airflow_client_cls,
        journal=fake_journal,
        journal_cls=journal_cls,
        take_snapshot=take_snapshot,
        collect_diffs=collect_diffs,
        footer_reader=footer_reader,
        github_app=fake_github_app,
        github_app_cls=github_app_cls,
        read_app_key=read_app_key,
        run_name=fake_run_name,
    )


class TestObserveCommand:
    def test_an_idle_pipeline_exits_zero_and_posts_nothing(
        self, observe_command: SimpleNamespace, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """No run has ever started is the ordinary case, not an error.

        Distinct from a run that finished between wakeups (F1's fallback via
        `most_recent_dag_run`, covered below): here neither discovery call finds
        anything, because nothing has ever run.
        """
        observe_command.airflow_client.active_dag_run.return_value = None
        assert main(['observe', '--issue', '5']) == 0
        observe_command.github_app.comment.assert_not_called()
        observe_command.journal_cls.assert_not_called()
        assert 'unified_pipeline' in capsys.readouterr().out

    def test_a_wakeup_with_nothing_new_posts_no_comment_but_marks_observation_started(
        self, observe_command: SimpleNamespace
    ) -> None:
        """No comment, but `observation_started` still gets journalled — see F7.

        Regression: before F7, `body is None` implied the journal loop wrote nothing at
        all, so a wakeup with nothing new left no trace it had run.
        """
        assert main(['observe', '--issue', '5']) == 0
        observe_command.github_app.comment.assert_not_called()
        events = [call.args[0] for call in observe_command.journal.append.call_args_list]
        non_heartbeats = [e for e in events if not is_heartbeat(e)]
        assert [e.event_type for e in non_heartbeats] == ['observation_started']
        assert non_heartbeats[0].payload == {'run_name': 'ds/some_run'}
        assert sum(1 for e in events if is_heartbeat(e)) == 1

    def test_dry_run_journals_no_heartbeat_either(self, observe_command: SimpleNamespace) -> None:
        assert main(['observe', '--issue', '5', '--dry-run']) == 0
        observe_command.journal.append.assert_not_called()

    def test_dry_run_posts_nothing_and_writes_no_journal_event(
        self, observe_command: SimpleNamespace, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observe_command.take_snapshot.return_value = _snapshot(
            failed=['pts_target.run_pts_target'], try_numbers={'pts_target.run_pts_target': 1}
        )
        assert main(['observe', '--issue', '5', '--dry-run']) == 0
        assert 'pts_target' in capsys.readouterr().out
        observe_command.journal.append.assert_not_called()
        observe_command.github_app.comment.assert_not_called()
        observe_command.github_app_cls.assert_not_called()

    def test_dry_run_mints_no_token(self, observe_command: SimpleNamespace) -> None:
        """`read_app_key` is the call that reaches Secret Manager; dry-run must not attempt it."""
        observe_command.take_snapshot.return_value = _snapshot(
            failed=['pts_target.run_pts_target'], try_numbers={'pts_target.run_pts_target': 1}
        )
        assert main(['observe', '--issue', '5', '--dry-run']) == 0
        observe_command.read_app_key.assert_not_called()

    def test_dry_run_with_nothing_new_says_so_rather_than_printing_nothing(
        self, observe_command: SimpleNamespace, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(['observe', '--issue', '5', '--dry-run']) == 0
        assert capsys.readouterr().out.strip() != ''

    def test_a_real_wakeup_with_something_new_journals_and_posts(self, observe_command: SimpleNamespace) -> None:
        observe_command.take_snapshot.return_value = _snapshot(
            failed=['pts_target.run_pts_target'], try_numbers={'pts_target.run_pts_target': 1}
        )
        assert main(['observe', '--issue', '5']) == 0
        events = [call.args[0] for call in observe_command.journal.append.call_args_list]
        non_heartbeats = {event.event_type for event in events if not is_heartbeat(event)}
        assert non_heartbeats == {'step_failed', 'observation_started'}
        assert sum(1 for event in events if is_heartbeat(event)) == 1
        failure = next(event for event in events if event.event_type == 'step_failed')
        assert failure.step == 'pts_target'
        observe_command.github_app.comment.assert_called_once()
        assert observe_command.github_app.comment.call_args.args[0] == 5

    def test_the_post_happens_before_the_journal_write(self, observe_command: SimpleNamespace) -> None:
        """Posting must be attempted first — see `main`'s comment on why the order matters."""
        observe_command.take_snapshot.return_value = _snapshot(
            failed=['pts_target.run_pts_target'], try_numbers={'pts_target.run_pts_target': 1}
        )
        order: list[str] = []
        observe_command.github_app.comment.side_effect = lambda *a, **kw: order.append('comment')
        observe_command.journal.append.side_effect = lambda *a, **kw: order.append('append')

        assert main(['observe', '--issue', '5']) == 0
        # Three journal writes now (`step_failed`, `observation_started`, and the
        # heartbeat every wakeup carries), all still after the single post — the post
        # itself must lead, not just come first.
        assert order == ['comment', 'append', 'append', 'append']

    def test_a_failed_post_leaves_the_journal_empty_and_is_retried_next_wakeup(
        self, observe_command: SimpleNamespace
    ) -> None:
        """Regression: journalling before posting would silently lose a report on a failed post.

        `github_app.comment` raising must leave the journal untouched, so the next wakeup's
        `observe()` recomputes the same observation (the journal fixture's `read()` still
        returns `[]`, exactly as if nothing had been recorded) and tries to post it again —
        a retry, not a silent drop.
        """
        observe_command.take_snapshot.return_value = _snapshot(
            failed=['pts_target.run_pts_target'], try_numbers={'pts_target.run_pts_target': 1}
        )
        observe_command.github_app.comment.side_effect = RuntimeError('comment post failed with HTTP 502: boom')

        assert main(['observe', '--issue', '5']) == 1
        observe_command.journal.append.assert_not_called()
        assert observe_command.github_app.comment.call_count == 1

        observe_command.github_app.comment.side_effect = None
        assert main(['observe', '--issue', '5']) == 0
        assert observe_command.github_app.comment.call_count == 2
        # `step_failed`, `observation_started` (see F7), and the heartbeat — the retried
        # wakeup journals all three.
        assert observe_command.journal.append.call_count == 3

    def test_a_failed_pipeline_step_exits_zero_not_one(self, observe_command: SimpleNamespace) -> None:
        """The observer's own health decides the exit code, never the pipeline's state."""
        observe_command.take_snapshot.return_value = _snapshot(
            failed=['pts_target.run_pts_target'], try_numbers={'pts_target.run_pts_target': 1}
        )
        assert main(['observe', '--issue', '5']) == 0

    def test_a_run_stall_is_posted_and_journalled_under_its_reason(self, observe_command: SimpleNamespace) -> None:
        observe_command.take_snapshot.return_value = _snapshot(
            run_stall=RunStallVerdict(reason='stuck_trigger', pending=4)
        )
        assert main(['observe', '--issue', '5']) == 0
        observe_command.github_app.comment.assert_called_once()
        assert 'Run stalled' in observe_command.github_app.comment.call_args.args[1]
        events = [call.args[0] for call in observe_command.journal.append.call_args_list]
        assert any(event.event_type == 'run_stall_detected_stuck_trigger' for event in events)

    def test_a_run_stall_already_journalled_is_not_reposted(self, observe_command: SimpleNamespace) -> None:
        observe_command.take_snapshot.return_value = _snapshot(
            run_stall=RunStallVerdict(reason='stuck_trigger', pending=4)
        )
        observe_command.journal.read.return_value = [
            JournalEvent(event_type='run_stall_detected_stuck_trigger', at=datetime(2026, 8, 24, 12, 0, tzinfo=UTC)),
        ]
        assert main(['observe', '--issue', '5']) == 0
        observe_command.github_app.comment.assert_not_called()
        events = [call.args[0] for call in observe_command.journal.append.call_args_list]
        assert not any(event.event_type == 'run_stall_detected_stuck_trigger' for event in events)

    def test_a_run_discovered_only_via_the_fallback_still_reports_and_journals_finishing(
        self, observe_command: SimpleNamespace
    ) -> None:
        """Regression for F1: a run that finished between wakeups must not go unreported.

        `active_dag_run` finding nothing is not the end of the story: without the
        `most_recent_dag_run` fallback, a run that finished between two wakeups is never
        discovered again, so its terminal-state comment and `run_finished` event never
        fire. This drives the whole path through that fallback — not just `main`'s exit
        code, which a weaker version of this test (mocking `active_dag_run` to return a
        *running* run while `take_snapshot` reported `failed`, a combination production
        cannot produce, since `take_snapshot`'s `run_state` describes the very run
        `active_dag_run` found) used to certify even with the fallback absent.
        """
        observe_command.airflow_client.active_dag_run.return_value = None
        observe_command.airflow_client.most_recent_dag_run.return_value = DagRun(dag_run_id='run123', state='failed')
        observe_command.take_snapshot.return_value = _snapshot(run_state='failed')

        assert main(['observe', '--issue', '5']) == 0

        observe_command.github_app.comment.assert_called_once()
        assert 'Run FAILED' in observe_command.github_app.comment.call_args.args[1]
        events = [call.args[0] for call in observe_command.journal.append.call_args_list]
        assert any(event.event_type == 'run_finished' for event in events)

    def test_a_terminal_run_journals_no_heartbeat(self, observe_command: SimpleNamespace) -> None:
        """A finished run must stop accumulating heartbeats, forever, one per wakeup.

        `most_recent_dag_run` rediscovers a finished run on every wakeup (see the
        module docstring), and `observation_started`/`run_finished` are each gated on
        their own idempotency key so they stop growing the journal after they first
        fire — but the heartbeat append had no such gate, so an idle pipeline kept
        writing 144 objects a day into a journal with nothing left to say. This pins
        that a terminal `run_state` writes none at all.
        """
        observe_command.take_snapshot.return_value = _snapshot(run_state='success')
        assert main(['observe', '--issue', '5']) == 0
        events = [call.args[0] for call in observe_command.journal.append.call_args_list]
        assert not any(is_heartbeat(event) for event in events)

    def test_a_failed_terminal_run_journals_no_heartbeat_either(self, observe_command: SimpleNamespace) -> None:
        """`'failed'` is terminal too, not only `'success'` — both stop the run."""
        observe_command.take_snapshot.return_value = _snapshot(run_state='failed')
        assert main(['observe', '--issue', '5']) == 0
        events = [call.args[0] for call in observe_command.journal.append.call_args_list]
        assert not any(is_heartbeat(event) for event in events)

    def test_the_diff_does_not_run_before_a_terminal_state(self, observe_command: SimpleNamespace) -> None:
        observe_command.take_snapshot.return_value = _snapshot(run_state='running')
        assert main(['observe', '--issue', '5', '--run', 'r', '--reference', 'rel']) == 0
        observe_command.collect_diffs.assert_not_called()

    def test_the_diff_runs_at_a_terminal_state(self, observe_command: SimpleNamespace) -> None:
        observe_command.take_snapshot.return_value = _snapshot(run_state='success')
        assert main(['observe', '--issue', '5', '--run', 'r', '--reference', 'rel']) == 0
        observe_command.collect_diffs.assert_called_once()
        args = observe_command.collect_diffs.call_args.args
        assert args[1] == 'r'
        assert args[3] == 'rel'

    def test_the_diff_is_skipped_at_a_terminal_state_without_run_and_reference(
        self, observe_command: SimpleNamespace
    ) -> None:
        observe_command.take_snapshot.return_value = _snapshot(run_state='success')
        assert main(['observe', '--issue', '5']) == 0
        observe_command.collect_diffs.assert_not_called()

    def test_the_diff_journals_its_own_completion_once_it_runs(self, observe_command: SimpleNamespace) -> None:
        """See F3: the terminal-state diff marks itself done, so a later wakeup can skip it."""
        observe_command.take_snapshot.return_value = _snapshot(run_state='success')
        assert main(['observe', '--issue', '5', '--run', 'r', '--reference', 'rel']) == 0
        events = [call.args[0] for call in observe_command.journal.append.call_args_list]
        assert any(event.event_type == 'dataset_diff_completed' for event in events)

    def test_the_diff_does_not_rerun_once_already_marked_complete(self, observe_command: SimpleNamespace) -> None:
        """Regression for F3: an already-marked run must not rerun the diff.

        Without this gate, a finished run's diff would rerun — and repost a full "Dataset
        comparison" section — on every ten-minute wakeup forever, re-reading every parquet
        footer on both sides each time `--rows` is set.
        """
        observe_command.take_snapshot.return_value = _snapshot(run_state='success')
        observe_command.journal.has.return_value = True
        assert main(['observe', '--issue', '5', '--run', 'r', '--reference', 'rel']) == 0
        observe_command.collect_diffs.assert_not_called()
        events = [call.args[0] for call in observe_command.journal.append.call_args_list]
        assert not any(event.event_type == 'dataset_diff_completed' for event in events)

    def test_row_counts_are_only_read_with_the_rows_flag(self, observe_command: SimpleNamespace) -> None:
        observe_command.take_snapshot.return_value = _snapshot(run_state='success')
        assert main(['observe', '--issue', '5', '--run', 'r', '--reference', 'rel']) == 0
        observe_command.footer_reader.assert_not_called()

    def test_rows_flag_reaches_the_footer_reader(self, observe_command: SimpleNamespace) -> None:
        observe_command.take_snapshot.return_value = _snapshot(run_state='success')
        assert main(['observe', '--issue', '5', '--run', 'r', '--reference', 'rel', '--rows']) == 0
        assert observe_command.footer_reader.call_count == 2

    def test_a_terminal_run_with_a_diff_renders_both_sections(
        self, observe_command: SimpleNamespace, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`diffs=[]` (ran, found nothing) is distinct from `diffs=None` (did not run).

        Both the run-finished heading and a dataset-comparison section must appear.
        """
        observe_command.take_snapshot.return_value = _snapshot(run_state='success')
        assert main(['observe', '--issue', '5', '--run', 'r', '--reference', 'rel', '--dry-run']) == 0
        out = capsys.readouterr().out
        assert 'Run succeeded' in out
        assert 'Dataset comparison' in out

    def test_missing_credentials_exit_non_zero(
        self, observe_command: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv('AIRFLOW_USERNAME', raising=False)
        monkeypatch.delenv('AIRFLOW_PASSWORD', raising=False)
        assert main(['observe', '--issue', '5']) == 1
        assert 'AIRFLOW_USERNAME' in capsys.readouterr().err

    def test_the_journal_prefix_is_keyed_on_the_discovered_dag_run_id(self, observe_command: SimpleNamespace) -> None:
        assert main(['observe', '--issue', '5']) == 0
        prefix = observe_command.journal_cls.call_args.kwargs['prefix']
        assert prefix == '_agent/unified_pipeline/run123/journal'

    def test_the_app_key_is_read_from_the_platform_project(self, observe_command: SimpleNamespace) -> None:
        observe_command.take_snapshot.return_value = _snapshot(
            failed=['pts_target.run_pts_target'], try_numbers={'pts_target.run_pts_target': 1}
        )
        assert main(['observe', '--issue', '5']) == 0
        args = observe_command.read_app_key.call_args.args
        assert args[1] == GCP_PROJECT_PLATFORM
        assert args[2] == 'supervisor-github-app-key'

    def test_the_github_app_is_constructed_with_the_verified_identity(
        self, observe_command: SimpleNamespace
    ) -> None:
        observe_command.take_snapshot.return_value = _snapshot(
            failed=['pts_target.run_pts_target'], try_numbers={'pts_target.run_pts_target': 1}
        )
        assert main(['observe', '--issue', '5']) == 0
        kwargs = observe_command.github_app_cls.call_args.kwargs
        assert kwargs['app_id'] == '4699938'
        assert kwargs['installation_id'] == 156145657
        assert kwargs['repo'] == 'opentargets/pipeline'
