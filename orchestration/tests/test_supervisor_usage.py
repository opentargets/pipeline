"""Tests for the supervisor usage module."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from orchestration.supervisor.usage import (
    _AGGREGATE,
    _LABELLED_CTE,
    MAX_BYTES_BILLED,
    BillingExport,
    StepUsage,
    WindowCoverage,
    run_usage_query,
    step_history_query,
    total_cost,
    usage_window,
    window_coverage_query,
)

TABLE = 'proj.ds.tbl'
WINDOW = (datetime(2026, 7, 21, 14, 0, tzinfo=UTC), datetime(2026, 7, 22, 2, 0, tzinfo=UTC))


class TestStepUsage:
    def test_construction(self) -> None:
        u = StepUsage(
            run='manual__2026-07-21t15-07-47-545737-00-00',
            step='pts_target',
            tool='pts',
            product='platform',
            started=datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
            ended=datetime(2026, 7, 21, 16, 0, tzinfo=UTC),
            span_hours=2.0,
            net_cost=1.5,
            currency='GBP',
            shared_cluster=False,
        )
        assert u.step == 'pts_target'
        assert u.span_hours == 2.0

    def test_tool_must_be_a_pipeline_tool(self) -> None:
        with pytest.raises(ValidationError):
            StepUsage.model_validate({
                'run': 'r',
                'step': 's',
                'tool': 'nextgen',
                'product': 'platform',
                'started': datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
                'ended': datetime(2026, 7, 21, 15, 0, tzinfo=UTC),
                'span_hours': 1.0,
                'net_cost': 0.1,
                'currency': 'GBP',
                'shared_cluster': False,
            })

    def test_product_may_be_missing(self) -> None:
        """Not every billed resource carries the label, and dropping those rows would lose cost."""
        u = StepUsage.model_validate({
            'run': 'r',
            'step': 's',
            'tool': 'pis',
            'product': None,
            'started': datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
            'ended': datetime(2026, 7, 21, 15, 0, tzinfo=UTC),
            'span_hours': 1.0,
            'net_cost': 0.1,
            'currency': 'GBP',
            'shared_cluster': False,
        })
        assert u.product is None

    def test_negative_cost_is_allowed(self) -> None:
        """A credit-dominated row can net negative. This must not be rejected."""
        u = StepUsage(
            run='r',
            step='s',
            tool='pis',
            product='platform',
            started=datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
            ended=datetime(2026, 7, 21, 15, 0, tzinfo=UTC),
            span_hours=1.0,
            net_cost=-0.02,
            currency='GBP',
            shared_cluster=False,
        )
        assert u.net_cost < 0


class TestRunUsageQuery:
    def test_parameters_are_bound_not_interpolated(self) -> None:
        sql, params = run_usage_query(TABLE, run='a-run', since=date(2026, 5, 1))
        assert 'a-run' not in sql
        names = {p.name: p.value for p in params}
        assert names['run'] == 'a-run'
        assert names['since'] == date(2026, 5, 1)

    def test_table_is_backticked(self) -> None:
        sql, _ = run_usage_query(TABLE, run='r', since=date(2026, 5, 1))
        assert '`proj.ds.tbl`' in sql

    def test_filters_on_partition_column(self) -> None:
        """The export is ingestion-time partitioned.

        A usage_start_time filter would scan the whole table.
        """
        sql, _ = run_usage_query(TABLE, run='r', since=date(2026, 5, 1))
        assert 'DATE(_PARTITIONTIME) >= @since' in sql

    def test_nets_credits_off_cost(self) -> None:
        sql, _ = run_usage_query(TABLE, run='r', since=date(2026, 5, 1))
        assert 'SUM(cost) + SUM(credit)' in sql

    def test_excludes_unlabelled_rows(self) -> None:
        """Unrelated workloads share this table and have no step label."""
        sql, _ = run_usage_query(TABLE, run='r', since=date(2026, 5, 1))
        assert 'step IS NOT NULL' in sql

    def test_filters_on_run(self) -> None:
        sql, _ = run_usage_query(TABLE, run='r', since=date(2026, 5, 1))
        assert 'run = @run' in sql

    def test_credit_subquery_nets_off_credits(self) -> None:
        sql, _ = run_usage_query(TABLE, run='r', since=date(2026, 5, 1))
        assert '(SELECT IFNULL(SUM(c.amount), 0) FROM UNNEST(credits) c) AS credit' in sql

    def test_filters_on_pipeline_tools(self) -> None:
        sql, _ = run_usage_query(TABLE, run='r', since=date(2026, 5, 1))
        assert "tool IN ('pis', 'pts', 'gentropy')" in sql

    def test_excludes_non_regular_cost_rows(self) -> None:
        """Taxes, adjustments and rounding rows share the export and are not step cost."""
        sql, _ = run_usage_query(TABLE, run='r', since=date(2026, 5, 1))
        assert "cost_type = 'regular'" in sql


def _label_aliases(sql: str) -> dict[str, str]:
    """Map each alias the CTE defines to the `labels` key it is extracted from."""
    pattern = r"\(SELECT value FROM UNNEST\(labels\) WHERE key = '([^']*)'\) AS (\w+)"
    return {alias: key for key, alias in re.findall(pattern, sql)}


def _aggregate_expressions(sql: str) -> dict[str, str]:
    """Map each column the aggregate SELECT produces to the expression behind it.

    A bare column maps to itself, so `currency` selected directly and `currency`
    collapsed with `ANY_VALUE` are distinguishable. Takes the aggregate on its own, not
    a full query: the CTEs select `FROM labelled` too.
    """
    select_block = sql.split('FROM labelled', maxsplit=1)[0]
    expressions = {}
    for line in select_block.splitlines():
        entry = line.strip().rstrip(',')
        if not entry or entry == 'SELECT':
            continue
        aliased = re.match(r'^(?P<expr>.+?)\s+AS\s+(?P<alias>\w+)$', entry)
        if aliased:
            expressions[aliased.group('alias')] = aliased.group('expr')
        else:
            expressions[entry] = entry
    return expressions


def _group_by_columns(sql: str) -> set[str]:
    """The columns the aggregate groups by.

    The last `GROUP BY` in the text, because `cluster_steps` has one of its own and it
    comes first.
    """
    clauses = re.findall(r'^GROUP BY (.+)$', sql, flags=re.MULTILINE)
    assert clauses, 'the aggregate must have a GROUP BY'
    return {column.strip() for column in clauses[-1].split(',')}


def _selected_columns(sql: str) -> set[str]:
    """Column names the aggregate SELECT produces, aliased or bare."""
    return set(_aggregate_expressions(sql))


class TestLabelExtraction:
    """The CTE decides what every row of every report *is*.

    Swapping two label keys transposes each row's identity, and mistyping one makes its
    filter match nothing, which reads as "no billed usage" and exits 0. Neither shows up
    in a mocked fetch, because the row fakes answer to any attribute name.
    """

    def test_each_label_is_extracted_from_the_key_of_the_same_name(self) -> None:
        assert _label_aliases(_LABELLED_CTE) == {
            'run': 'run',
            'step': 'step',
            'tool': 'tool',
            'product': 'product',
            'cluster': 'goog-dataproc-cluster-name',
        }

    def test_the_extractor_actually_finds_the_keys(self) -> None:
        """Guard the guard: a parser that returned nothing would assert nothing."""
        sample = "(SELECT value FROM UNNEST(labels) WHERE key = 'a') AS b"
        assert _label_aliases(sample) == {'b': 'a'}
        assert _label_aliases('SELECT 1') == {}

    def test_the_product_label_is_read(self) -> None:
        """PPP runs the same DAG and can carry the same run label.

        Without this the two products' rows are summed into one, roughly doubling every
        figure with nothing on screen to say so.
        """
        assert _label_aliases(_LABELLED_CTE)['product'] == 'product'


class TestAggregate:
    def test_started_is_the_earliest_start_and_ended_the_latest_end(self) -> None:
        """Transposed, these invert the window and make every span negative."""
        expressions = _aggregate_expressions(_AGGREGATE)
        assert expressions['started'] == 'MIN(usage_start_time)'
        assert expressions['ended'] == 'MAX(usage_end_time)'

    def test_span_runs_from_the_earliest_start_to_the_latest_end(self) -> None:
        expressions = _aggregate_expressions(_AGGREGATE)
        assert expressions['span_hours'] == (
            'TIMESTAMP_DIFF(MAX(usage_end_time), MIN(usage_start_time), SECOND) / 3600'
        )

    def test_the_expression_parser_reads_both_forms(self) -> None:
        """Guard the guard: bare columns and aliased expressions must both be seen."""
        parsed = _aggregate_expressions('SELECT\n  a,\n  MIN(x) AS b\nFROM labelled')
        assert parsed == {'a': 'a', 'b': 'MIN(x)'}

    def test_the_ordering_is_total(self) -> None:
        """`started` alone ties constantly: the export buckets by the hour.

        Steps that began in the same hour then come back in whatever order the plan
        happened to produce, so the same run rendered twice comes out in two different
        orders. The tie-break is what makes two runs of the report diffable.
        """
        assert 'ORDER BY started, run, step' in _AGGREGATE

    def test_currency_is_grouped_rather_than_collapsed(self) -> None:
        """`ANY_VALUE(currency)` over a mixed group stamps one currency on a mixed sum."""
        assert _aggregate_expressions(_AGGREGATE)['currency'] == 'currency'
        assert 'currency' in _group_by_columns(_AGGREGATE)

    def test_product_is_grouped(self) -> None:
        assert 'product' in _group_by_columns(_AGGREGATE)

    def test_every_ungrouped_column_is_an_aggregate(self) -> None:
        """Anything selected bare and not grouped is summed across, silently."""
        bare = {
            column
            for column, expression in _aggregate_expressions(_AGGREGATE).items()
            if column == expression
        }
        assert bare == _group_by_columns(_AGGREGATE)


def _cluster_steps_cte(sql: str) -> str:
    """The body of the `cluster_steps` CTE, which counts the steps sharing a cluster."""
    block = re.search(r'cluster_steps AS \((.*?)\n\)', sql, flags=re.DOTALL)
    assert block is not None, 'the sharing count must live in its own CTE'
    return block.group(1)


class TestClusterSharing:
    """Dataproc clusters are created with `use_if_exists=True` and never relabelled.

    A cluster serving several steps bills all its hours to whichever step created it, so
    those rows are not the step's own cost. Detected and flagged, never re-apportioned.
    """

    def test_the_cluster_name_label_is_read(self) -> None:
        assert _label_aliases(_LABELLED_CTE)['cluster'] == 'goog-dataproc-cluster-name'

    def test_the_count_is_taken_before_the_predicate_is_applied(self) -> None:
        """This is the whole feature.

        Counting a cluster's steps after `step = @step` would see exactly one step every
        time, so no history would ever report a shared cluster and the flag would be
        uniformly false with nothing to show it had failed.
        """
        for sql, _ in (
            run_usage_query(TABLE, run='r', since=date(2026, 5, 1)),
            step_history_query(TABLE, step='pts_target', since=date(2026, 5, 1)),
        ):
            cte = _cluster_steps_cte(sql)
            assert '@run' not in cte
            assert '@step' not in cte
            assert 'COUNT(DISTINCT step)' in cte
            assert 'GROUP BY run, cluster' in cte

    def test_both_queries_count_sharing_identically(self) -> None:
        """A history and a run report of the same rows must not disagree on the flag."""
        usage_sql, _ = run_usage_query(TABLE, run='r', since=date(2026, 5, 1))
        history_sql, _ = step_history_query(TABLE, step='s', since=date(2026, 5, 1))
        assert _cluster_steps_cte(usage_sql) == _cluster_steps_cte(history_sql)

    def test_the_flag_needs_more_than_one_step(self) -> None:
        """Without the `> 1`, every step on any cluster is flagged and the mark is noise."""
        assert _aggregate_expressions(_AGGREGATE)['shared_cluster'] == (
            'IFNULL(LOGICAL_OR(steps_on_cluster > 1), FALSE)'
        )

    def test_the_count_is_joined_on_the_run_and_the_cluster(self) -> None:
        """Joining on the cluster alone would flag a run for another run's sharing."""
        assert 'LEFT JOIN cluster_steps ON run = cluster_run AND cluster = cluster_name' in _AGGREGATE

    def test_a_step_on_no_cluster_is_not_flagged(self) -> None:
        """GCE steps join to nothing, and a null must read as unshared rather than unknown."""
        assert 'IFNULL' in _aggregate_expressions(_AGGREGATE)['shared_cluster']
        assert 'LEFT JOIN' in _AGGREGATE

    def test_the_cte_extractor_actually_finds_the_block(self) -> None:
        """Guard the guard: a parser that matched nothing would assert nothing."""
        sample = 'WITH a AS (\n  SELECT 1\n),\ncluster_steps AS (\n  SELECT 2\n)\n'
        assert _cluster_steps_cte(sample).strip() == 'SELECT 2'


class TestSelectMatchesTheModel:
    def test_every_selected_column_is_a_model_field(self) -> None:
        """`_fetch` reads row attributes by name, and the row fakes answer to anything.

        A renamed SQL alias would therefore pass every mocked test and only fail
        against the real export, so pin the two names lists against each other here.
        """
        assert _selected_columns(_AGGREGATE) == set(StepUsage.model_fields)

    def test_the_extractor_actually_finds_the_aliases(self) -> None:
        """Guard the guard: a parser that returned nothing would assert nothing."""
        assert 'net_cost' in _selected_columns(_AGGREGATE)
        assert _selected_columns('SELECT\n  a,\n  MIN(x) AS b\nFROM labelled') == {'a', 'b'}


class TestStepHistoryQuery:
    def test_binds_step_and_since(self) -> None:
        sql, params = step_history_query(TABLE, step='pts_target', since=date(2026, 6, 1))
        assert 'pts_target' not in sql
        names = {p.name: p.value for p in params}
        assert names['step'] == 'pts_target'
        assert names['since'] == date(2026, 6, 1)

    def test_groups_by_every_identity_column(self) -> None:
        sql, _ = step_history_query(TABLE, step='pts_target', since=date(2026, 6, 1))
        assert _group_by_columns(sql) == {'run', 'step', 'tool', 'product', 'currency'}

    def test_filters_on_step(self) -> None:
        sql, _ = step_history_query(TABLE, step='pts_target', since=date(2026, 6, 1))
        assert 'step = @step' in sql


class TestWindowCoverageQuery:
    def test_binds_the_window_and_the_partition_floor(self) -> None:
        sql, params = window_coverage_query(TABLE, WINDOW, since=date(2026, 5, 1))
        assert '2026-07-21' not in sql
        names = {p.name: p.value for p in params}
        assert names['window_start'] == WINDOW[0]
        assert names['window_end'] == WINDOW[1]
        assert names['since'] == date(2026, 5, 1)

    def test_still_prunes_partitions(self) -> None:
        """The window is on usage time, which is not the partitioning column."""
        sql, _ = window_coverage_query(TABLE, WINDOW, since=date(2026, 5, 1))
        assert 'DATE(_PARTITIONTIME) >= @since' in sql

    def test_denominator_is_every_row_the_pipeline_created(self) -> None:
        """Unlabelled rows are the point of the measurement, so they cannot be filtered out."""
        sql, _ = window_coverage_query(TABLE, WINDOW, since=date(2026, 5, 1))
        assert "key = 'created_by') IN ('unified-pipeline', 'gentropy-pipelines')" in sql
        assert 'WHERE step IS NOT NULL' not in sql

    def test_numerator_matches_what_the_report_shows(self) -> None:
        sql, _ = window_coverage_query(TABLE, WINDOW, since=date(2026, 5, 1))
        assert "IF(step IS NOT NULL AND tool IN ('pis', 'pts', 'gentropy'), net_cost, 0)" in sql

    def test_totals_are_kept_per_currency(self) -> None:
        sql, _ = window_coverage_query(TABLE, WINDOW, since=date(2026, 5, 1))
        assert 'GROUP BY currency' in sql


class TestWindowCoverage:
    def test_share_is_the_labelled_fraction(self) -> None:
        coverage = WindowCoverage(currency='GBP', labelled_cost=75.0, pipeline_cost=100.0)
        assert coverage.labelled_share == pytest.approx(0.75)

    def test_share_is_none_when_nothing_was_billed(self) -> None:
        """Zero over zero is not 100% coverage, and must not be rendered as one."""
        coverage = WindowCoverage(currency='GBP', labelled_cost=0.0, pipeline_cost=0.0)
        assert coverage.labelled_share is None


class TestUsageWindow:
    def test_spans_the_earliest_start_to_the_latest_end(self) -> None:
        usages = [
            StepUsage.model_validate(vars(_row(
                started=datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
                ended=datetime(2026, 7, 21, 15, 0, tzinfo=UTC),
            ))),
            StepUsage.model_validate(vars(_row(
                started=datetime(2026, 7, 21, 18, 0, tzinfo=UTC),
                ended=datetime(2026, 7, 21, 20, 0, tzinfo=UTC),
            ))),
        ]
        assert usage_window(usages) == (
            datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
            datetime(2026, 7, 21, 20, 0, tzinfo=UTC),
        )

    def test_empty_has_no_window(self) -> None:
        """A run that has billed nothing has no period to measure coverage over."""
        assert usage_window([]) is None


def _row(**kw: object) -> SimpleNamespace:
    """Stand in for a bigquery.Row, which supports attribute access."""
    base = {
        'run': 'r',
        'step': 'pts_target',
        'tool': 'pts',
        'product': 'platform',
        'started': datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
        'ended': datetime(2026, 7, 21, 16, 0, tzinfo=UTC),
        'span_hours': 2.0,
        'net_cost': 1.5,
        'currency': 'GBP',
        'shared_cluster': False,
    }
    base.update(kw)
    return SimpleNamespace(**base)


def _client(rows: list[SimpleNamespace]) -> MagicMock:
    client = MagicMock()
    client.query.return_value.result.return_value = rows
    return client


class TestBillingExport:
    def test_maps_rows_to_models(self) -> None:
        export = BillingExport(client=_client([_row()]), table=TABLE)
        result = export.run_usage(run='r', since=date(2026, 5, 1))
        assert len(result) == 1
        assert isinstance(result[0], StepUsage)
        assert result[0] == StepUsage.model_validate(vars(_row()))

    def test_passes_parameters_to_bigquery(self) -> None:
        client = _client([])
        export = BillingExport(client=client, table=TABLE)
        export.run_usage(run='my-run', since=date(2026, 5, 1))
        job_config = client.query.call_args.kwargs['job_config']
        assert {p.name for p in job_config.query_parameters} == {'run', 'since'}

    def test_scan_is_capped(self) -> None:
        """Uncapped, a query that lost its partition filter bills the whole export."""
        client = _client([])
        export = BillingExport(client=client, table=TABLE)
        export.run_usage(run='r', since=date(2026, 5, 1))
        job_config = client.query.call_args.kwargs['job_config']
        assert job_config.maximum_bytes_billed == MAX_BYTES_BILLED

    def test_empty_result_is_not_an_error(self) -> None:
        """A run with no billed rows yet is normal early in a pipeline run."""
        export = BillingExport(client=_client([]), table=TABLE)
        assert export.run_usage(run='r', since=date(2026, 5, 1)) == []

    def test_step_history_uses_the_history_query(self) -> None:
        client = _client([_row(run='older-run')])
        export = BillingExport(client=client, table=TABLE)
        result = export.step_history(step='pts_target', since=date(2026, 5, 1))
        assert result[0].run == 'older-run'
        job_config = client.query.call_args.kwargs['job_config']
        assert {p.name for p in job_config.query_parameters} == {'step', 'since'}

    def test_window_coverage_maps_rows_to_models(self) -> None:
        rows = [SimpleNamespace(currency='GBP', labelled_cost=3170.71, pipeline_cost=4232.82)]
        export = BillingExport(client=_client(rows), table=TABLE)
        coverage = export.window_coverage(window=WINDOW, since=date(2026, 5, 1))
        assert coverage == [
            WindowCoverage(currency='GBP', labelled_cost=3170.71, pipeline_cost=4232.82)
        ]

    def test_window_coverage_binds_the_window(self) -> None:
        client = _client([])
        export = BillingExport(client=client, table=TABLE)
        export.window_coverage(window=WINDOW, since=date(2026, 5, 1))
        job_config = client.query.call_args.kwargs['job_config']
        assert {p.name for p in job_config.query_parameters} == {
            'window_start',
            'window_end',
            'since',
        }
        assert job_config.maximum_bytes_billed == MAX_BYTES_BILLED


class TestTotalCost:
    def test_sums_net_cost(self) -> None:
        usages = [
            StepUsage.model_validate(vars(_row(net_cost=1.5))),
            StepUsage.model_validate(vars(_row(net_cost=2.25))),
        ]
        assert total_cost(usages) == pytest.approx(3.75)

    def test_empty_is_zero(self) -> None:
        assert total_cost([]) == 0.0
