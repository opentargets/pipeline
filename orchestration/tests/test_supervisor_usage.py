"""Tests for the supervisor usage module."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import get_args
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from orchestration.supervisor.usage import (
    _AGGREGATE,
    _LABELLED_CTE,
    MAX_BYTES_BILLED,
    BillingExport,
    PipelineCreator,
    StepUsage,
    WindowCoverage,
    run_usage_query,
    step_history_query,
    total_cost,
    usage_window,
    window_coverage_query,
)
from orchestration.utils.common import GCP_PROJECT_GENETICS, GCP_PROJECT_PLATFORM
from orchestration.utils.labels import default_labels

TABLE = 'proj.ds.tbl'
RUN = 'manual__2026-07-21t15-07-47-545737-00-00'
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
            billed_hours=2,
            net_cost=1.5,
            currency='GBP',
            shared_cluster=False,
            core_seconds=7200.0,
            spot_core_seconds=0.0,
            machine_families=['N1'],
        )
        assert u.step == 'pts_target'
        assert u.product == 'platform'
        assert u.billed_hours == 2

    def test_tool_must_be_a_pipeline_tool(self) -> None:
        with pytest.raises(ValidationError):
            StepUsage.model_validate({
                'run': 'r',
                'step': 's',
                'tool': 'nextgen',
                'product': 'platform',
                'started': datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
                'ended': datetime(2026, 7, 21, 15, 0, tzinfo=UTC),
                'billed_hours': 1,
                'net_cost': 0.1,
                'currency': 'GBP',
                'shared_cluster': False,
                'core_seconds': None,
                'spot_core_seconds': None,
                'machine_families': [],
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
            'billed_hours': 1,
            'net_cost': 0.1,
            'currency': 'GBP',
            'shared_cluster': False,
            'core_seconds': None,
            'spot_core_seconds': None,
            'machine_families': [],
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
            billed_hours=1,
            net_cost=-0.02,
            currency='GBP',
            shared_cluster=False,
            core_seconds=None,
            spot_core_seconds=None,
            machine_families=[],
        )
        assert u.net_cost < 0

    def test_core_seconds_and_spot_core_seconds_may_both_be_absent(self) -> None:
        """A Batch step, or one that is storage/network cost only, billed no core SKU.

        `None` here must not collapse to `0.0` — that would claim measured zero CPU
        rather than "this figure cannot see this step's CPU at all".
        """
        u = StepUsage(
            run='r',
            step='s',
            tool='pis',
            product='platform',
            started=datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
            ended=datetime(2026, 7, 21, 15, 0, tzinfo=UTC),
            billed_hours=1,
            net_cost=0.01,
            currency='GBP',
            shared_cluster=False,
            core_seconds=None,
            spot_core_seconds=None,
            machine_families=[],
        )
        assert u.core_seconds is None
        assert u.spot_core_seconds is None
        assert u.machine_families == []

    def test_spot_core_seconds_can_be_a_measured_zero(self) -> None:
        """Core time billed and none of it spot is a fact, distinct from `None`."""
        u = StepUsage(
            run='r',
            step='s',
            tool='pis',
            product='platform',
            started=datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
            ended=datetime(2026, 7, 21, 15, 0, tzinfo=UTC),
            billed_hours=1,
            net_cost=0.01,
            currency='GBP',
            shared_cluster=False,
            core_seconds=3600.0,
            spot_core_seconds=0.0,
            machine_families=['N1'],
        )
        assert u.core_seconds == 3600.0
        assert u.spot_core_seconds == 0.0


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
        """Unrelated workloads share this table and have no step label.

        Asserted against `_AGGREGATE` rather than the assembled query: `cluster_steps`
        contains the same string, so against the whole SQL this passes even with the
        aggregate's own `WHERE` deleted.
        """
        assert 'WHERE step IS NOT NULL' in _AGGREGATE

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


_STRUCTURAL = ('SELECT', 'FROM', 'WHERE', 'AND', 'GROUP BY', 'ORDER BY', 'WITH', 'LEFT JOIN', ')')
"""Line starts that are SQL structure rather than a projected column."""


def _select_expressions(sql: str, until: str) -> dict[str, str]:
    """Map each column a SELECT projects to the expression behind it, reading up to `until`.

    A bare column maps to itself, so `currency` selected directly and `currency` collapsed
    with `ANY_VALUE` are distinguishable. `until` bounds the block, because a query has
    several SELECTs and they project different things: it is read back from `until` to the
    nearest unindented `SELECT`, which is the one projecting those columns. A CTE's own
    `SELECT` is indented, so it never captures the block of an outer one.
    """
    head = sql.split(until, maxsplit=1)[0]
    expressions = {}
    for line in head.rsplit('\nSELECT\n', maxsplit=1)[-1].splitlines():
        entry = line.strip().rstrip(',')
        if not entry or entry.startswith(_STRUCTURAL):
            continue
        aliased = re.match(r'^(?P<expr>.+?)\s+AS\s+(?P<alias>\w+)$', entry)
        if aliased:
            expressions[aliased.group('alias')] = aliased.group('expr')
        else:
            expressions[entry] = entry
    return expressions


def _aggregate_expressions(sql: str) -> dict[str, str]:
    """What the aggregate SELECT produces, column by column.

    Takes the aggregate on its own, not a full query: the CTEs select `FROM labelled` too.
    """
    return _select_expressions(sql, 'FROM labelled')


def _group_by_columns(sql: str) -> set[str]:
    """The columns the aggregate groups by.

    The last `GROUP BY` at column zero. `cluster_steps` has a clause of its own, but it is
    indented and `^` under `MULTILINE` does not reach it, so today the last is also the
    only one. Taking the last rather than the first survives that indentation changing.
    """
    clauses = re.findall(r'^GROUP BY (.+)$', sql, flags=re.MULTILINE)
    assert clauses, 'the aggregate must have a GROUP BY'
    return {column.strip() for column in clauses[-1].split(',')}


def _selected_columns(sql: str) -> set[str]:
    """Column names the aggregate SELECT produces, aliased or bare."""
    return set(_aggregate_expressions(sql))


class TestPipelineCreatorTracksTheLabeller:
    def test_every_created_by_the_pipeline_stamps_is_a_pipeline_creator(self) -> None:
        """`PipelineCreator` duplicates strings owned by `utils/labels.default_labels`.

        Hardcoding them in the test too would leave three copies and no link between any
        of them, so a renamed label would silently empty the coverage denominator. Read
        them from the labeller instead.
        """
        stamped = {
            default_labels(project=project)['created_by']
            for project in (GCP_PROJECT_PLATFORM, GCP_PROJECT_GENETICS)
        }
        assert stamped <= set(get_args(PipelineCreator))


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
            'cluster_instance': 'goog-dataproc-cluster-uuid',
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


class TestCoreSkuExtraction:
    """The row-level flags `core_seconds`/`spot_core_seconds`/`machine_families` are built from.

    Verified live (2026-08-24) against the two real core SKU descriptions in the export:
    `N1 Predefined Instance Core running in EMEA` and
    `Spot Preemptible N1 Predefined Instance Core running in EMEA`. Both contain the literal
    substring `Instance Core`; only the second starts with `Spot `; both yield `N1` from the
    family regex.
    """

    def test_the_core_flag_matches_on_the_instance_core_substring(self) -> None:
        assert "sku.description LIKE '%Instance Core%' AS is_core" in _LABELLED_CTE

    def test_the_spot_flag_matches_the_spot_prefix_not_preemptible_generally(self) -> None:
        """`STARTS_WITH`, not `LIKE '%Spot%'`.

        A description mentioning spot mid-string (there are none today, but a future SKU
        rename could add one) must not be misread as pricing this row spot.
        """
        assert "STARTS_WITH(sku.description, 'Spot ') AS is_spot" in _LABELLED_CTE

    def test_usage_amount_is_carried_through_unconverted(self) -> None:
        """Core SKUs bill `usage.amount` in seconds already; no unit conversion belongs here."""
        assert 'usage.amount AS usage_amount' in _LABELLED_CTE

    def test_machine_family_is_null_for_non_core_rows(self) -> None:
        """A RAM or disk row must not contribute a family — only core rows name one."""
        match = re.search(
            r'IF\(\s*sku\.description LIKE \'%Instance Core%\',\s*'
            r"REGEXP_EXTRACT\(sku\.description, r'([^']*)'\),\s*NULL\s*\) AS machine_family",
            _LABELLED_CTE,
        )
        assert match is not None, 'the machine_family expression must guard on is-core'

    def test_the_family_regex_reads_the_token_before_instance_core(self) -> None:
        """Pinned against the two real descriptions rather than reasoned about only.

        `Spot Preemptible N1 Predefined Instance Core running in EMEA` -> `N1`: the
        `Spot Preemptible ` prefix sits before the captured token, not inside it.
        """
        pattern = re.search(r"REGEXP_EXTRACT\(sku\.description, r'([^']*)'\)", _LABELLED_CTE)
        assert pattern is not None
        family_re = re.compile(pattern.group(1))

        on_demand = family_re.search('N1 Predefined Instance Core running in EMEA')
        assert on_demand is not None
        assert on_demand.group(1) == 'N1'

        spot = family_re.search('Spot Preemptible N1 Predefined Instance Core running in EMEA')
        assert spot is not None
        assert spot.group(1) == 'N1'


class TestAggregate:
    def test_started_is_the_earliest_start_and_ended_the_latest_end(self) -> None:
        """The only guard on these two columns, and both are load-bearing.

        `started` orders the rows and, with `ended`, bounds the window `window_coverage`
        measures. Transposed, the query still runs and still returns numbers: the window
        inverts, and the coverage denominator silently covers a period no run occupied.
        """
        expressions = _aggregate_expressions(_AGGREGATE)
        assert expressions['started'] == 'MIN(usage_start_time)'
        assert expressions['ended'] == 'MAX(usage_end_time)'

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
        assert 'ORDER BY started, run, step, tool, product, currency' in _AGGREGATE

    def test_currency_is_grouped_rather_than_collapsed(self) -> None:
        """`ANY_VALUE(currency)` over a mixed group stamps one currency on a mixed sum."""
        assert _aggregate_expressions(_AGGREGATE)['currency'] == 'currency'
        assert 'currency' in _group_by_columns(_AGGREGATE)

    def test_billed_hours_counts_distinct_hourly_buckets_not_the_row_count(self) -> None:
        """`COUNT(*)` would count rows (several SKUs per hour); this counts distinct hours.

        That distinction is the whole feature: a step billed across several SKUs in one
        hour must still report 1, not one per SKU.
        """
        assert _aggregate_expressions(_AGGREGATE)['billed_hours'] == (
            'COUNT(DISTINCT TIMESTAMP_TRUNC(usage_start_time, HOUR))'
        )

    def test_core_seconds_is_null_not_zero_when_the_step_billed_no_core_sku(self) -> None:
        """`SUM(IF(is_core, usage_amount, 0))` alone would report 0.0 for a Batch step.

        The `COUNTIF(is_core) = 0` guard is what turns "no core rows" into `NULL` rather
        than a summed zero that reads as "measured zero CPU".
        """
        assert _aggregate_expressions(_AGGREGATE)['core_seconds'] == (
            'IF(COUNTIF(is_core) = 0, NULL, SUM(IF(is_core, usage_amount, 0)))'
        )

    def test_spot_core_seconds_shares_the_same_null_guard_as_core_seconds(self) -> None:
        """Both must go absent together: a step with core rows but no spot rows is 0.0."""
        assert _aggregate_expressions(_AGGREGATE)['spot_core_seconds'] == (
            'IF(COUNTIF(is_core) = 0, NULL, SUM(IF(is_core AND is_spot, usage_amount, 0)))'
        )

    def test_machine_families_are_deduplicated_ordered_and_drop_the_non_core_nulls(self) -> None:
        assert _aggregate_expressions(_AGGREGATE)['machine_families'] == (
            'ARRAY_AGG(DISTINCT machine_family IGNORE NULLS ORDER BY machine_family)'
        )

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
    """A cluster instance outliving its creating step would bill under that step's labels.

    That is what `use_if_exists=True` in `operators/dataproc.py` permits, and the flag is
    a guard on it. The path does not manifest today. Grouping must be on
    the instance uuid: cluster *names* are reused several times within a run, each reuse a
    separate instance carrying its own creating step's label, so grouping on the name
    reports correct rows as suspect.
    """

    def test_the_cluster_instance_is_identified_by_uuid_not_name(self) -> None:
        """Measured 2026-08-21: one name covered 6 steps across 12 separate instances.

        By name that reads as sharing. By uuid every one of the 181 instances in the
        export served exactly one step, which is the truth.
        """
        aliases = _label_aliases(_LABELLED_CTE)
        assert aliases['cluster_instance'] == 'goog-dataproc-cluster-uuid'
        assert 'goog-dataproc-cluster-name' not in aliases.values()

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
            assert 'GROUP BY run, cluster_instance' in cte

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
        assert (
            'LEFT JOIN cluster_steps ON run = cluster_run AND cluster_instance = cluster_key'
        ) in _AGGREGATE

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


def _coverage_sources(sql: str) -> dict[str, str]:
    """What the coverage query's source CTE projects, column by column."""
    return _select_expressions(sql, 'FROM `')


def _coverage_outputs(sql: str) -> dict[str, str]:
    """What the coverage query returns, column by column."""
    return _select_expressions(sql, 'FROM windowed')


def _window_filters(sql: str) -> list[str]:
    """The conditions the coverage query's source CTE keeps a row on, in order."""
    lines = [line.strip() for line in sql.split('FROM `', maxsplit=1)[1].splitlines()]
    return [
        entry.split(' ', maxsplit=1)[1]
        for entry in lines
        if entry.startswith(('WHERE ', 'AND '))
    ]


class TestWindowCoverageQuery:
    """Pinned the way `_AGGREGATE` is, and for the same reason.

    Every one of these was a substring-presence check, and five semantic mutations of this
    query survived the whole suite: netting credits the wrong way, either window
    comparison flipped, the two output aliases transposed, and the `cost_type` filter
    deleted. All five are silently wrong money rather than a failure.
    """

    def test_binds_the_run_the_window_and_the_partition_floor(self) -> None:
        sql, params = window_coverage_query(TABLE, RUN, WINDOW, since=date(2026, 5, 1))
        assert '2026-07-21' not in sql
        assert RUN not in sql
        names = {p.name: p.value for p in params}
        assert names['run'] == RUN
        assert names['window_start'] == WINDOW[0]
        assert names['window_end'] == WINDOW[1]
        assert names['since'] == date(2026, 5, 1)

    def test_the_row_filter_is_exactly_these_four_conditions(self) -> None:
        """Pins each comparison with its operator.

        `usage_start_time <= @window_start` and `usage_end_time >= @window_end` are both
        valid SQL that quietly measure a different set of rows, and dropping `cost_type`
        folds taxes and adjustments into the denominator.
        """
        sql, _ = window_coverage_query(TABLE, RUN, WINDOW, since=date(2026, 5, 1))
        assert _window_filters(sql) == [
            'DATE(_PARTITIONTIME) >= @since',
            "cost_type = 'regular'",
            'usage_start_time >= @window_start',
            'usage_end_time <= @window_end',
        ]

    def test_credits_are_netted_off_not_added_on(self) -> None:
        """Credits are negative, so `cost - SUM(amount)` inflates both figures instead."""
        sql, _ = window_coverage_query(TABLE, RUN, WINDOW, since=date(2026, 5, 1))
        assert _coverage_sources(sql)['net_cost'] == (
            'cost + (SELECT IFNULL(SUM(c.amount), 0) FROM UNNEST(credits) c)'
        )

    def test_the_numerator_is_the_rows_the_report_shows(self) -> None:
        """Carries the same three predicates as the aggregate.

        So it is the table's total by construction, not a number that happens to be close.
        """
        sql, _ = window_coverage_query(TABLE, RUN, WINDOW, since=date(2026, 5, 1))
        assert _coverage_outputs(sql)['labelled_cost'] == (
            "SUM(IF(run = @run AND step IS NOT NULL AND tool IN ('pis', 'pts', 'gentropy'), net_cost, 0))"
        )

    def test_the_denominator_is_all_pipeline_spend_in_the_window(self) -> None:
        """Not filtered by run: an overlapping run's spend is real spend in the window."""
        sql, _ = window_coverage_query(TABLE, RUN, WINDOW, since=date(2026, 5, 1))
        assert _coverage_outputs(sql)['pipeline_cost'] == (
            "SUM(IF(created_by IN ('unified-pipeline', 'gentropy-pipelines'), net_cost, 0))"
        )
        assert '@run' not in _coverage_outputs(sql)['pipeline_cost']

    def test_the_two_aliases_are_not_transposed(self) -> None:
        """Transposed, this prints '3170.71 of the 1062.11 GBP (298.5%)'."""
        outputs = _coverage_outputs(sql=window_coverage_query(TABLE, RUN, WINDOW, date(2026, 5, 1))[0])
        assert 'step IS NOT NULL' in outputs['labelled_cost']
        assert 'created_by' in outputs['pipeline_cost']
        assert 'step IS NOT NULL' not in outputs['pipeline_cost']

    def test_the_numerator_does_not_require_created_by(self) -> None:
        """The table's rows are not filtered on it, so the numerator must not be either.

        Otherwise a step-labelled row missing `created_by` appears in the table but not in
        the figure that claims to total the table.
        """
        sql, _ = window_coverage_query(TABLE, RUN, WINDOW, since=date(2026, 5, 1))
        assert 'created_by' not in _coverage_outputs(sql)['labelled_cost']

    def test_totals_are_kept_per_currency(self) -> None:
        sql, _ = window_coverage_query(TABLE, RUN, WINDOW, since=date(2026, 5, 1))
        assert _group_by_columns(sql) == {'currency'}

    def test_the_parsers_actually_read_this_query(self) -> None:
        """Guard the guards: three parsers here, each worthless if it matches nothing."""
        sql, _ = window_coverage_query(TABLE, RUN, WINDOW, since=date(2026, 5, 1))
        assert set(_coverage_sources(sql)) == {
            'run',
            'step',
            'tool',
            'created_by',
            'net_cost',
            'currency',
        }
        assert set(_coverage_outputs(sql)) == {'currency', 'labelled_cost', 'pipeline_cost'}
        assert len(_window_filters(sql)) == 4


class TestWindowCoverage:
    def test_share_is_the_labelled_fraction(self) -> None:
        coverage = WindowCoverage(currency='GBP', labelled_cost=75.0, pipeline_cost=100.0)
        assert coverage.labelled_share == pytest.approx(0.75)

    def test_share_is_none_when_nothing_was_billed(self) -> None:
        """Zero over zero is not 100% coverage, and must not be rendered as one."""
        coverage = WindowCoverage(currency='GBP', labelled_cost=0.0, pipeline_cost=0.0)
        assert coverage.labelled_share is None

    def test_a_numerator_above_the_denominator_is_flagged_as_impossible(self) -> None:
        """A share above 100% is a broken measurement, not excellent coverage."""
        coverage = WindowCoverage(currency='GBP', labelled_cost=120.0, pipeline_cost=100.0)
        assert coverage.exceeds_pipeline_cost
        assert coverage.labelled_share == pytest.approx(1.2)

    def test_a_normal_share_is_not_flagged(self) -> None:
        assert not WindowCoverage(
            currency='GBP', labelled_cost=75.0, pipeline_cost=100.0
        ).exceeds_pipeline_cost


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
        'billed_hours': 2,
        'net_cost': 1.5,
        'currency': 'GBP',
        'shared_cluster': False,
        'core_seconds': 7200.0,
        'spot_core_seconds': 0.0,
        'machine_families': ['N1'],
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

    def test_a_step_with_no_core_sku_maps_to_none_not_zero(self) -> None:
        """BigQuery's own `NULL` for `core_seconds`/`spot_core_seconds` must reach the model as `None`.

        A row-mapping bug that defaulted a missing value to `0.0` would pass every other
        test here, because every other fixture row has core rows.
        """
        row = _row(core_seconds=None, spot_core_seconds=None, machine_families=[])
        export = BillingExport(client=_client([row]), table=TABLE)
        result = export.run_usage(run='r', since=date(2026, 5, 1))
        assert result[0].core_seconds is None
        assert result[0].spot_core_seconds is None
        assert result[0].machine_families == []

    def test_machine_families_from_the_row_become_a_plain_list(self) -> None:
        """BigQuery returns a repeated field as its own array type; the model wants `list[str]`."""
        row = _row(machine_families=('N1',))
        export = BillingExport(client=_client([row]), table=TABLE)
        result = export.run_usage(run='r', since=date(2026, 5, 1))
        assert result[0].machine_families == ['N1']
        assert isinstance(result[0].machine_families, list)

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
        coverage = export.window_coverage(run=RUN, window=WINDOW, since=date(2026, 5, 1))
        assert coverage == [
            WindowCoverage(currency='GBP', labelled_cost=3170.71, pipeline_cost=4232.82)
        ]

    def test_window_coverage_binds_the_window(self) -> None:
        client = _client([])
        export = BillingExport(client=client, table=TABLE)
        export.window_coverage(run=RUN, window=WINDOW, since=date(2026, 5, 1))
        job_config = client.query.call_args.kwargs['job_config']
        assert {p.name for p in job_config.query_parameters} == {
            'run',
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
