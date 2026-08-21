"""Tests for the supervisor usage module."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from orchestration.supervisor.usage import StepUsage, run_usage_query, step_history_query

TABLE = 'proj.ds.tbl'


class TestStepUsage:
    def test_construction(self) -> None:
        u = StepUsage(
            run='manual__2026-07-21t15-07-47-545737-00-00',
            step='pts_target',
            tool='pts',
            started=datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
            ended=datetime(2026, 7, 21, 16, 0, tzinfo=UTC),
            span_hours=2.0,
            net_cost=1.5,
            currency='GBP',
        )
        assert u.step == 'pts_target'
        assert u.span_hours == 2.0

    def test_tool_must_be_a_pipeline_tool(self) -> None:
        with pytest.raises(ValidationError):
            StepUsage(
                run='r',
                step='s',
                tool='nextgen',  # ty: ignore[invalid-argument-type]
                started=datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
                ended=datetime(2026, 7, 21, 15, 0, tzinfo=UTC),
                span_hours=1.0,
                net_cost=0.1,
                currency='GBP',
            )

    def test_negative_cost_is_allowed(self) -> None:
        """A credit-dominated row can net negative. This must not be rejected."""
        u = StepUsage(
            run='r',
            step='s',
            tool='pis',
            started=datetime(2026, 7, 21, 14, 0, tzinfo=UTC),
            ended=datetime(2026, 7, 21, 15, 0, tzinfo=UTC),
            span_hours=1.0,
            net_cost=-0.02,
            currency='GBP',
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


class TestStepHistoryQuery:
    def test_binds_step_and_since(self) -> None:
        sql, params = step_history_query(TABLE, step='pts_target', since=date(2026, 6, 1))
        assert 'pts_target' not in sql
        names = {p.name: p.value for p in params}
        assert names['step'] == 'pts_target'
        assert names['since'] == date(2026, 6, 1)

    def test_groups_by_run(self) -> None:
        sql, _ = step_history_query(TABLE, step='pts_target', since=date(2026, 6, 1))
        assert 'GROUP BY run, step, tool' in sql
