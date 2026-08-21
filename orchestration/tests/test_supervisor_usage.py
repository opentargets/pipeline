"""Tests for the supervisor usage module."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from orchestration.supervisor.usage import StepUsage


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
