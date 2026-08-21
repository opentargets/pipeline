"""Live checks against the real billing export.

Skipped unless RUN_BIGQUERY_TESTS is set, because these need credentials and network.
Run with: RUN_BIGQUERY_TESTS=1 uv run --frozen pytest tests/test_supervisor_usage_live.py -rxs
"""

from __future__ import annotations

import os
from datetime import date

import pytest
from google.cloud import bigquery

from orchestration.supervisor.usage import BillingExport, total_cost

pytestmark = pytest.mark.skipif(
    not os.environ.get('RUN_BIGQUERY_TESTS'),
    reason='needs BigQuery credentials, set RUN_BIGQUERY_TESTS=1 to run',
)

KNOWN_RUN = 'manual__2026-07-21t15-07-47-545737-00-00'
"""A real run verified on 2026-08-21 to carry 18 distinct steps.

Gross cost is 24.82 GBP and net-of-credits is 21.60 GBP. The assertion below uses
the net figure, because that is what `total_cost` returns. Do not "fix" a failure
here by making the code report gross — netting credits is the whole point.
"""


@pytest.fixture
def export() -> BillingExport:
    return BillingExport(client=bigquery.Client(project='open-targets-eu-dev'))


class TestLiveExport:
    def test_known_run_matches_the_verified_figures(self, export: BillingExport) -> None:
        usages = export.run_usage(run=KNOWN_RUN, since=date(2026, 5, 1))
        assert len({u.step for u in usages}) == 18
        assert total_cost(usages) == pytest.approx(21.60, abs=0.01)
        assert {u.currency for u in usages} == {'GBP'}

    def test_only_pipeline_tools_are_returned(self, export: BillingExport) -> None:
        usages = export.run_usage(run=KNOWN_RUN, since=date(2026, 5, 1))
        assert {u.tool for u in usages} <= {'pis', 'pts', 'gentropy'}

    def test_every_span_is_a_whole_number_of_hours(self, export: BillingExport) -> None:
        """Documents the export's hourly bucketing.

        If this ever fails, the quantisation caveat in usage.py needs revisiting.
        """
        usages = export.run_usage(run=KNOWN_RUN, since=date(2026, 5, 1))
        assert all(u.span_hours == int(u.span_hours) for u in usages)
