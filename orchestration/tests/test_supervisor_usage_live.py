"""Live checks against the real billing export.

Skipped unless RUN_BIGQUERY_TESTS is set, because these need credentials and network.
Run with: RUN_BIGQUERY_TESTS=1 uv run --frozen pytest tests/test_supervisor_usage_live.py -rxs
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import date

import pytest
from google.cloud import bigquery

from orchestration.supervisor.usage import (
    MAX_BYTES_BILLED,
    BillingExport,
    run_usage_query,
    step_history_query,
    total_cost,
    usage_window,
    window_coverage_query,
)
from orchestration.utils.common import GCP_PROJECT_PLATFORM

pytestmark = pytest.mark.skipif(
    not os.environ.get('RUN_BIGQUERY_TESTS'),
    reason='needs BigQuery credentials, set RUN_BIGQUERY_TESTS=1 to run',
)

QueryBuilder = Callable[[str, str, date], tuple[str, list[bigquery.ScalarQueryParameter]]]
"""Both label-filtered query builders take (table, label, since) positionally."""

KNOWN_RUN = 'manual__2026-07-21t15-07-47-545737-00-00'
"""A real run verified on 2026-08-21 to carry 18 distinct steps.

Gross cost is 24.82 GBP and net-of-credits is 21.60 GBP. The assertion below uses
the net figure, because that is what `total_cost` returns. Do not "fix" a failure
here by making the code report gross — netting credits is the whole point.
"""

SHARED_STEP = 'pts_literature_ontoma'
"""A step verified on 2026-08-21 to share a Dataproc cluster in some runs and not others.

It had `up-pts-literature-f5014` to itself in KNOWN_RUN, and shared
`up-pts-literature-38bc6` with `pts_literature_embedding` in `2026-06-19_ppp_up` and
`up-pts-literature-5df4f` with `pts_literature_publication_match` in `up-20260527-1458`.
Both halves matter: one pins that sharing is detected across a history, the other that the
flag is not simply stuck on.
"""

FULL_SCAN_CEILING = 4 * 1024**3
"""Upper bound on the bytes a full-retention query may process, in bytes (4 GiB).

Tighter than `MAX_BYTES_BILLED`, which is the runtime guard rail. This is the test's
own tripwire for partition pruning: a query that scanned the whole export unpruned
would still be under the cap today and would sail past an `is not None` assertion,
which is exactly what this used to do.
"""


@pytest.fixture
def export() -> BillingExport:
    return BillingExport(client=bigquery.Client(project=GCP_PROJECT_PLATFORM))


class TestLiveExport:
    def test_known_run_matches_the_verified_figures(self, export: BillingExport) -> None:
        usages = export.run_usage(run=KNOWN_RUN, since=date(2026, 5, 1))
        assert len({u.step for u in usages}) == 18
        assert total_cost(usages) == pytest.approx(21.60, abs=0.01)
        assert {u.currency for u in usages} == {'GBP'}

    def test_only_pipeline_tools_are_returned(self, export: BillingExport) -> None:
        usages = export.run_usage(run=KNOWN_RUN, since=date(2026, 5, 1))
        assert usages, 'an empty result satisfies the subset check vacuously'
        assert {u.tool for u in usages} <= {'pis', 'pts', 'gentropy'}

    def test_every_span_is_a_whole_number_of_hours(self, export: BillingExport) -> None:
        """Documents the export's hourly bucketing.

        If this ever fails, the quantisation caveat in usage.py needs revisiting.
        """
        usages = export.run_usage(run=KNOWN_RUN, since=date(2026, 5, 1))
        assert usages, 'an empty result satisfies the all() check vacuously'
        assert all(u.span_hours == int(u.span_hours) for u in usages)

    def test_the_run_is_one_product(self, export: BillingExport) -> None:
        """A platform run and a PPP run of the same DAG must never be summed together."""
        usages = export.run_usage(run=KNOWN_RUN, since=date(2026, 5, 1))
        assert usages
        assert {u.product for u in usages} == {'platform'}

    def test_the_known_run_shares_no_cluster(self, export: BillingExport) -> None:
        """Verified 2026-08-21: its three Dataproc steps each had a cluster to themselves.

        So this run's per-step costs really are per-step, which is what makes it usable
        as the fixed cost baseline above.
        """
        usages = export.run_usage(run=KNOWN_RUN, since=date(2026, 5, 1))
        assert usages
        assert not any(u.shared_cluster for u in usages)


class TestLiveClusterSharing:
    """Sharing has to be counted over the whole run, before the query's own filter.

    Counted after it, a history sees one step per cluster and reports nothing as shared —
    a failure that looks exactly like "no cluster was ever shared".
    """

    def test_a_history_still_sees_the_other_steps_on_the_cluster(self, export: BillingExport) -> None:
        usages = export.step_history(step=SHARED_STEP, since=date(2026, 5, 1))
        assert usages
        assert any(u.shared_cluster for u in usages), (
            f'{SHARED_STEP} shares a cluster in at least one run, so the flag cannot be '
            'uniformly false unless the count is being taken after the step filter'
        )

    def test_the_same_step_is_unflagged_in_a_run_where_it_had_the_cluster_alone(
        self, export: BillingExport
    ) -> None:
        """Guards the opposite failure: a flag stuck on true says nothing either."""
        usages = export.step_history(step=SHARED_STEP, since=date(2026, 5, 1))
        assert usages
        assert not all(u.shared_cluster for u in usages)
        assert not any(u.shared_cluster for u in usages if u.run == KNOWN_RUN)


class TestLiveCoverage:
    def test_labelled_spend_is_a_subset_of_pipeline_spend(self, export: BillingExport) -> None:
        """The headline total is not the cost of the run, and the report has to say so."""
        usages = export.run_usage(run=KNOWN_RUN, since=date(2026, 5, 1))
        window = usage_window(usages)
        assert window is not None
        coverage = export.window_coverage(window=window, since=date(2026, 5, 1))
        assert [c.currency for c in coverage] == ['GBP']
        entry = coverage[0]
        assert entry.labelled_cost >= total_cost(usages) - 0.01
        assert entry.labelled_cost < entry.pipeline_cost
        share = entry.labelled_share
        assert share is not None
        assert 0 < share < 1


class TestQueryValidatesAgainstTheRealSchema:
    """Dry runs, so these check syntax and column existence at zero bytes billed.

    Unlike the cost assertions above, these never go stale: they hold for as long as
    the export's schema does, and fail the moment a column is renamed or dropped.
    """

    @staticmethod
    def _dry_run(export: BillingExport, sql: str, params: list[bigquery.ScalarQueryParameter]) -> int:
        job = export.client.query(
            sql,
            job_config=bigquery.QueryJobConfig(query_parameters=params, dry_run=True, use_query_cache=False),
        )
        assert job.total_bytes_processed is not None
        return job.total_bytes_processed

    @pytest.mark.parametrize(
        'build',
        [
            pytest.param(run_usage_query, id='run usage'),
            pytest.param(step_history_query, id='step history'),
        ],
    )
    def test_query_is_valid_and_prunes_partitions(self, export: BillingExport, build: QueryBuilder) -> None:
        """The byte count is the assertion: without pruning the widest scan is unbounded."""
        sql, params = build(export.table, 'a-label', date(2026, 5, 1))
        scanned = self._dry_run(export, sql, params)
        assert 0 < scanned < FULL_SCAN_CEILING
        assert scanned < MAX_BYTES_BILLED

    def test_coverage_query_is_valid_and_prunes_partitions(self, export: BillingExport) -> None:
        usages = export.run_usage(run=KNOWN_RUN, since=date(2026, 5, 1))
        window = usage_window(usages)
        assert window is not None
        sql, params = window_coverage_query(export.table, window, date(2026, 5, 1))
        scanned = self._dry_run(export, sql, params)
        assert 0 < scanned < FULL_SCAN_CEILING
