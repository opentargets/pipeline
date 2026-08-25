"""Live checks against the real billing export.

Skipped unless RUN_BIGQUERY_TESTS is set, because these need credentials and network.
Run with: RUN_BIGQUERY_TESTS=1 uv run --frozen pytest tests/test_supervisor_usage_live.py -rxs
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, date, datetime

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

REUSED_NAME_STEP = 'pts_literature_ontoma'
"""A step whose runs reuse one Dataproc cluster *name* for several steps.

In `2026-06-19_ppp_up` the name `up-pts-literature-38bc6` covers this step and
`pts_literature_embedding`, and in `up-20260527-1458` `up-pts-literature-5df4f` covers it
and `pts_literature_publication_match`. Grouped by name, both look shared. Grouped by
instance uuid, neither is: each step had its own cluster instance. Nothing here may be
flagged, and that is the point of the fixture.
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

    def test_the_run_is_one_product(self, export: BillingExport) -> None:
        """A platform run and a PPP run of the same DAG must never be summed together."""
        usages = export.run_usage(run=KNOWN_RUN, since=date(2026, 5, 1))
        assert usages
        assert {u.product for u in usages} == {'platform'}


REACTOME_STEP = 'pts_reactome'
"""Verified live (2026-08-24) as the step whose envelope most exceeds its billed hours.

Its rows span 30 hourly SKU groups by `started`/`ended` but land in only 4 distinct billed
hours, cost £6.69 net, and put ~59.1 of its ~162.1 core-hours on Spot Preemptible N1.
"""


class TestLiveCoreUsage:
    """Reproduces the exact figures this task exists to add, against the real export.

    Not "some step has core data" — the specific step, the specific numbers, matched
    against a live independent query in the report this task's brief was built from.
    """

    def test_pts_reactome_matches_the_verified_billed_hours_and_cost(
        self, export: BillingExport
    ) -> None:
        usages = export.run_usage(run=KNOWN_RUN, since=date(2026, 5, 1))
        reactome = next(u for u in usages if u.step == REACTOME_STEP)
        assert reactome.billed_hours == 4
        assert reactome.net_cost == pytest.approx(6.69, abs=0.01)

    def test_pts_reactome_matches_the_verified_core_hours(self, export: BillingExport) -> None:
        usages = export.run_usage(run=KNOWN_RUN, since=date(2026, 5, 1))
        reactome = next(u for u in usages if u.step == REACTOME_STEP)
        assert reactome.core_seconds is not None
        assert reactome.core_seconds / 3600 == pytest.approx(162.1, abs=0.1)

    def test_pts_reactome_shows_spot_core_time_and_the_n1_family(self, export: BillingExport) -> None:
        usages = export.run_usage(run=KNOWN_RUN, since=date(2026, 5, 1))
        reactome = next(u for u in usages if u.step == REACTOME_STEP)
        assert reactome.core_seconds is not None
        assert reactome.spot_core_seconds is not None
        assert reactome.spot_core_seconds > 0
        assert reactome.spot_core_seconds < reactome.core_seconds
        assert reactome.machine_families == ['N1']

    def test_every_step_in_the_known_run_has_core_data(self, export: BillingExport) -> None:
        """Every step here ran on Dataproc or GCE, so none should show `None`.

        A `None`-for-everything result would satisfy the three tests above vacuously if
        the guard were inverted; this is what would catch that.
        """
        usages = export.run_usage(run=KNOWN_RUN, since=date(2026, 5, 1))
        assert usages
        assert all(u.core_seconds is not None for u in usages)
        assert all(u.machine_families == ['N1'] for u in usages)

    def test_the_total_cost_the_usage_command_reports_is_unmoved(self, export: BillingExport) -> None:
        """This task must not change what `total_cost` already reported for this run."""
        usages = export.run_usage(run=KNOWN_RUN, since=date(2026, 5, 1))
        assert total_cost(usages) == pytest.approx(21.60, abs=0.01)


class TestLiveClusterSharing:
    """The flag is a guard on a path that does not manifest, so live it must stay silent.

    Which makes these tests weak on their own — a flag that never fires and a flag that
    cannot fire look identical from here. `TestSyntheticSharing` is what proves it can.
    """

    def test_reused_cluster_names_are_not_reported_as_shared(self, export: BillingExport) -> None:
        """The correction that motivated switching from name to uuid.

        Grouped by name this step is shared in two of its runs. Grouped by instance it is
        shared in none, which is the truth: each instance served exactly one step.
        """
        usages = export.step_history(step=REUSED_NAME_STEP, since=date(2026, 5, 1))
        assert usages
        assert not any(u.shared_cluster for u in usages)

    def test_no_run_in_the_export_shares_a_cluster_instance(self, export: BillingExport) -> None:
        """Measured 2026-08-21: 181 instances, every one serving exactly one step.

        A failure here is a real finding, not a broken test: it means a cluster instance
        outlived the step that created it and another step billed against it. Investigate
        the run before touching this assertion.
        """
        usages = export.run_usage(run=KNOWN_RUN, since=date(2026, 5, 1))
        assert usages
        assert not any(u.shared_cluster for u in usages)

    def test_the_uuid_label_is_on_exactly_the_rows_the_name_label_is(
        self, export: BillingExport
    ) -> None:
        """Switching the grouping key must not quietly drop rows out of the check."""
        sql = f"""
        SELECT
          COUNTIF(name IS NOT NULL AND uuid IS NULL) AS name_only,
          COUNTIF(uuid IS NOT NULL AND name IS NULL) AS uuid_only,
          COUNTIF(uuid IS NOT NULL) AS with_uuid
        FROM (
          SELECT
            (SELECT value FROM UNNEST(labels) WHERE key = 'goog-dataproc-cluster-name') AS name,
            (SELECT value FROM UNNEST(labels) WHERE key = 'goog-dataproc-cluster-uuid') AS uuid
          FROM `{export.table}`
          WHERE DATE(_PARTITIONTIME) >= '2026-05-01'
        )
        """  # noqa: S608 - the only interpolation is the export's own table constant
        row = next(iter(export.client.query(sql).result()))
        assert row.with_uuid > 0
        assert row.name_only == 0
        assert row.uuid_only == 0


SYNTHETIC_TABLE = 'synthetic.billing.rows'
"""Stand-in table name, replaced by inline rows before the query runs."""


def _synthetic_row(step: str, uuid: str, name: str) -> str:
    """One billing row as a STRUCT literal, shaped like the export's schema.

    `sku` and `usage` are here only so the query resolves — `_LABELLED_CTE` reads
    `sku.description` and `usage.amount` unconditionally now, for every row regardless of
    what this test is checking. Values chosen so nothing here counts as a core SKU, which
    keeps these rows out of `core_seconds`/`spot_core_seconds`/`machine_families` and
    leaves the sharing checks these rows exist for undisturbed.
    """
    return f"""STRUCT(
      TIMESTAMP '2026-07-21 15:00:00' AS _PARTITIONTIME,
      TIMESTAMP '2026-07-21 15:00:00' AS usage_start_time,
      TIMESTAMP '2026-07-21 16:00:00' AS usage_end_time,
      1.0 AS cost,
      'GBP' AS currency,
      'regular' AS cost_type,
      [STRUCT(-0.1 AS amount)] AS credits,
      [
        STRUCT('run' AS key, 'synthetic-run' AS value),
        STRUCT('step' AS key, '{step}' AS value),
        STRUCT('tool' AS key, 'pts' AS value),
        STRUCT('product' AS key, 'platform' AS value),
        STRUCT('goog-dataproc-cluster-uuid' AS key, '{uuid}' AS value),
        STRUCT('goog-dataproc-cluster-name' AS key, '{name}' AS value)
      ] AS labels,
      STRUCT('Synthetic SKU, not a core SKU' AS description) AS sku,
      STRUCT(0.0 AS amount) AS usage
    )"""


class TestSyntheticSharing:
    """Runs the real query against constructed rows, to prove the flag can fire at all.

    The only substitution is the table reference. Every CTE, the join, the predicates and
    the flag expression are exactly the production text, so this cannot pass while the
    detection is broken. It bills no bytes: the rows are literals.
    """

    @staticmethod
    def _flags(
        export: BillingExport,
        rows: list[str],
        query: tuple[str, list[bigquery.ScalarQueryParameter]] | None = None,
    ) -> dict[str, bool]:
        sql, params = query or run_usage_query(SYNTHETIC_TABLE, 'synthetic-run', date(2026, 5, 1))
        source = f'(SELECT * FROM UNNEST([{",".join(rows)}]))'  # noqa: S608 — literals built here
        inline = sql.replace(f'`{SYNTHETIC_TABLE}`', source)
        job = export.client.query(inline, job_config=bigquery.QueryJobConfig(query_parameters=params))
        return {row.step: row.shared_cluster for row in job.result()}

    def test_one_instance_serving_two_steps_is_flagged(self, export: BillingExport) -> None:
        flags = self._flags(
            export,
            [
                _synthetic_row('step_a', 'uuid-shared', 'cluster-x'),
                _synthetic_row('step_b', 'uuid-shared', 'cluster-x'),
            ],
        )
        assert flags == {'step_a': True, 'step_b': True}

    def test_one_name_reused_by_two_instances_is_not_flagged(self, export: BillingExport) -> None:
        """The live shape: a name created, deleted and created again inside one run."""
        flags = self._flags(
            export,
            [
                _synthetic_row('step_c', 'uuid-first', 'cluster-reused'),
                _synthetic_row('step_d', 'uuid-second', 'cluster-reused'),
            ],
        )
        assert flags == {'step_c': False, 'step_d': False}

    def test_a_history_of_one_step_still_sees_the_other_on_the_instance(
        self, export: BillingExport
    ) -> None:
        """The trap, checked end to end rather than structurally.

        A history binds `step = @step`. If the sharing count were taken after that
        predicate it would see one step on the instance and flag nothing, which is
        indistinguishable from a run where nothing was shared.
        """
        rows = [
            _synthetic_row('step_a', 'uuid-shared', 'cluster-x'),
            _synthetic_row('step_b', 'uuid-shared', 'cluster-x'),
        ]
        flags = self._flags(
            export, rows, query=step_history_query(SYNTHETIC_TABLE, 'step_a', date(2026, 5, 1))
        )
        assert flags == {'step_a': True}

    def test_both_cases_are_distinguished_in_one_result(self, export: BillingExport) -> None:
        """Together, since a flag stuck on or off passes one of the two tests above."""
        flags = self._flags(
            export,
            [
                _synthetic_row('step_a', 'uuid-shared', 'cluster-x'),
                _synthetic_row('step_b', 'uuid-shared', 'cluster-x'),
                _synthetic_row('step_c', 'uuid-first', 'cluster-reused'),
                _synthetic_row('step_d', 'uuid-second', 'cluster-reused'),
            ],
        )
        assert flags == {'step_a': True, 'step_b': True, 'step_c': False, 'step_d': False}


class TestLiveCoverage:
    def test_the_numerator_is_the_runs_own_total(self, export: BillingExport) -> None:
        """Bound to the run, so it is the table's total rather than every run in the window."""
        usages = export.run_usage(run=KNOWN_RUN, since=date(2026, 5, 1))
        window = usage_window(usages)
        assert window is not None
        coverage = export.window_coverage(run=KNOWN_RUN, window=window, since=date(2026, 5, 1))
        assert [c.currency for c in coverage] == ['GBP']
        entry = coverage[0]
        assert entry.labelled_cost == pytest.approx(total_cost(usages), abs=0.01)

    def test_labelled_spend_is_a_subset_of_pipeline_spend(self, export: BillingExport) -> None:
        """The headline total is not the cost of the run, and the report has to say so."""
        usages = export.run_usage(run=KNOWN_RUN, since=date(2026, 5, 1))
        window = usage_window(usages)
        assert window is not None
        coverage = export.window_coverage(run=KNOWN_RUN, window=window, since=date(2026, 5, 1))
        entry = coverage[0]
        assert entry.labelled_cost < entry.pipeline_cost
        assert not entry.exceeds_pipeline_cost
        share = entry.labelled_share
        assert share is not None
        assert 0 < share < 1

    def test_every_denominator_row_is_one_hour(self, export: BillingExport) -> None:
        """Containment drops nothing, measured rather than assumed.

        A daily-bucketed SKU would straddle the window and be excluded from both figures.
        There are none: every row spans exactly 3600 seconds.
        """
        sql = f"""
        SELECT COUNT(DISTINCT TIMESTAMP_DIFF(usage_end_time, usage_start_time, SECOND)) AS sizes,
               MIN(TIMESTAMP_DIFF(usage_end_time, usage_start_time, SECOND)) AS seconds
        FROM `{export.table}`
        WHERE DATE(_PARTITIONTIME) >= '2026-05-01'
          AND cost_type = 'regular'
          AND (SELECT value FROM UNNEST(labels) WHERE key = 'created_by')
              IN ('unified-pipeline', 'gentropy-pipelines')
        """  # noqa: S608 - the only interpolation is the export's own table constant
        row = next(iter(export.client.query(sql).result()))
        assert row.sizes == 1
        assert row.seconds == 3600


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
        window = (datetime(2026, 7, 21, 15, tzinfo=UTC), datetime(2026, 7, 21, 21, tzinfo=UTC))
        sql, params = window_coverage_query(export.table, KNOWN_RUN, window, date(2026, 5, 1))
        scanned = self._dry_run(export, sql, params)
        assert 0 < scanned < FULL_SCAN_CEILING
