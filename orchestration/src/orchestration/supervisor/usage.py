"""Per-step billed usage for a unified pipeline run, read from the GCP billing export.

The export is hourly-bucketed: every usage row spans exactly 3600 seconds. Any time
span derived from it is therefore quantised *up* to the hour, which is why this module
exposes `span_hours` and never a "duration". A step that ran for twelve minutes and one
that ran for fifty-five minutes are indistinguishable here. For real per-step durations,
use Airflow task instances instead.

`span_hours` is also not a count of billed hours. It is the envelope from the start of a
step's first billed hour to the end of its last, gaps included, and a single step labels
several billed resources at once — a GCE step labels its instance and both of its disks,
a Dataproc step labels every node of its cluster — so the resource-hours actually billed
are a multiple of it.

Not every pound the pipeline spends carries a `step` label, either. Google Batch jobs are
unlabelled and some Dataproc disk and licensing rows fall outside the step's labels, so
the per-step figures are a subset of pipeline spend, never all of it. `window_coverage`
measures that subset against everything the pipeline billed over the same period.

`shared_cluster` guards the one way a labelled pound could fail to be the labelled step's
own. `operators/dataproc.py` passes `use_if_exists=True` and does not relabel, so a cluster
instance outliving the step that created it would bill its hours under that step's labels.
The flag marks any row where more than one step billed against a single cluster *instance*
in one run. **It does not fire on today's data and is expected not to**: measured over the
export on 2026-08-21, all 181 cluster instances served exactly one step (£2,039.82).

Instance means the `goog-dataproc-cluster-uuid` label, never the name. Names are reused
within a run — `up-pts-5df4f` in `up-20260527-1458` covers 6 steps across 12 separate
uuids, each created, used and deleted in turn — and every one of those instances carries
its own creating step's label, so the attribution is right. Grouping by name reports all
of them as suspect, which is worse than not checking: it teaches the reader to distrust
correct numbers.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from typing import Literal, get_args

from google.cloud import bigquery
from pydantic import BaseModel

from orchestration.utils.common import GCP_BILLING_EXPORT_TABLE

PipelineTool = Literal['pis', 'pts', 'gentropy']
"""The `tool` label values that belong to the unified pipeline.

Other values appear in the same billing export (`nextgen`, `standalone`, `pos`,
`orchestrator`, `genetics-output-support`) and belong to unrelated workloads.
"""

PipelineCreator = Literal['unified-pipeline', 'gentropy-pipelines']
"""The `created_by` label values `default_labels` stamps on pipeline resources.

Everything the pipeline bills carries one of these, labelled per step or not, so they
delimit the denominator of `window_coverage`.
"""

MAX_BYTES_BILLED = 16 * 1024**3
"""Cap on the bytes any one query in this module may bill, in bytes (16 GiB).

`--since` defaults to the whole export retention, and the export is resource-level, so
every invocation scans more than the last as the retention grows. A recent full-retention
dry run reported ~1.77 GB, so this leaves roughly nine times headroom for growth while
still stopping the runaway case: a query that loses its `_PARTITIONTIME` filter goes from
scanning gigabytes to scanning the entire table, and BigQuery bills it either way. Raise
this when a legitimate full scan approaches it, rather than removing it.
"""

_PIPELINE_TOOLS_SQL = ', '.join(f"'{tool}'" for tool in get_args(PipelineTool))
"""The `PipelineTool` values as a SQL `IN` list, so the filter and the model cannot drift.

Generating the list from the `Literal` makes one direction impossible and leaves the other
wide open. The filter can never admit a tool the model rejects. But a fourth `tool` label
added in `dags/unified_pipeline.py` without also being added to `PipelineTool` is filtered
out by this list, and its cost is then silently missing from every report — no error, just
a smaller number. Adding a tool label means editing `PipelineTool` in the same change.
"""

_PIPELINE_CREATORS_SQL = ', '.join(f"'{creator}'" for creator in get_args(PipelineCreator))
"""The `PipelineCreator` values as a SQL `IN` list."""


class StepUsage(BaseModel):
    """Billed usage for one pipeline step within one run.

    Args:
        run: The `run` label. That is the Airflow DAG run ID, or the `run_label` param
            when one was set, passed through `clean_label` — so
            `manual__2026-07-21T15:07:47.545737+00:00` is stored here as
            `manual__2026-07-21t15-07-47-545737-00-00`.
        step: The `step` label, e.g. `pts_target`.
        tool: Which of the three pipeline tools produced the resource.
        product: The `product` label, `platform` or `ppp`. The partner preview pipeline
            runs the same DAG, so two runs can share a `run` label and differ only here.
            `None` for rows that carry no `product` label.
        started: Earliest `usage_start_time` across the step's billed rows.
        ended: Latest `usage_end_time` across the step's billed rows.
        span_hours: Hours from `started` to `ended`. The envelope of the step's billed
            rows, gaps included, quantised up to the hour by the export's bucketing. It
            is neither a wall-clock duration nor a count of billed resource-hours.
        net_cost: `SUM(cost) + SUM(credits.amount)`. Credits are negative in the export,
            so this subtracts them. Ignoring credits overstates cost by roughly 7%.
        currency: Currency code from the export. `GBP` for this billing account. Part of
            the grouping key, so an amount here is never a mix of currencies.
        shared_cluster: True when more than one step of this run billed against a single
            Dataproc cluster *instance* (`goog-dataproc-cluster-uuid`, not the reused
            name). Such a row is charged the instance's whole cost rather than the step's
            own, because the instance keeps the labels it was created with. False for
            every row in the export as of 2026-08-21 — this is a guard on a code path
            (`use_if_exists=True`) that does not currently manifest, not a description of
            observed billing. Detected, never re-apportioned: splitting an instance's cost
            between steps needs the Dataproc Jobs API.
    """

    run: str
    step: str
    tool: PipelineTool
    product: str | None
    started: datetime
    ended: datetime
    span_hours: float
    net_cost: float
    currency: str
    shared_cluster: bool


class WindowCoverage(BaseModel):
    """How much of a time window's pipeline spend carries step labels.

    Args:
        currency: Currency code the two amounts are denominated in.
        labelled_cost: Net cost of exactly the rows the report shows: this run's rows, with
            a `step` label and a `tool` in `PipelineTool`.
        pipeline_cost: Net cost of every row created by the pipeline in the window,
            labelled per step or not, and whichever run produced it. Deliberately not
            filtered by run: a run that overlapped this one really did spend that money in
            this window, and hiding it would understate what the window cost.
    """

    currency: str
    labelled_cost: float
    pipeline_cost: float

    @property
    def labelled_share(self) -> float | None:
        """The labelled fraction of pipeline spend, or `None` when there is none to divide."""
        if not self.pipeline_cost:
            return None
        return self.labelled_cost / self.pipeline_cost

    @property
    def exceeds_pipeline_cost(self) -> bool:
        """Whether the numerator is larger than the total it claims to be a share of.

        Impossible if both figures measure what they say, so it means the report counted
        rows the denominator did not: a step-labelled row without a `created_by` the
        pipeline recognises. It is a broken measurement, not excellent coverage, and has
        to read as one.
        """
        return self.labelled_cost > self.pipeline_cost


_LABELLED_CTE = """
WITH labelled AS (
  SELECT
    (SELECT value FROM UNNEST(labels) WHERE key = 'run') AS run,
    (SELECT value FROM UNNEST(labels) WHERE key = 'step') AS step,
    (SELECT value FROM UNNEST(labels) WHERE key = 'tool') AS tool,
    (SELECT value FROM UNNEST(labels) WHERE key = 'product') AS product,
    (SELECT value FROM UNNEST(labels) WHERE key = 'goog-dataproc-cluster-uuid') AS cluster_instance,
    usage_start_time,
    usage_end_time,
    cost,
    (SELECT IFNULL(SUM(c.amount), 0) FROM UNNEST(credits) c) AS credit,
    currency
  FROM `{table}`
  WHERE DATE(_PARTITIONTIME) >= @since
    AND cost_type = 'regular'
),
cluster_steps AS (
  SELECT
    run AS cluster_run,
    cluster_instance AS cluster_key,
    COUNT(DISTINCT step) AS steps_on_cluster
  FROM labelled
  WHERE cluster_instance IS NOT NULL
    AND step IS NOT NULL
  GROUP BY run, cluster_instance
)
"""

_AGGREGATE = """
SELECT
  run,
  step,
  tool,
  product,
  MIN(usage_start_time) AS started,
  MAX(usage_end_time) AS ended,
  TIMESTAMP_DIFF(MAX(usage_end_time), MIN(usage_start_time), SECOND) / 3600 AS span_hours,
  SUM(cost) + SUM(credit) AS net_cost,
  currency,
  IFNULL(LOGICAL_OR(steps_on_cluster > 1), FALSE) AS shared_cluster
FROM labelled
LEFT JOIN cluster_steps ON run = cluster_run AND cluster_instance = cluster_key
WHERE step IS NOT NULL
  AND tool IN ({tools})
  AND {predicate}
GROUP BY run, step, tool, product, currency
ORDER BY started, run, step, tool, product, currency
"""

_COVERAGE = """
WITH windowed AS (
  SELECT
    (SELECT value FROM UNNEST(labels) WHERE key = 'run') AS run,
    (SELECT value FROM UNNEST(labels) WHERE key = 'step') AS step,
    (SELECT value FROM UNNEST(labels) WHERE key = 'tool') AS tool,
    (SELECT value FROM UNNEST(labels) WHERE key = 'created_by') AS created_by,
    cost + (SELECT IFNULL(SUM(c.amount), 0) FROM UNNEST(credits) c) AS net_cost,
    currency
  FROM `{table}`
  WHERE DATE(_PARTITIONTIME) >= @since
    AND cost_type = 'regular'
    AND usage_start_time >= @window_start
    AND usage_end_time <= @window_end
)
SELECT
  currency,
  SUM(IF(run = @run AND step IS NOT NULL AND tool IN ({tools}), net_cost, 0)) AS labelled_cost,
  SUM(IF(created_by IN ({creators}), net_cost, 0)) AS pipeline_cost
FROM windowed
GROUP BY currency
ORDER BY currency
"""


def _query(
    table: str,
    predicate: str,
    params: list[bigquery.ScalarQueryParameter],
) -> tuple[str, list[bigquery.ScalarQueryParameter]]:
    """Assemble the labelled CTEs and the aggregate for a given row predicate.

    The predicate belongs to the aggregate and never to the CTEs. `cluster_steps` has to
    count a cluster's steps across the whole run, so filtering it to one step or one run
    first would make every history report its own step as the only one on its cluster,
    and nothing would ever be flagged as shared.
    """
    aggregate = _AGGREGATE.format(predicate=predicate, tools=_PIPELINE_TOOLS_SQL)
    return _LABELLED_CTE.format(table=table) + aggregate, params


def run_usage_query(
    table: str, run: str, since: date
) -> tuple[str, list[bigquery.ScalarQueryParameter]]:
    """Build the query for every step's billed usage within one run.

    Args:
        table: Fully-qualified billing export table.
        run: The `run` label to filter on.
        since: Earliest partition date to scan.

    Returns:
        The SQL text and its bound parameters.
    """
    return _query(
        table,
        'run = @run',
        [
            bigquery.ScalarQueryParameter('run', 'STRING', run),
            bigquery.ScalarQueryParameter('since', 'DATE', since),
        ],
    )


def step_history_query(
    table: str, step: str, since: date
) -> tuple[str, list[bigquery.ScalarQueryParameter]]:
    """Build the query for one step's billed usage across every run.

    Args:
        table: Fully-qualified billing export table.
        step: The `step` label to filter on.
        since: Earliest partition date to scan.

    Returns:
        The SQL text and its bound parameters.
    """
    return _query(
        table,
        'step = @step',
        [
            bigquery.ScalarQueryParameter('step', 'STRING', step),
            bigquery.ScalarQueryParameter('since', 'DATE', since),
        ],
    )


def window_coverage_query(
    table: str, run: str, window: tuple[datetime, datetime], since: date
) -> tuple[str, list[bigquery.ScalarQueryParameter]]:
    """Build the query comparing one run's step-labelled spend with all pipeline spend.

    The numerator carries the same three predicates as the report's own rows — this run,
    a `step` label, a `tool` in `PipelineTool` — so it is the table's total by
    construction and not merely close to it. The denominator is every pipeline-created row
    in the window regardless of run, because an overlapping run's spend is real spend in
    that period.

    Args:
        table: Fully-qualified billing export table.
        run: The `run` label whose rows form the numerator.
        window: Inclusive `(start, end)` of the period to measure. A row counts only if it
            falls entirely inside it. That is exact rather than approximate: every row in
            the denominator was measured on 2026-08-21 to span exactly 3600 seconds
            (3,060,837 rows, one bucket size), so there are no daily-bucketed SKUs for
            containment to drop.
        since: Earliest partition date to scan. Prunes partitions; the window itself is on
            usage time, which is not the partitioning column.

    Returns:
        The SQL text and its bound parameters.
    """
    start, end = window
    sql = _COVERAGE.format(table=table, tools=_PIPELINE_TOOLS_SQL, creators=_PIPELINE_CREATORS_SQL)
    return sql, [
        bigquery.ScalarQueryParameter('run', 'STRING', run),
        bigquery.ScalarQueryParameter('window_start', 'TIMESTAMP', start),
        bigquery.ScalarQueryParameter('window_end', 'TIMESTAMP', end),
        bigquery.ScalarQueryParameter('since', 'DATE', since),
    ]


class BillingExport:
    """Reads per-step billed usage from the GCP billing export.

    Args:
        client: An authenticated BigQuery client. Injected so that tests never
            need credentials or network access.
        table: Fully-qualified export table. Defaults to the platform's export.
    """

    def __init__(self, client: bigquery.Client, table: str = GCP_BILLING_EXPORT_TABLE) -> None:
        self.client = client
        self.table = table

    def _run_query(
        self, sql: str, params: list[bigquery.ScalarQueryParameter]
    ) -> bigquery.table.RowIterator:
        job_config = bigquery.QueryJobConfig(
            query_parameters=params,
            maximum_bytes_billed=MAX_BYTES_BILLED,
        )
        return self.client.query(sql, job_config=job_config).result()

    def _fetch(self, sql: str, params: list[bigquery.ScalarQueryParameter]) -> list[StepUsage]:
        return [
            StepUsage(
                run=row.run,
                step=row.step,
                tool=row.tool,
                product=row.product,
                started=row.started,
                ended=row.ended,
                span_hours=row.span_hours,
                net_cost=row.net_cost,
                currency=row.currency,
                shared_cluster=row.shared_cluster,
            )
            for row in self._run_query(sql, params)
        ]

    def run_usage(self, run: str, since: date) -> list[StepUsage]:
        """Billed usage for every step in one run, ordered by start time.

        Args:
            run: The `run` label to filter on.
            since: Earliest partition date to scan. The export holds nothing
                before 2026-05-01.

        Returns:
            One `StepUsage` per (step, tool, product, currency). Empty if the run has not
            billed yet.
        """
        return self._fetch(*run_usage_query(self.table, run, since))

    def step_history(self, step: str, since: date) -> list[StepUsage]:
        """Billed usage for one step across every run that produced it.

        Args:
            step: The `step` label to filter on.
            since: Earliest partition date to scan.

        Returns:
            One `StepUsage` per (run, tool, product, currency), ordered by start time. A
            run that billed the step on more than one tool contributes one row per tool.
        """
        return self._fetch(*step_history_query(self.table, step, since))

    def window_coverage(
        self, run: str, window: tuple[datetime, datetime], since: date
    ) -> list[WindowCoverage]:
        """How much of the pipeline's spend over a window one run's reported steps cover.

        Args:
            run: The `run` label whose reported rows form the numerator. Must be the run
                the window came from, or the two figures describe different things.
            window: Inclusive `(start, end)` of the period to measure, normally
                `usage_window` of that run's usages.
            since: Earliest partition date to scan.

        Returns:
            One `WindowCoverage` per currency, ordered by currency code. Empty when the
            pipeline billed nothing in the window.
        """
        sql, params = window_coverage_query(self.table, run, window, since)
        return [
            WindowCoverage(
                currency=row.currency,
                labelled_cost=row.labelled_cost,
                pipeline_cost=row.pipeline_cost,
            )
            for row in self._run_query(sql, params)
        ]


def total_cost(usages: Iterable[StepUsage]) -> float:
    """Sum net cost across usages.

    Args:
        usages: The usages to total.

    Returns:
        Total net cost. Zero for an empty input.
    """
    return sum((u.net_cost for u in usages), 0.0)


def usage_window(usages: Iterable[StepUsage]) -> tuple[datetime, datetime] | None:
    """The period the usages span, from the earliest start to the latest end.

    Args:
        usages: The usages to bound.

    Returns:
        The `(start, end)` of the period, or `None` for an empty input, where there is no
        window to report and reporting one would be an invention.
    """
    materialised = list(usages)
    if not materialised:
        return None
    return min(u.started for u in materialised), max(u.ended for u in materialised)
