"""Per-step billed usage for a unified pipeline run, read from the GCP billing export.

The export is hourly-bucketed: every usage row spans exactly 3600 seconds. Any time
span derived from it is therefore quantised *up* to the hour, which is why this module
exposes `span_hours` and never a "duration". A step that ran for twelve minutes and one
that ran for fifty-five minutes are indistinguishable here. For real per-step durations,
use Airflow task instances instead.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from google.cloud import bigquery
from pydantic import BaseModel

PipelineTool = Literal['pis', 'pts', 'gentropy']
"""The `tool` label values that belong to the unified pipeline.

Other values appear in the same billing export (`nextgen`, `standalone`, `pos`,
`orchestrator`, `genetics-output-support`) and belong to unrelated workloads.
"""


class StepUsage(BaseModel):
    """Billed usage for one pipeline step within one run.

    Args:
        run: The `run` label, which is the Airflow DAG run ID unless `run_label` was set.
        step: The `step` label, e.g. `pts_target`.
        tool: Which of the three pipeline tools produced the resource.
        started: Earliest `usage_start_time` across the step's billed rows.
        ended: Latest `usage_end_time` across the step's billed rows.
        span_hours: Hours between `started` and `ended`. Quantised up to the hour by the
            export's bucketing, so this is an upper bound on wall-clock time, not a
            measurement of it.
        net_cost: `SUM(cost) + SUM(credits.amount)`. Credits are negative in the export,
            so this subtracts them. Ignoring credits overstates cost by roughly 7%.
        currency: Currency code from the export. `GBP` for this billing account.
    """

    run: str
    step: str
    tool: PipelineTool
    started: datetime
    ended: datetime
    span_hours: float
    net_cost: float
    currency: str


_LABELLED_CTE = """
WITH labelled AS (
  SELECT
    (SELECT value FROM UNNEST(labels) WHERE key = 'run') AS run,
    (SELECT value FROM UNNEST(labels) WHERE key = 'step') AS step,
    (SELECT value FROM UNNEST(labels) WHERE key = 'tool') AS tool,
    usage_start_time,
    usage_end_time,
    cost,
    (SELECT IFNULL(SUM(c.amount), 0) FROM UNNEST(credits) c) AS credit,
    currency
  FROM `{table}`
  WHERE DATE(_PARTITIONTIME) >= @since
    AND cost_type = 'regular'
)
"""

_AGGREGATE = """
SELECT
  run,
  step,
  tool,
  MIN(usage_start_time) AS started,
  MAX(usage_end_time) AS ended,
  TIMESTAMP_DIFF(MAX(usage_end_time), MIN(usage_start_time), SECOND) / 3600 AS span_hours,
  SUM(cost) + SUM(credit) AS net_cost,
  ANY_VALUE(currency) AS currency
FROM labelled
WHERE step IS NOT NULL
  AND tool IN ('pis', 'pts', 'gentropy')
  AND {predicate}
GROUP BY run, step, tool
ORDER BY started
"""


def _query(
    table: str,
    predicate: str,
    params: list[bigquery.ScalarQueryParameter],
) -> tuple[str, list[bigquery.ScalarQueryParameter]]:
    """Assemble the labelled CTE and the aggregate for a given row predicate."""
    sql = _LABELLED_CTE.format(table=table) + _AGGREGATE.format(predicate=predicate)
    return sql, params


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
