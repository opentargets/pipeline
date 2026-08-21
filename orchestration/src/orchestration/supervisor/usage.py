"""Per-step billed usage for a unified pipeline run, read from the GCP billing export.

The export is hourly-bucketed: every usage row spans exactly 3600 seconds. Any time
span derived from it is therefore quantised *up* to the hour, which is why this module
exposes `span_hours` and never a "duration". A step that ran for twelve minutes and one
that ran for fifty-five minutes are indistinguishable here. For real per-step durations,
use Airflow task instances instead.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

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
