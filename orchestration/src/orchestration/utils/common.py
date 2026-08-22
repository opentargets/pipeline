"""Airflow boilerplate code which can be shared by several DAGs."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

# Cloud configuration.
GCP_PROJECT_PLATFORM = 'open-targets-eu-dev'
GCP_PROJECT_GENETICS = 'open-targets-genetics-dev'
GCP_SERVICE_ACCOUNT = 'up-airflow-dev@open-targets-eu-dev.iam.gserviceaccount.com'
GCP_REGION = 'europe-west1'
GCP_ZONE = 'europe-west1-d'

GCP_BILLING_EXPORT_TABLE = (
    'open-targets-eu-dev.billing_export.gcp_billing_export_resource_v1_0001BA_599363_94C6B1'
)
"""Resource-level GCP billing export. Ingestion-time day-partitioned, so filter on
`DATE(_PARTITIONTIME)`. Oldest partition is 2026-05-01."""

BILLING_EXPORT_START = date(2026, 5, 1)
"""Oldest partition in the billing export, so the widest scan it can serve."""

AIRFLOW_BASE_URL = 'http://localhost:8080'
"""The Airflow API server, as reachable from the VM that runs it.

`compose.yaml` publishes a port for `airflow-apiserver` only, on 127.0.0.1:8080. The
`postgres` service publishes none, so the REST API is the only way to read Airflow
state even though a database driver is installed.
"""

GCS_PIPELINE_RUNS_BUCKET = 'open-targets-pipeline-runs'
"""Bucket holding development pipeline runs, and therefore the agent's journal.

Matches `dev_uri` in `dags/config/unified_pipeline.py`. Release runs write to
`open-targets-pre-data-releases` instead, which phase 1 does not target.
"""

STALL_CEILING_SECONDS = 6 * 60 * 60
"""Absolute stall threshold for a step with no observed history.

Most steps have no baseline on early runs: the billing export holds at most 18 of
roughly 150 steps, and Airflow's own history is destroyed with the VM. This is the
fallback, not the normal rule — a step with history is judged against its own
observed maximum instead.
"""

STALL_MULTIPLIER = 2.5
"""How far past its observed maximum a step may run before it is called stalled."""


def clean_label(value: str) -> str:
    """Normalise a value the way Google Cloud requires of a label value.

    Per the GCP docs, a label value may only contain lowercase letters, numeric
    characters, underscores and dashes, and may be at most 63 characters long.
    Anything else is replaced by a dash, which is lossy: Airflow's DAG run ID
    `manual__2026-07-21T15:07:47.545737+00:00` is stored as the label
    `manual__2026-07-21t15-07-47-545737-00-00`. Anything comparing against a label
    already in Google Cloud has to normalise its input the same way first.

    Args:
        value: The raw value to normalise.

    Returns:
        The normalised label value.
    """
    return re.sub(r'[^a-z0-9-_]', '-', value.lower())[0:63]


shared_dag_args: dict[str, Any] = {
    'owner': 'Open Targets Data Team',
    'retries': 0,
}

shared_dag_kwargs: dict[str, Any] = {
    'tags': ['genetics_etl', 'experimental'],
    'start_date': datetime.now(tz=UTC) - timedelta(days=1),
    'schedule': None,
    'catchup': False,
}
