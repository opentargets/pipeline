"""Airflow boilerplate code which can be shared by several DAGs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
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
