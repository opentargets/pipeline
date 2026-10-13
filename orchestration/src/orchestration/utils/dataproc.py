"""Airflow boilerplate code that interfaces with Dataproc operators which can be shared by several DAGs."""

from __future__ import annotations

import logging

from airflow.sdk import TriggerRule

from orchestration.operators.dataproc import (
    CreateClusterOperator,
    CustomClusterConfig,
    DeleteClusterOperator,
)
from orchestration.utils.common import (
    GCP_PROJECT_GENETICS,
    GCP_REGION,
)
from orchestration.utils.labels import Labels

log: logging.Logger = logging.getLogger(__name__)


def create_cluster(cluster_name: str, cluster_config: CustomClusterConfig) -> CreateClusterOperator:
    """Wrapper over `CreateClusterOperator` tailored for staging dataproc clusters."""
    return CreateClusterOperator(
        project_id=GCP_PROJECT_GENETICS,
        region=GCP_REGION,
        task_id=f'create_cluster_{cluster_name}',
        cluster_name=cluster_name,
        cluster_config=cluster_config,
        labels=Labels(project=GCP_PROJECT_GENETICS),
        deferrable=True,
    )


def delete_cluster(cluster_name: str) -> DeleteClusterOperator:
    """Generate an Airflow task to delete a Dataproc cluster.

    Args:
        cluster_name (str): Name of the cluster.

    Returns:
        DeleteClusterOperator: Airflow task to delete a Dataproc cluster.
    """
    return DeleteClusterOperator(
        task_id=f'delete_cluster_{cluster_name}',
        project_id=GCP_PROJECT_GENETICS,
        cluster_name=cluster_name,
        region=GCP_REGION,
        trigger_rule=TriggerRule.ALL_SUCCESS,
        deferrable=True,
    )
