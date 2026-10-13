"""Unit test configuration for the project."""

from __future__ import annotations

import pytest
from airflow.models import DagBag


@pytest.fixture(params=['src/orchestration/dags'])
def dag_bag(request: pytest.FixtureRequest) -> DagBag:
    """Return a DAG bag for testing."""
    # example dags are their own bundles since airflow 3.3, so a folder bag never loads them
    return DagBag(dag_folder=request.param)
