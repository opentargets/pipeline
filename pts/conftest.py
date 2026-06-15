"""Pytest configuration for doctests in src/pts."""

from typing import Any

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope='session')
def spark():
    """Create a Spark session for testing (including doctests).

    This fixture is session-scoped to avoid recreating Spark sessions
    for multiple tests, which improves test performance.
    """
    spark_session = (
        SparkSession.builder
        .appName('pts_doctest')
        .master('local[1]')
        .config('spark.sql.adaptive.enabled', 'false')
        .config('spark.sql.adaptive.coalescePartitions.enabled', 'false')
        .config('spark.serializer', 'org.apache.spark.serializer.KryoSerializer')
        .config('spark.sql.execution.arrow.pyspark.enabled', 'true')
        .getOrCreate()
    )
    spark_session.sparkContext.setLogLevel('ERROR')
    yield spark_session
    spark_session.stop()


@pytest.fixture(autouse=True)
def doctest_namespace(spark: SparkSession) -> dict[str, Any]:
    """Provide names available inside doctest examples.

    The returned dict is merged into the globals for every doctest.

    Args:
        spark (SparkSession): spark session object

    Returns:
        dict[str, Any]: namespace to inject into doctests
    """
    return {'spark': spark}
