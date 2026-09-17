"""Pytest configuration and shared fixtures for PTS tests."""

# Imported first, before any test module. See pts/transformers/l2g/__init__.py: sklearn and
# xgboost each ship their own LLVM OpenMP runtime on macOS, and whichever loads first wins.
# The l2g package guard only helps once that package is imported, so a test module importing
# skops at its top would still lose the race and die with SIGSEGV at first fit. This makes the
# ordering hold for the whole suite whatever the collection order.
import xgboost  # noqa: F401  # isort: skip

import pytest
from pyspark.sql import SparkSession

from pts.pyspark.common.session import Session


@pytest.fixture(scope='session')
def spark():
    """Create a Spark session for testing.

    This fixture is session-scoped to avoid recreating Spark sessions
    for multiple tests, which improves test performance.
    """
    spark_session = (
        SparkSession.builder
        .appName('pts_test_suite')
        .master('local[1]')
        .config('spark.sql.adaptive.enabled', 'false')
        .config('spark.sql.adaptive.coalescePartitions.enabled', 'false')
        .config('spark.serializer', 'org.apache.spark.serializer.KryoSerializer')
        .config('spark.sql.execution.arrow.pyspark.enabled', 'true')
        .config('spark.sql.shuffle.partitions', '1')
        .getOrCreate()
    )
    spark_session.sparkContext.setLogLevel('ERROR')
    yield spark_session
    spark_session.stop()


@pytest.fixture(scope='session')
def pts_session():
    """Return the repository Session wrapper (not raw SparkSession).

    Scope: session -> start once per test run, stop at the end.
    """
    # Use small resources in CI if desired:
    props = {
        # example: override any defaults if needed
        # 'spark.driver.memory': '1g',
    }

    s = Session(app_name='pts-test', properties=props)
    try:
        yield s
    finally:
        # ensure the spark session stops even if tests fail
        s.stop()
