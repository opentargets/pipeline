"""Pytest configuration and shared fixtures for PTS tests."""

import pytest

from pts.pyspark.common.session import Session


@pytest.fixture(scope='session')
def spark(pts_session):
    """Create a Spark session for testing.

    This fixture is session-scoped to avoid recreating Spark sessions
    for multiple tests, which improves test performance.

    It depends on ``pts_session`` so both share the same underlying
    ``SparkSession`` singleton. Otherwise the raw ``SparkSession``
    created here (without ``spark.jars.packages``) and the
    ``Session``-created one (with ``gcs-connector`` + ``spark-nlp``)
    clash via ``getOrCreate()`` depending on test order - the first
    fixture to run wins and the other reuses its config, making
    ``OnToma._spark_nlp_available`` flaky.
    """
    return pts_session.spark


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
