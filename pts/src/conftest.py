"""Helper function to enable testing spark functions."""

from typing import Any

import pytest
from pyspark.sql import SparkSession

from pts.pyspark.common.session import Session


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


@pytest.fixture(scope='session')
def spark(pts_session: Session) -> SparkSession:
    return pts_session.spark


# Inject names into doctest globals
@pytest.fixture
def doctest_namespace(spark: SparkSession) -> dict[str, Any]:
    """Provide names available inside doctest examples.

    The returned dict is merged into the globals for every doctest.

    Args:
        spark (SparkSession): spark sesion object

    Returns:
        dict[str, Any]:
    """
    return {'spark': spark, 'pts_session': spark}
