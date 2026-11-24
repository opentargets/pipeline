"""Shared fixtures for the pps tests."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession

from pps.session import Session


@pytest.fixture(scope='session')
def spark() -> Generator[SparkSession]:
    """Local spark session for testing purposes."""
    conf = (
        SparkConf()
        .set('spark.driver.bindAddress', '127.0.0.1')
        .set('spark.sql.shuffle.partitions', '1')
        .set('spark.ui.enabled', 'false')
        .set('spark.ui.showConsoleProgress', 'false')
        .set('spark.driver.memory', '2g')
    )
    spark = SparkSession.Builder().config(conf=conf).master('local[1]').appName('test').getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def session(spark: SparkSession) -> Session:
    """pps Session wrapping the testing spark session."""
    return Session()
