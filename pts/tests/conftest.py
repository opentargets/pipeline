"""Pytest configuration and shared fixtures for PTS tests."""

# Imported first, before any test module. See pts/transformers/l2g/__init__.py: sklearn and
# xgboost each ship their own LLVM OpenMP runtime on macOS, and whichever loads first wins.
# The l2g package guard only helps once that package is imported, so a test module importing
# skops at its top would still lose the race and die with SIGSEGV at first fit. This makes the
# ordering hold for the whole suite whatever the collection order.
import xgboost  # noqa: F401  # isort: skip

from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock

import polars as pl
import pytest
from otter.config.model import Config
from pyspark.sql import SparkSession

from pts.pyspark.common.session import Session
from pts.schemas.credible_set import credible_set_schema, credible_set_variant_schema
from pts.transformers.credible_set.validation import ValidationSettings, validation


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
        .config('spark.sql.execution.arrow.pyspark.fallback.enabled', 'false')
        .config('spark.sql.shuffle.partitions', '1')
        # as Session and the pts clusters do: Spark 4 defaults ANSI mode on
        .config('spark.sql.ansi.enabled', 'false')
        # as the literature_publication_match step does
        .config('spark.executorEnv.POLARS_MAX_THREADS', '1')
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
    s = Session(app_name='pts-test')
    try:
        yield s
    finally:
        # ensure the spark session stops even if tests fail
        s.stop()


@pytest.fixture(params=['false', 'true'], ids=['ansi_off', 'ansi_on'])
def ansi(spark: SparkSession, request: pytest.FixtureRequest):
    """Run a test with ANSI mode off, as pts runs, and on, as Spark 4 defaults.

    Every session in the run is the same SparkSession, so the setting is put back.
    """
    previous = spark.conf.get('spark.sql.ansi.enabled')
    spark.conf.set('spark.sql.ansi.enabled', request.param)
    try:
        yield request.param == 'true'
    finally:
        spark.conf.set('spark.sql.ansi.enabled', previous)


@pytest.fixture
def validate_source(tmp_path: Path) -> Callable[..., tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]]:
    """Validate one source's QC'd sets and variants with the credible set validation task.

    The frames are written as parquet in the layout the task reads, with every column of their
    schema; the returned callable gives the valid sets, valid variants, excluded sets and excluded
    variants it wrote.
    """

    def validate(
        credible_set: pl.LazyFrame, credible_set_variant: pl.LazyFrame, settings: ValidationSettings
    ) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
        source = {}
        for name, frame, schema in [
            ('credible_set', credible_set, credible_set_schema),
            ('credible_set_variant', credible_set_variant, credible_set_variant_schema),
        ]:
            path = tmp_path / 'qc/gwas/source' / name
            path.mkdir(parents=True)
            pl.concat([pl.LazyFrame(schema=schema), frame], how='diagonal_relaxed').collect().write_parquet(
                path / 'part-0.parquet'
            )
            source[name] = [str(path)]
        output = ('valid_set', 'valid_variant', 'excluded_set', 'excluded_variant')
        validation(
            source=source,
            destination={name: str(tmp_path / 'out' / name) for name in output},
            settings=settings.model_dump(),
            config=Mock(spec=Config, work_path=str(tmp_path), release_uri=None),
        )
        valid_set, valid_variant, excluded_set, excluded_variant = (
            pl.scan_parquet(tmp_path / 'out' / name / '*.parquet') for name in output
        )
        return valid_set, valid_variant, excluded_set, excluded_variant

    return validate
