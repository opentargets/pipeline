"""Builders of small credible set and credible set variant frames for the colocalisation tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pts.schemas.colocalisation import CREDIBLE_SET_COLUMNS, CREDIBLE_SET_VARIANT_COLUMNS
from pts.schemas.credible_set import credible_set_schema, credible_set_variant_schema
from pts.schemas.spark import to_spark_schema

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pyspark.sql import DataFrame, SparkSession


def credible_sets(spark: SparkSession, rows: Iterable[tuple[str, str, str]]) -> DataFrame:
    """Build credible sets from `(id, studyType, chromosome)` tuples."""
    return spark.createDataFrame(
        list(rows),
        to_spark_schema(credible_set_schema, CREDIBLE_SET_COLUMNS),
    )


def credible_set_variants(
    spark: SparkSession,
    rows: Iterable[tuple[str, str, float | None, float | None, float | None]],
) -> DataFrame:
    """Build variants from `(credibleSetId, variantId, pp, conditionalBeta, reportedBeta)` tuples."""
    return spark.createDataFrame(
        list(rows),
        to_spark_schema(credible_set_variant_schema, CREDIBLE_SET_VARIANT_COLUMNS),
    )
