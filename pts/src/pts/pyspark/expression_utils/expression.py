"""Baseline expression QC flagging helpers.

Module-level functions over a Spark DataFrame; the wrapper class was removed
because the only consumer is `baseline_expression_validate`.
"""

from __future__ import annotations

from enum import StrEnum

from pyspark.sql import DataFrame
from pyspark.sql import functions as f
from pyspark.sql import types as t

from pts.pyspark.common.utils import update_quality_flag

QC_COLUMN = 'qualityControls'


class BaselineExpressionFlags(StrEnum):
    INVALID_TARGET = 'No valid target'
    INVALID_BIOSAMPLE = 'No valid biosample'


def _ensure_qc(df: DataFrame) -> DataFrame:
    if QC_COLUMN in df.columns:
        return df
    return df.withColumn(QC_COLUMN, f.array().cast(t.ArrayType(t.StringType())))


def _flag(df: DataFrame, condition, flag: BaselineExpressionFlags) -> DataFrame:
    return df.withColumn(QC_COLUMN, update_quality_flag(f.col(QC_COLUMN), condition, flag))


def validate_target(
    df: DataFrame,
    target_lut: DataFrame,
) -> DataFrame:
    """Resolve `targetId` aliases via the target LUT."""
    df = _ensure_qc(df)
    lut = f.broadcast(target_lut).select(
        f.col('targetId').alias('_mappedTargetId'), f.col('targetFromSourceId').alias('_lookupId')
    )
    joined = df.join(lut, df['targetId'] == f.col('_lookupId'), 'leftouter')
    flagged = _flag(joined, f.col('_mappedTargetId').isNull(), BaselineExpressionFlags.INVALID_TARGET)
    return flagged.withColumn('targetId', f.coalesce(f.col('_mappedTargetId'), f.col('targetId'))).drop(
        '_mappedTargetId', '_lookupId'
    )


def validate_biosample(
    df: DataFrame,
    biosample_lut: DataFrame,
    id_col: str,
    source_col: str,
) -> DataFrame:
    """Resolve a tissue or celltype biosample id via the biosample LUT and flag misses.

    Rows without either `id_col` or `source_col` populated are treated as not
    requiring this biosample dimension and are left untouched.
    """
    if id_col not in df.columns and source_col not in df.columns:
        return df
    df = _ensure_qc(df)
    if id_col not in df.columns:
        df = df.withColumn(id_col, f.lit(None).cast(t.StringType()))

    requires = f.col(id_col).isNotNull()
    if source_col in df.columns:
        requires |= f.col(source_col).isNotNull()

    lut = f.broadcast(biosample_lut).select(
        f.col('biosampleId').alias('_mappedBiosampleId'),
        f.col('biosampleFromSourceMappedId').alias('_lookupId'),
    )
    joined = df.join(lut, df[id_col] == f.col('_lookupId'), 'leftouter')
    flagged = _flag(joined, requires & f.col('_mappedBiosampleId').isNull(), BaselineExpressionFlags.INVALID_BIOSAMPLE)
    return flagged.withColumn(id_col, f.coalesce(f.col('_mappedBiosampleId'), f.col(id_col))).drop(
        '_mappedBiosampleId', '_lookupId'
    )


def split_valid_invalid(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Split a flagged DataFrame into (valid, invalid) by emptiness of `qualityControls`."""
    return df.filter(f.size(QC_COLUMN) == 0), df.filter(f.size(QC_COLUMN) != 0)
