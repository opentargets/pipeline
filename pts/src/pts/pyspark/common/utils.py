import importlib.resources as pkg_resources
import json
from collections.abc import Callable, Iterable, Mapping
from enum import Enum
from functools import wraps
from typing import Any, TypeVar, cast

import numpy as np
import pandas as pd
import polars as pl
import pyspark.sql.functions as f
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    ByteType,
    DataType,
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from pts import schemas

F = TypeVar('F', bound=Callable[..., Any])

# The inverse of the spark->polars mapping that `pts/schemas/evidence.json` used to be read
# through, kept exhaustive over that mapping so a schema round-trips unchanged.
_POLARS_TO_SPARK_TYPE: dict[Any, type[DataType]] = {
    pl.Boolean: BooleanType,
    pl.Int8: ByteType,
    pl.Int16: ShortType,
    pl.Int32: IntegerType,
    pl.Int64: LongType,
    pl.Float32: FloatType,
    pl.Float64: DoubleType,
    pl.String: StringType,
    pl.Date: DateType,
    pl.Datetime: TimestampType,
}


def update_quality_flag(qc: Column, flag_condition: Column, flag_text: Enum) -> Column:
    """Update the provided quality control list with a new flag if condition is met.

    Args:
        qc (Column): Array column with the current list of qc flags.
        flag_condition (Column): This is a column of booleans, signing which row should be flagged
        flag_text (Enum): Text for the new quality control flag

    Returns:
        Column: Array column with the updated list of qc flags.


    Examples:
        >>> s = "study STRING, qualityControls ARRAY<STRING>, condition BOOLEAN"
        >>> d =  [("S1", ["qc1"], True), ("S2", ["qc3"], False)]
        >>> df = spark.createDataFrame(d, s)
        >>> df.show()
        +-----+---------------+---------+
        |study|qualityControls|condition|
        +-----+---------------+---------+
        |   S1|          [qc1]|     true|
        |   S2|          [qc3]|    false|
        +-----+---------------+---------+
        <BLANKLINE>

        >>> class QC(Enum):
        ...     QC1 = "qc1"
        ...     QC2 = "qc2"
        ...     QC3 = "qc3"

        >>> condition = f.col("condition")
        >>> new_qc = update_quality_flag(f.col("qualityControls"), condition, QC.QC2)
        >>> df.withColumn("qualityControls", new_qc).show()
        +-----+---------------+---------+
        |study|qualityControls|condition|
        +-----+---------------+---------+
        |   S1|     [qc1, qc2]|     true|
        |   S2|          [qc3]|    false|
        +-----+---------------+---------+
        <BLANKLINE>
    """
    qc = f.when(qc.isNull(), f.array()).otherwise(qc)
    return f.when(
        flag_condition,
        f.array_sort(f.array_distinct(f.array_union(qc, f.array(f.lit(flag_text.value))))),
    ).otherwise(qc)


def required_columns(cols: Iterable[str]) -> Callable[[F], F]:
    """Decorator to ensure that all required columns are present in the DataFrame (`self.df`) before executing a method.

    This decorator is typically used inside ETL pipeline classes that
    hold a Spark DataFrame as an attribute (`self.df`). If any of the
    required columns are missing, it raises a `ValueError` before
    calling the decorated method.

    Args:
        cols (Iterable[str]):
            A list or iterable of required column names that must exist
            in `self.df.columns`.

    Returns:
        Callable[[F], F]:
            The decorated function with column presence validation.

    Raises:
        ValueError: If any required column is missing from `self.df.columns`.


    Example:
        >>> class EvidenceETL:
        ...     def __init__(self, df: DataFrame) -> None:
        ...         self.df = df
        ...     @required_columns(["gene_id", "pvalue"])
        ...     def score(self) -> DataFrame:
        ...         return self.df.withColumn("score", self.df.pvalue * 2)
        >>> test_df = spark.createDataFrame([('a',),], ['test_col'])
        >>> etl = EvidenceETL(test_df)
        >>> etl.score()  # Raises ValueError if 'gene_id' or 'pvalue' is missing
        Traceback (most recent call last):
        ...
        ValueError: Columns ['gene_id', 'pvalue'] required by "score" are missing. Available columns: ['test_col']
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self, *args: Any, **kwargs: Any) -> Any:
            if not hasattr(self, 'df') or not isinstance(self.df, DataFrame):
                func_name = getattr(func, '__name__', repr(func))
                raise AttributeError(f'Function "{func_name}" expects "self.df" to be a Spark DataFrame.')

            data_cols = self.df.columns
            missing = [col for col in cols if col not in data_cols]
            if missing:
                func_name = getattr(func, '__name__', repr(func))
                raise ValueError(
                    f'Columns {missing} required by "{func_name}" are missing. Available columns: {sorted(data_cols)}'
                )

            return func(self, *args, **kwargs)

        return cast(F, wrapper)

    return decorator


def parse_spark_schema(schema_json: str) -> StructType:
    """Parse Spark schema from JSON.

    Args:
        schema_json (str): JSON filename containing spark schema in the schemas package

    Returns:
        StructType: Spark schema
    """
    pkg = pkg_resources.files(schemas).joinpath(schema_json)
    with pkg.open(encoding='utf-8') as schema:
        core_schema = json.load(schema)

    return StructType.fromJson(core_schema)


def polars_type_to_spark(dtype: Any) -> DataType:
    """Convert a Polars dtype into its Spark equivalent.

    Args:
        dtype (Any): a Polars dtype, either a class (`pl.String`) or an instance (`pl.List(...)`).

    Returns:
        DataType: the equivalent Spark type.

    Raises:
        ValueError: if the dtype has no Spark equivalent here, rather than silently substituting
            one -- a wrong type in a read schema changes what spark parses, not just how it is
            labelled.
    """
    if isinstance(dtype, pl.List):
        return ArrayType(polars_type_to_spark(dtype.inner))

    if isinstance(dtype, pl.Struct):
        # nullable=True throughout: spark's own `StructType.fromJson` defaulted every evidence
        # field to nullable, and narrowing that would make spark reject rows it used to read.
        return StructType([
            StructField(field.name, polars_type_to_spark(field.dtype), nullable=True) for field in dtype.fields
        ])

    spark_type = _POLARS_TO_SPARK_TYPE.get(dtype.base_type() if isinstance(dtype, pl.DataType) else dtype)
    if spark_type is None:
        msg = f'Unsupported Polars type: {dtype}'
        raise ValueError(msg)
    return spark_type()


def polars_schema_to_spark(schema: Mapping[str, Any]) -> StructType:
    """Convert a Polars schema mapping into a Spark `StructType`.

    Lets `pts.schemas` stay polars-native (see `pts/schemas/evidence.py`) while the remaining
    pyspark jobs still get the `StructType` their readers need. Field order is preserved, since
    the mapping's order is the schema's column order.

    Args:
        schema (Mapping[str, Any]): field name to Polars dtype, in column order.

    Returns:
        StructType: the equivalent Spark schema.
    """
    return StructType([
        StructField(name, polars_type_to_spark(dtype), nullable=True) for name, dtype in schema.items()
    ])


def maybe_coalesce(df: DataFrame, n: int | None) -> DataFrame:
    """Coalesce a DataFrame to a fixed partition count when configured.

    Used at write boundaries to avoid emitting many small parquet files when
    the upstream shuffle partition count is high. Coalesce is a narrow
    transformation that merges partitions without a network shuffle, so it
    only reduces file count -- it does not rebalance data.

    Args:
        df: DataFrame to coalesce.
        n: Target partition count. Falsy values leave ``df`` unchanged.

    Returns:
        The coalesced DataFrame, or ``df`` unchanged when ``n`` is falsy.
    """
    if n:
        return df.coalesce(n)
    return df


def maybe_repartition(df: DataFrame, n: int | None) -> DataFrame:
    """Repartition a DataFrame to a fixed partition count when configured.

    Used after a read of scattered small files to keep downstream Spark task
    counts sane. Repartition is a wide transformation that hash-shuffles rows
    across partitions, so it both rebalances data and sets the partition count
    -- the cost is one extra shuffle stage.

    Args:
        df: DataFrame to repartition.
        n: Target partition count. Falsy values leave ``df`` unchanged.

    Returns:
        The repartitioned DataFrame, or ``df`` unchanged when ``n`` is falsy.
    """
    if n:
        return df.repartition(n)
    return df


def _cell_to_string(value: Any) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool | np.bool_):
        return '1' if value else '0'
    return str(value)


def pandas_to_spark_strings(spark: SparkSession, pdf: pd.DataFrame, columns: list[str] | None = None) -> DataFrame:
    """Convert a pandas frame to Spark with every cell as a string and blanks as null.

    Arrow rejects a column that mixes numbers and text, so spreadsheet cells cross as
    strings and the caller casts the columns it uses. Booleans become '1'/'0', so a
    one-hot flag casts to a number either way.

    Args:
        spark: Spark session.
        pdf: pandas frame, e.g. a sheet from ``pd.read_excel``.
        columns: Columns to keep. All columns when omitted.

    Returns:
        DataFrame with one nullable string column per kept column.
    """
    if columns is not None:
        pdf = pdf[columns]
    values = pdf.astype(object).map(_cell_to_string)
    schema = StructType([StructField(c, StringType(), True) for c in values.columns])
    return spark.createDataFrame(values, schema=schema)
