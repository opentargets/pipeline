from collections.abc import Callable, Iterable
from enum import Enum
from functools import wraps
from math import log10
from typing import TYPE_CHECKING, Any, TypeVar, cast

import pyspark.sql.functions as f
from pyspark.sql import Column, DataFrame

F = TypeVar('F', bound=Callable[..., Any])

if TYPE_CHECKING:
    from enum import Enum


class GenerateDiseaseCellLines:
    """Generate "diseaseCellLines" object from a cell passport file.

    !!!
    There's one important bit here: I have noticed that we frequenty get cell line names
    with missing dashes. Therefore the cell line names are cleaned up by removing dashes.
    It has to be done when joining with other datasets.
    !!!

    Args:
        cell_passport_file: Path to the cell passport file.
    """

    def __init__(
        self,
        cell_passport_data: DataFrame,
        cell_line_to_uberon_mapping: DataFrame,
    ) -> None:
        self.cell_passport_data = cell_passport_data
        self.tissue_to_uberon_map = cell_line_to_uberon_mapping

    def generate_disease_cell_lines(self) -> DataFrame:
        """Reading and procesing cell line data from the cell passport file.

        The schema of the returned dataframe is:

        root
        |-- name: string (nullable = true)
        |-- id: string (nullable = true)
        |-- biomarkerList: array (nullable = true)
        |    |-- element: struct (containsNull = true)
        |    |    |-- name: string (nullable = true)
        |    |    |-- description: string (nullable = true)
        |-- diseaseCellLine: struct (nullable = false)
        |    |-- tissue: string (nullable = true)
        |    |-- name: string (nullable = true)
        |    |-- id: string (nullable = true)
        |    |-- tissueId: string (nullable = true)

        Note:
            * Microsatellite stability is the only inferred biomarker.
            * The cell line name has the dashes removed.
            * Id is the cell line identifier from Sanger
            * Tissue id is the UBERON identifier for the tissue, based on manual curation.
        """
        cell_df = (
            self.cell_passport_data.select(
                f.col('model_name').alias('name'),
                f.col('model_id').alias('id'),
                f.lower(f.col('tissue')).alias('tissueFromSource'),
                f.array(self.parse_msi_status(f.col('msi_status'))).alias('biomarkerList'),
            )
            .join(self.tissue_to_uberon_map, on='tissueFromSource', how='left')
            .persist()
        )
        return cell_df.select(
            f.regexp_replace(f.col('name'), '-', '').alias('name'),
            'id',
            'biomarkerList',
            f.struct(f.col('tissueName').alias('tissue'), 'name', 'id', 'tissueId').alias('diseaseCellLine'),
        )

    @staticmethod
    def parse_msi_status(status: Column) -> Column:
        """Based on the content of the MSI status, we generate the corresponding biomarker object."""
        return f.when(
            status == 'MSI',
            f.struct(
                f.lit('MSI').alias('name'),
                f.lit('Microsatellite instable').alias('description'),
            ),
        ).when(
            status == 'MSS',
            f.struct(
                f.lit('MSS').alias('name'),
                f.lit('Microsatellite stable').alias('description'),
            ),
        )


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

    Raises:
        ValueError: If any required column is missing from `self.df.columns`.

    Returns:
        Callable[[F], F]:
            The decorated function with column presence validation.

    Example:
        >>> class EvidenceETL:
        ...     def __init__(self, df: DataFrame):
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
                raise AttributeError(f'Function "{func.__name__}" expects "self.df" to be a Spark DataFrame.')

            data_cols = self.df.columns
            missing = [col for col in cols if col not in data_cols]
            if missing:
                raise ValueError(
                    f'Columns {missing} required by "{func.__name__}" are missing. '
                    f'Available columns: {sorted(data_cols)}'
                )

            return func(self, *args, **kwargs)

        return cast(F, wrapper)

    return decorator


def linear_rescaling(
    value: float, in_range_min: float, in_range_max: float, out_range_min: float, out_range_max: float
) -> float:
    """Linearly rescaling a value across ranges.

    Args:
        value (float): value to rescale
        in_range_min (float): minimum of the starting range
        in_range_max (float): maximum of the starting range
        out_range_min (float): minimum of the resulting range
        out_range_max (float): maximum of the resulting range

    Returns:
        float: re-scaled value

    Examples:
        >>> # normal mapping: 5 maps from [0,10] -> [0,1] to 0.5
        >>> linear_rescaling(5.0, 0.0, 10.0, 0.0, 1.0)
        0.5

        >>> # values below/above input range are clamped to the output range
        >>> linear_rescaling(-10.0, 0.0, 10.0, 0.0, 1.0)
        0.0
        >>> linear_rescaling(20.0, 0.0, 10.0, 0.0, 1.0)
        1.0

        >>> # input range degenerate (in_range_min == in_range_max) and
        >>> # output range degenerate (out_range_min == out_range_max):
        >>> # score is set to the input value then clamped to the degenerate output range
        >>> linear_rescaling(1.0, 1.0, 1.0, 2.0, 2.0)
        2.0

        >>> # input range degenerate but output range non-degenerate:
        >>> # returns out_range_min (per implementation)
        >>> linear_rescaling(1.0, 1.0, 1.0, 0.0, 1.0)
        0.0
    """
    delta1 = float(in_range_max - in_range_min)
    delta2 = float(out_range_max - out_range_min)

    if delta1 != 0.0:
        score = (float(delta2) * (value - float(in_range_min)) / delta1) + float(out_range_min)
    elif delta1 == 0.0 and delta2 == 0.0:
        score = value
    else:
        score = out_range_min

    # clamp to [out_range_min, out_range_max]
    return min(max(score, out_range_min), out_range_max)


def pvalue_linear_rescaling(
    pvalue: float,
    in_range_min: float = 1.0,
    in_range_max: float = 1e-10,
    out_range_min: float = 0.0,
    out_range_max: float = 1.0,
) -> float:
    """Convert p-value to log10 space and apply linear_rescaling.

    Args:
        pvalue (float): p-value to rescale
        in_range_min (float): minimum of the starting p-value range
        in_range_max (float): maximum of the starting p-value range
        out_range_min (float): minimum of the resulting range
        out_range_max (float): maximum of the resulting range

    Returns:
        float: re-scaled value

    Examples:
    Examples:
        >>> # p = 1  -> log10(1) == 0 -> maps to out_range_min (0.0)
        >>> pvalue_linear_rescaling(1.0)
        0.0

        >>> # p = 1e-10 -> log10 == -10 -> maps to out_range_max (1.0)
        >>> pvalue_linear_rescaling(1e-10)
        1.0

        >>> # p = 1e-5 -> log10 == -5 -> halfway between 0 and -10 -> 0.5
        >>> pvalue_linear_rescaling(1e-5)
        0.5

        >>> # p smaller than in_range_max: maps beyond out_range_max but clamped to 1.0
        >>> pvalue_linear_rescaling(1e-12)
        1.0

        >>> # p larger than in_range_min: would map below 0.0 but is clamped to 0.0
        >>> pvalue_linear_rescaling(2.0)
        0.0
    """
    pvalue_log = log10(pvalue)
    in_min_log = log10(in_range_min)
    in_max_log = log10(in_range_max)

    return linear_rescaling(pvalue_log, in_min_log, in_max_log, out_range_min, out_range_max)
