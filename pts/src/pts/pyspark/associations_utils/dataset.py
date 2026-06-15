"""Definition of the baseclass."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Self

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as f


@dataclass
class Dataset:
    """Base class of all datamodels."""

    _df: DataFrame

    # The theoretical maximum of harmonic sum for score normalisation:
    MAX_HARMONIC_SUM: float = reduce(lambda acc, iv: acc + iv[1] / iv[0] ** 2, enumerate([1] * 1000, start=1), 0.0)

    def __post_init__(
        self: Dataset,
    ) -> None:
        """Initialise an instance of Dataset class."""
        # Validating type:
        match self._df:
            case DataFrame():
                pass
            case _:
                raise TypeError(f'Invalid type for _df: {type(self._df)}')

        # Validating required columns:
        if hasattr(self, 'MANDATORY_COLUMNS'):
            self.validate_columns(self._df, self.MANDATORY_COLUMNS)  # ty:ignore[invalid-argument-type]

    @staticmethod
    def validate_columns(df: DataFrame, required_columns: list[str]) -> None:
        """Validate dataframe for the presence of a given set of columns.

        Args:
            df (DataFrame): input dataframe that needs to be validated.
            required_columns (list[str]): list of column names to validate.

        Raises:
            ValueError, Required column: {column} is missing from dataset
        """
        if required_columns is None:
            pass

        dataframe_columns = df.columns

        for column in required_columns:
            if column not in dataframe_columns:
                raise ValueError(f'Required column: {column} is missing from dataset!')

    @property
    def df(self: Dataset) -> DataFrame:
        """Dataframe included in the Dataset.

        Returns:
            DataFrame: Dataframe included in the Dataset
        """
        return self._df

    @classmethod
    def from_parquet(
        cls: type[Self],
        spark: SparkSession,
        path: str | list[str],
    ) -> Self:
        """Reads parquet into a Dataset with a given schema.

        Args:
            spark (SparkSession): Spark session
            path (str | list[str]): Path to the parquet dataset.

        Returns:
            Self: Dataset with the parquet file contents.

        Raises:
            TypeError: Provided input is not a string or list of string.
            ValueError: Parquet file is empty.
        """
        if isinstance(path, list):
            df = spark.read.parquet(*path)
        elif isinstance(path, str):
            df = spark.read.parquet(path)
        else:
            raise TypeError('Provided input is not a string or list of string.')

        # Raise error if the loaded dataset is empty:
        if df.isEmpty():
            raise ValueError(f'Parquet file is empty: {path}')

        return cls(df)

    def filter(self: Self, filter_expression: Column) -> Self:
        """Filter dataset based on a spark expression.

        Args:
            filter_expression (Column): Spark filter expression.

        Returns:
            Self: the same dataset with filtered content.
        """
        return type(self)(self.df.filter(filter_expression))

    @staticmethod
    def _get_harmonic_sum(collected_scores: Column) -> Column:
        """Calculating harmonic sum on an array of score values.

        Process:
            1. Sorting collected scores
            2. Get top 50 scores
            3. get weighted indices
            4. compute harmonic sum of scores
            5. Normalise harmonic sum

        Args:
            collected_scores (Column): list of floats in a spark columns

        Returns:
            Column: float normalised harmonic sum of the input array

        Examples:
            >>> (
            ...     spark.createDataFrame([([0.1, 0.5, 0.1, 0.1, 0.9, 0.85],)], ['evidenceScores'])
            ...     .select(Dataset._get_harmonic_sum(f.col("evidenceScores")).alias('col'))
            ...     .show()
            ... )
            +------------------+
            |               col|
            +------------------+
            |1.1810833333333335|
            +------------------+
            <BLANKLINE>
        """
        # For the harmonic sum calculation we only use the highest 50 scores in an ordered fashion:
        sorted_filtered_scores = Dataset._get_top_high_scores(collected_scores)

        # For each of the score values, we generate the corresponding index:
        indices = Dataset._get_score_indices(sorted_filtered_scores)

        weighted_scores = f.transform(
            f.arrays_zip(sorted_filtered_scores, indices),
            lambda pair: pair.sorted_filtered_scores / f.pow(pair.indices, f.lit(2)),
        )

        return f.aggregate(weighted_scores, f.lit(0.0), lambda acc, x: acc + x)  # noqa: FURB118  # PySpark requires a lambda here

    @staticmethod
    def _get_top_high_scores(collected_scores: Column, max_array_size: int = 50) -> Column:
        """Get the TOP n scores in an ordered array.

        Args:
            collected_scores (Column): array column with scores.
            max_array_size (int): the maximum size of the returned array.

        Returns:
            Column: ordered array of the top scores, size capped.

        Examples:
            >>> from pyspark.sql import Row
            >>> df = (
            ...     spark.createDataFrame([([0.1, 0.5, 0.1, 0.1, 0.9, 0.85],)], ['value'])
            ...     .select(f.explode(Dataset._get_top_high_scores(f.col("value"), 3).alias('returned')))
            ...     .persist()
            ... )
            >>> assert df.first() == Row(scores=0.9)
            >>> assert df.count() == 3
        """
        return f.slice(f.reverse(f.array_sort(collected_scores)), 1, max_array_size).alias('sorted_filtered_scores')

    @staticmethod
    def _get_score_indices(sorted_filtered_scores: Column) -> Column:
        """Get index array equal in size to the score array.

        Args:
            sorted_filtered_scores (Column): Top, ordered scores.

        Returns:
            Column: array of a sequence of intergers from 1, size as the corresponding scores array

        Examples:
            >>> df = (
            ...     spark.createDataFrame([([0.1, 0.5, 0.2],)], ['scores'])
            ...     .select(Dataset._get_score_indices(f.col('scores')))
            ... )
            >>> df.first()
            Row(indices=[1, 2, 3])

            >>> # works with larger arrays as well
            >>> df2 = (
            ...     spark.createDataFrame([([0.9, 0.85, 0.5, 0.2],)], ['scores'])
            ...     .select(Dataset._get_score_indices(f.col('scores')))
            ... )
            >>> df2.first()
            Row(indices=[1, 2, 3, 4])
        """
        return f.sequence(f.lit(1), f.size(sorted_filtered_scores)).alias('indices')
