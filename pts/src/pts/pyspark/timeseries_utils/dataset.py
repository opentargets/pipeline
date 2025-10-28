"""Definition of the baseclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as f


@dataclass
class Dataset:
    """Base class of all datamodels."""

    _df: DataFrame

    # This is the default set of columns we group evidence by, used for overall associations:
    GROUPBY_COLUMNS: list[str] = field(default_factory=lambda: ['diseaseId', 'targetId', 'year'])

    # This is the default set of columns we window over when combining scores from different years:
    WINDOW_COLUMNS: list[str] = field(default_factory=lambda: ['diseaseId', 'targetId'])

    # Optional aggregate column:
    AGGREGATE_COLUMN: str | None = None

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
            self.validate_columns(self._df, self.MANDATORY_COLUMNS)

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

    @staticmethod
    def _windowing_around_peak(peak: Column, window: int) -> Column:
        """Window around peak.

        Args:
            peak (Column): peak column to window.
            window (int): size of the window.

        Returns:
            Column: windowed peak.
        """
        return f.posexplode(f.sequence(peak, peak + f.lit(window))).alias('year-peakYear', 'year')

    def filter(self: Self, filter_expression: Column) -> Self:
        """Filter dataset based on a spark expression.

        Args:
            filter_expression (Column): Spark filter expression.

        Returns:
            Self: the same dataset with filtered content.
        """
        return type(self)(self.df.filter(filter_expression))
