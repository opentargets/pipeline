"""Definition of Association class."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from pyspark.sql.window import Window, WindowSpec

from pts.pyspark.timeseries_utils.dataset import Dataset


@dataclass
class Association(Dataset):
    """Definition of associations."""

    # Mandatory association column:
    MANDATORY_COLUMNS: ClassVar[list[str]] = [
        'diseaseId',
        'targetId',
        'year',
        'yearlyEvidenceScores',
        'yearlyAssociationScore',
        'aggregationType',
        'aggregationValue',
    ]

    # This is the default set of columns we group evidence by, used for association calculation:
    GROUPBY_COLUMNS: ClassVar[list[str]] = ['diseaseId', 'targetId', 'aggregationType', 'aggregationValue']

    @staticmethod
    def _get_peak(scores: Column, window: WindowSpec) -> Column:
        """Identify peaks.

        Any time association score increases in subsequent years, that means that
        the given year the novelty peaks. If association scores don't change, the novelty is flat.

        Args:
            scores (Column): Column with the yearly assocation score based on all evidence UNTIL that year.
            window (WindowSpec): Depending on the level of aggregation the windowing cannot be fixed.

        Returns:
            Column: Change in the score values in subsequent years.

        Examples:
            >>> w = Window.partitionBy('label').orderBy('year')
            >>> (
            ...    spark.createDataFrame(
            ...         [
            ...            ('a', 1990, 0.0),
            ...            ('a', 1991, 0.1),
            ...            ('a', 1992, 0.1),
            ...            ('a', 1993, 0.2),
            ...            ('b', 1989, 0.0),
            ...            ('b', 1990, 0.2)
            ...         ],
            ...         ['label', 'year', 'score']
            ...    )
            ...    .select("*", Association._get_peak(f.col('score'), w).alias('col'))
            ...    .show()
            ... )
            +-----+----+-----+---+
            |label|year|score|col|
            +-----+----+-----+---+
            |    a|1990|  0.0|0.0|
            |    a|1991|  0.1|0.1|
            |    a|1992|  0.1|0.0|
            |    a|1993|  0.2|0.1|
            |    b|1989|  0.0|0.0|
            |    b|1990|  0.2|0.2|
            +-----+----+-----+---+
            <BLANKLINE>
        """
        peak_value = scores - f.lag(scores, offset=1).over(window)

        return f.when(peak_value.isNull(), f.lit(0)).otherwise(peak_value)

    @staticmethod
    def _create_yearly_view(df) -> DataFrame:
        """Create yearly view on Assocations.

        A sequence of all years generated between (first evidence - 5 year) and (current year).

        Args:
            df (DataFrame): evidence dataframe

        Returns:
            DataFrame:
        """
        last_year = datetime.now().year
        window_spec = (
            Window.partitionBy('targetId', 'diseaseId')
            .orderBy('year')
            .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        )

        return (
            df.select(*Association.GROUPBY_COLUMNS, 'year')
            .distinct()
            .withColumn('first_evidence_year', f.min('year').over(window_spec))
            .select(*Association.GROUPBY_COLUMNS, 'first_evidence_year')
            .distinct()
            .withColumn(
                'years',
                f.sequence(
                    # First year depends on the first year with evidence:
                    f.col('first_evidence_year') - 5,
                    # Last year is fixed:
                    f.lit(last_year),
                ),
            )
            .select(*Association.GROUPBY_COLUMNS, f.explode('years').alias('year'))
        )

    @staticmethod
    def _back_fill_missing_years(association_df: DataFrame) -> DataFrame:
        """Fill in data for years where no evidence has arrived.

        In the association dataset scores are only available for years when evidence is available.
        For missing years, association and evidence data is propagated from earlier years.
        This will create a complete dataset telling the association score in any given year
        from the first year with evidence.

        Args:
            association_df (DataFrame): association dataframe to be processed.
            groupby_columns (list[str]): list of columns to group data by

        Return:
            DataFrame: where each missing year from the first year with evidence to current year is filled.
        """
        window_spec = Window.partitionBy(Association.GROUPBY_COLUMNS).orderBy('year')

        return (
            # Create a table with all possible target/disease/{optional aggregation column}/year
            Association._create_yearly_view(association_df)
            # Join with the complete association table - this join will leave multiple rows without association score.
            .join(association_df, on=[*Association.GROUPBY_COLUMNS, 'year'], how='left')
            # Filling missing association scores from previous non-null years:
            .withColumn(
                # Propagating non-null association scores from earlier years:
                'yearlyAssociationScore',
                f.coalesce(
                    f.last('yearlyAssociationScore', ignorenulls=True).over(
                        window_spec.rowsBetween(Window.unboundedPreceding, 0)
                    ),
                    f.lit(0),
                ),
            )
        )

    @staticmethod
    def _get_novelty(
        intermediate_dataset: DataFrame,
        groupby_columns: list[str],
        novelty_window: int,
        novelty_shift: int,
        novelty_scale: int,
    ) -> DataFrame:
        """Calculate novelty values.

        Args:
            intermediate_dataset (DataFrame): backfilled association dataset.
            groupby_columns (list[str]): list of columns to group data by.
            novelty_window (int): size of the window.
            novelty_shift (int): Novelty shift
            novelty_scale (int): how quickly the novelty decays.

        Returns:
            DataFrame: novelty dataset.
        """
        window_spec = Window.partitionBy(groupby_columns).orderBy('year')
        return (
            intermediate_dataset
            # Marking peaks:
            .withColumn('peak', Association._get_peak(f.col('yearlyAssociationScore'), window_spec))
            .withColumnRenamed('year', 'associationYear')
            # Drawing around the window:
            .select(
                '*', Association._windowing_around_peak(f.col('associationYear'), novelty_window).alias('year-peakYear')
            )
            # Grouping data again:
            .groupBy([*groupby_columns, 'year'])
            .agg(
                Association.calculate_logistic_decay(
                    f.col('peak'), f.col('year-peakYear'), f.lit(novelty_scale), novelty_shift
                ).alias('novelty')
            )
        )

    @staticmethod
    def calculate_logistic_decay(
        score_value: Column, window_difference: Column, sigmoid_midpoint: Column, decay_steepness: float
    ) -> Column:
        """Calculate change of novelty over the years.

        After getting more evidence, the increasing association score imporoves novelty,
        which decays in subsequent years following a logistic fashion.
        More information on novelty calculation: https://www.researchsquare.com/article/rs-5669559/v1

        Args:
            score_value (Column): the latest score shift registered
            window_difference (Column): year difference when the last score shift was registered
            sigmoid_midpoint (Column): sigmoid decay curve midpoint
            decay_steepness (float): the logistic growth rate or steepness of the decay curve

        Returns:
            Column: representing the novelty value at a given year.

        Example:
            >>> data = [
            ...     ("A", 2020, 0.99, 0),
            ...     ("A", 2021, 0.99, 1),
            ...     ("A", 2022, 0.99, 2),
            ...     ("A", 2023, 0.99, 3),
            ...     ("A", 2023, 0.99, 4),
            ... ]
            >>> df = spark.createDataFrame(data, ["group", "year", "peak", "year-peakYear"])
            >>> df.groupBy("group", "year").agg(
            ...     Association.calculate_logistic_decay(
            ...        f.col("peak"), f.col("year-peakYear"), f.lit(1.0), 0.5
            ...     ).alias("novelty")
            ... ).show(truncate=False)
            +-----+----+-------------------+
            |group|year|novelty            |
            +-----+----+-------------------+
            |A    |2020|0.616234737889836  |
            |A    |2021|0.495              |
            |A    |2022|0.373765262110164  |
            |A    |2023|0.26625200715629516|
            +-----+----+-------------------+
            <BLANKLINE>
        """
        return f.max(score_value / (1 + f.exp(decay_steepness * (window_difference - f.lit(sigmoid_midpoint)))))

    def compute_novelty(
        self: Association,
        novelty_window: int = 10,
        novelty_scale: int = 2,
        novelty_shift: int = 2,
    ) -> DataFrame:
        """Calculate novelty.

        Args:
            novelty_window (int): size of the window. Default value 10.
            novelty_scale (int): how quickly the novelty decays. Default value 2.
            novelty_shift (int): Novelty shift

        Returns:
            DataFrame: novelty dataset.
        """
        # Generate a complete dataset with filled data for missing years:
        intermediate_dataset = self._back_fill_missing_years(self.df.na.fill({'aggregationValue': 'NA'}))

        # Calculate novelty based on subsequent years:
        novelty = self._get_novelty(
            intermediate_dataset, self.GROUPBY_COLUMNS, novelty_window, novelty_shift, novelty_scale
        )

        return (
            # spark.read.parquet('test_association')
            intermediate_dataset
            # add max. novelty values to original dataframe disease-target-year
            .join(
                novelty,
                [*self.GROUPBY_COLUMNS, 'year'],
                'left',
            )
            .filter(f.col('novelty').isNotNull())
            .groupby(self.GROUPBY_COLUMNS)
            .agg(
                f.max(f.col('yearlyAssociationScore')).alias('associationScore'),
                f.collect_list(f.struct('year', 'yearlyAssociationScore', 'novelty', 'yearlyEvidenceScores')).alias(
                    'timeseries'
                ),
            )
            # Adding current novelty, which allows for quick filtering:
            .withColumns({
                'currentNovelty': f.filter('timeseries', lambda x: x.year == datetime.now().year)[0].novelty,
                'aggregationValue': f.when(f.col('aggregationValue') != 'NA', f.col('aggregationValue')),
            })
        )

    @staticmethod
    def _windowing_around_peak(peak: Column, window: int) -> Column:
        """Window around peak.

        Tells in a given year, how many years has passed since the peak year.

        Args:
            peak (Column): peak column to window.
            window (int): size of the window.

        Returns:
            Column: windowed peak.

        Examples:
        >>> df = spark.createDataFrame([(1990,), (1991,), (1992,)], ['peak'])
        >>> df.select(Association._windowing_around_peak(f.col('peak'), 2)).show()
        +-------------+----+
        |year-peakYear|year|
        +-------------+----+
        |            0|1990|
        |            1|1991|
        |            2|1992|
        |            0|1991|
        |            1|1992|
        |            2|1993|
        |            0|1992|
        |            1|1993|
        |            2|1994|
        +-------------+----+
        <BLANKLINE>
        """
        return f.posexplode(f.sequence(peak, peak + f.lit(window))).alias('year-peakYear', 'year')
