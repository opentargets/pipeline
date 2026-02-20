"""Definition of Association class."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql.window import Window, WindowSpec

from pts.pyspark.associations_utils.dataset import Dataset


@dataclass
class Association(Dataset):
    """Definition of associations."""

    # Mandatory association column:
    MANDATORY_COLUMNS: ClassVar[list[str]] = [
        'diseaseId',
        'targetId',
        'year',
        'associationScore',
        'aggregationType',
        'aggregationValue',
        'yearlyEvidenceCount',
    ]

    # This is the default set of columns we group evidence by, used for association calculation:
    GROUPBY_COLUMNS: ClassVar[list[str]] = ['diseaseId', 'targetId', 'aggregationType', 'aggregationValue']

    def aggregate_overall(self: Association, datasource_weights: DataFrame) -> Association:
        """Apply overall aggregation on the datasource data.

        Args:
            datasource_weights (DataFrame): for each datasource we need to provide weights and corresponding datatype

        Returns:
            Association: where the datasources are further aggregated fully.
        """
        # Aggregate value expression:
        aggregation_type_expression = f.lit('overall')
        aggregation_value_expression = f.lit('None')

        # Association scores are weighted by datasource weight:
        score_expression = f.col('associationScore') * f.col('weight')

        # To calculate the association score we normalise with the maximum value of the harmonic sum:
        association_expression = self._get_harmonic_sum(f.col('datasourceMaxScores')) / self.MAX_HARMONIC_SUM

        return self._aggregate_associations(
            aggregation_type_expression,
            aggregation_value_expression,
            score_expression,
            association_expression,
            datasource_weights,
        )

    def aggregate_by_datatype(self: Association, datasource_weights: DataFrame) -> Association:
        """Apply further aggregation based on data types.

        Args:
            datasource_weights (DataFrame): for each datasource we need to provide weights and corresponding datatype

        Returns:
            Association: where the datasources are further aggregated by datatypes.
        """
        # Aggregate value expression:
        aggregation_type_expression = f.lit('datatypeId')
        aggregation_value_expression = f.col('datatypeId')

        # Association scores are not weighted by datasource weitht:
        score_expression = f.col('associationScore')

        # To calculate the association score, we normalise by the theoretical maximum of the harmonic sum:
        association_expression = self._get_harmonic_sum(f.col('datasourceMaxScores')) / self._get_harmonic_sum(
            f.array_repeat(f.lit(1.0), f.size('datasourceMaxScores'))
        )

        return self._aggregate_associations(
            aggregation_type_expression,
            aggregation_value_expression,
            score_expression,
            association_expression,
            datasource_weights,
        )

    def _aggregate_associations(
        self: Association,
        aggregation_type_expression: Column,
        aggregation_value_expression: Column,
        score_expression: Column,
        association_expression: Column,
        datasource_weights: DataFrame,
    ) -> Association:
        """Further aggregate association data.

        Upon aggregating evidence by datasource, we get an "Association" object. This, however
        can be further aggregated by datatypes or overall. These further aggregation would yield
        datasets with identical schema.

        Args:
            aggregation_type_expression (Columns): column expression describing how the data is aggregated
            aggregation_value_expression (Columns): column expression describing aggregated group
            score_expression (Column): column expression describing how the scores are processed
            association_expression (Column): expression describing how the association score is computed
            datasource_weights (DataFrame): for each datasource we need to provide weights and corresponding datatype

        Returns:
            Association object. associations with type of aggregation, with more or less rows.
        """
        collect_window = (
            Window.partitionBy(['targetId', 'diseaseId', 'aggregationValue'])
            .orderBy(f.col('year').asc())
            .rowsBetween(Window.unboundedPreceding, 0)
        )

        return Association(
            self.df.select(
                'targetId',
                'diseaseId',
                'yearlyEvidenceCount',
                'associationScore',
                'year',
                f.col('aggregationValue').alias('datasourceId'),
            )
            .join(datasource_weights, on='datasourceId', how='inner')
            .select(
                'targetId',
                'diseaseId',
                'year',
                score_expression.alias('associationScore'),
                aggregation_type_expression.alias('aggregationType'),
                aggregation_value_expression.alias('aggregationValue'),
                'yearlyEvidenceCount',
                'datasourceId',
            )
            .drop('weight')
            .withColumn(
                'dataset_scores',
                f.collect_list(f.struct('datasourceId', 'associationScore')).over(collect_window),
            )
            .withColumn('datasourceMaxScores', self._retain_max_scores(f.col('dataset_scores')))
            .withColumn('associationScore', association_expression)
            .groupby('targetId', 'diseaseId', 'aggregationValue', 'year', 'aggregationType')
            .agg(
                f.max('associationScore').alias('associationScore'),
                f.sum('yearlyEvidenceCount').alias('yearlyEvidenceCount'),
            )
            .orderBy('targetId', 'diseaseId', 'year')
        )

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

        # Next year:
        current_year = datetime.now().year

        return (
            intermediate_dataset
            # add max novelty values to original dataframe disease-target-year
            .join(
                novelty,
                [*self.GROUPBY_COLUMNS, 'year'],
                'left',
            )
            # Remove the future date from time series - future dates represent evidence with no evidence date:
            .withColumns({
                'year': f.when(f.col('year') <= current_year, f.col('year')).otherwise(None),
                'novelty': f.when(f.col('year') <= current_year, f.col('novelty')).otherwise(None),
            })
            # Collate the yearly associations into a single association object:
            .groupby(self.GROUPBY_COLUMNS)
            .agg(
                # Get the highest association value:
                f.max(f.col('associationScore')).alias('associationScore'),
                # Get the total number of supporting evidence:
                f.sum('yearlyEvidenceCount').alias('evidenceCount'),
                # Get the timeseries object:
                f.collect_list(f.struct('year', 'associationScore', 'novelty', 'yearlyEvidenceCount')).alias(
                    'timeseries'
                ),
            )
            # Adding current novelty, which allows for quick filtering:
            .withColumns({
                # The current novelty is the novelty in the current year:
                'currentNovelty': f.filter('timeseries', lambda x: x.year == current_year)[0].novelty,
                'aggregationValue': f.when(f.col('aggregationValue') != 'NA', f.col('aggregationValue')),
                'associationScore': f.filter('timeseries', lambda x: x.year.isNull())[0].associationScore,
            })
        )

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
            ...    .orderBy('label', 'year')
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
        last_year = datetime.now().year + 1
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
                'associationScore',
                f.coalesce(
                    f.last('associationScore', ignorenulls=True).over(
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
            .withColumn('peak', Association._get_peak(f.col('associationScore'), window_spec))
            .withColumnRenamed('year', 'associationYear')
            # Drawing around the window:
            .select('*', Association._windowing_around_peak(f.col('associationYear'), novelty_window))
            # Grouping data again:
            .groupBy([*groupby_columns, 'year'])
            .agg(
                Association.calculate_logistic_decay(
                    score_value=f.col('peak'),
                    window_difference=f.col('year-peakYear'),
                    sigmoid_midpoint=novelty_shift,
                    decay_steepness=novelty_scale,
                ).alias('novelty')
            )
        )

    @staticmethod
    def calculate_logistic_decay(
        score_value: Column, window_difference: Column, sigmoid_midpoint: float, decay_steepness: float
    ) -> Column:
        """Calculate change of novelty over the years.

        After getting more evidence, the increasing association score imporoves novelty,
        which decays in subsequent years following a logistic fashion.
        More information on novelty calculation: https://www.researchsquare.com/article/rs-5669559/v1

        Args:
            score_value (Column): the latest score shift registered
            window_difference (Column): year difference when the last score shift was registered
            sigmoid_midpoint (float): sigmoid decay curve midpoint
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
            ...        f.col("peak"), f.col("year-peakYear"), 1.0, 0.5
            ...     ).alias("novelty")
            ... ).orderBy('year').show(truncate=False)
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

    @staticmethod
    def _retain_max_scores(array_col: Column) -> Column:
        """Return an array of structs with unique datasourceId and max associationScore.

        Reasoning behind this method:
        - Association dataset has a year dimension.
        - Each row tells that in that given year, as new evidence came in, the datasource specific
          association score was a given value.
        - When summarizing associations across different datasources in a given year, we have to look back
          in time, as some of the datasources might not have new evidence/association score in that year.
        - If we look back, datasources will be repeated, but we only need to account for the latest association
          score, which has to be the highest because over time more evidence has to yield higher
          association score.

        Args:
            array_col (Column): column with the association scores.

        Examples:
            >>> from pyspark.sql import functions as f, types as t
            >>> data = [
            ...     ([
            ...         {"datasourceId": "source1", "associationScore": 0.5}, # <= this needs to be ignored!
            ...         {"datasourceId": "source2", "associationScore": 0.8},
            ...         {"datasourceId": "source1", "associationScore": 0.7},
            ...     ],),
            ... ]
            >>> schema = t.StructType([
            ...     t.StructField("scores", t.ArrayType(
            ...         t.StructType([
            ...             t.StructField("datasourceId", t.StringType(), True),
            ...             t.StructField("associationScore", t.DoubleType(), True),
            ...         ])
            ...     ), True)
            ... ])
            >>> df = spark.createDataFrame(data, schema)
            >>> result = df.select(Association._retain_max_scores(f.col("scores")).alias("result")).collect()
            >>> sorted(result[0]["result"])
            [0.7, 0.8]
        """
        empty_arr = f.array().cast(
            t.ArrayType(
                t.StructType([
                    t.StructField('datasourceId', t.StringType(), True),
                    t.StructField('associationScore', t.DoubleType(), True),
                ])
            )
        )
        max_scores = f.aggregate(
            array_col,
            empty_arr,
            lambda acc, x: f.when(
                f.exists(acc, lambda a: a.datasourceId == x.datasourceId),
                f.transform(
                    acc,
                    lambda a: f.when(
                        a.datasourceId == x.datasourceId,
                        f.struct(
                            a.datasourceId.alias('datasourceId'),
                            f.greatest(a.associationScore, x.associationScore).alias('associationScore'),
                        ),
                    ).otherwise(a),
                ),
            ).otherwise(f.concat(acc, f.array(x))),
        )

        return f.transform(max_scores, lambda datasource: datasource.getField('associationScore'))
