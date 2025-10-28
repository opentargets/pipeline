"""Definition of Association class."""

from __future__ import annotations
from functools import reduce

from dataclasses import dataclass, field
from datetime import datetime

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql.window import Window, WindowSpec

from .dataset import Dataset



@dataclass
class Association(Dataset):
    """Definition of associations."""

    # Mandatory association column:
    MANDATORY_COLUMNS: list[str] = field(
        default_factory=lambda: [
            'diseaseId',
            'targetId',
            'year',
            'yearlyEvidenceScores',
            'retrospectiveEvidenceScores',
            'yearlyAssociationScore',
            'aggregationType',
            'aggregationValue'
        ]
    )

    # This is the default set of columns we group evidence by, used for association calculation:
    GROUPBY_COLUMNS: list[str] = field(
        default_factory=lambda: ['diseaseId', 'targetId', 'aggregationType', 'aggregationValue']
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
            ...         [('a', 1990, 0.0),('a', 1991, 0.1), ('a', 1992, 0.1), ('a', 1993, 0.2),],
            ...         ['label', 'year', 'score']
            ...    )
            ...    .select(Association._get_peak(f.col('score'), w).alias('col'))
            ...    .show()
            ... )
            +---+
            |col|
            +---+
            |0.0|
            |0.1|
            |0.0|
            |0.1|
            +---+
            <BLANKLINE>
        """
        peak_value = scores - f.lag(scores, offset=1).over(window)

        return f.when(
            peak_value.isNull(), f.lit(0)
        ).otherwise(peak_value)

    @staticmethod
    def _create_yearly_view(df, groupby_columns: list[str]) -> DataFrame:
        """Create yearly view on Assocations. 

        A sequence of all years generated between (first evidence - 5 year) and (current year).

        Args:
            df (DataFrame): evidence dataframe
            window_columns (list[str]): Column list for windowing data.

        Returns:
            DataFrame:
        """
        last_year = datetime.now().year
        window_spec = Window.partitionBy('targetId', 'diseaseId').orderBy('year').rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

        return (
            df.select(*groupby_columns, 'year')
            .distinct()
            .withColumn(
                'first_evidence_year', f.min('year').over(window_spec)
            )
            .select(*groupby_columns, 'first_evidence_year')
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
            .select(*groupby_columns, f.explode('years').alias('year'))
        )
    
    @staticmethod
    def _back_fill_missing_years(association_df: DataFrame, groupby_columns: list[str]) -> DataFrame:
        """Fill in data for years where no evidence has arrived.
        
        In the association dataset scores are only available for years when evidence is available. 
        For missing years, association and evidence data is propagated from earlier years.
        This will create a complete dataset telling the association score in any given year from the first year with evidence.
        
        Args:
            association_df (DataFrame): association dataframe to be processed.
            groupby_columns (list[str]): list of columns to group data by

        Return:
            DataFrame: where each missing year from the first year with evidence to current year is filled.
        """
        window_spec = Window.partitionBy(groupby_columns).orderBy('year')

        return (
            # Create a table with all possible target/disease/{optional aggregation column}/year
            Association._create_yearly_view(association_df, groupby_columns)
            # Join with the complete association table - this join will leave multiple rows without association score.
            .join(association_df, on=[*groupby_columns, 'year'], how='left')
            # Filling missing association scores from previous non-null years:
            .withColumns(
                {
                    # Propagating non-null association scores from erlier years:
                    'yearlyAssociationScore': f.coalesce(
                        f.last('yearlyAssociationScore', ignorenulls=True).over(
                            window_spec.rowsBetween(Window.unboundedPreceding, 0)
                        ),
                        f.lit(0),
                    ),
                    # Propagating non-null evidence scores from earlier years:
                    'retrospectiveEvidenceScores': f.coalesce(
                        f.last('retrospectiveEvidenceScores', ignorenulls=True).over(
                            window_spec.rowsBetween(Window.unboundedPreceding, 0)
                        ),
                        f.array().cast(t.ArrayType(t.FloatType())),
                    ),
                }
            )
        )
    
    @staticmethod
    def _get_novelty(
        intermediate_dataset: DataFrame, 
        groupby_columns: list[str], 
        novelty_window: int, 
        novelty_shift: int, 
        novelty_scale:int
    ) -> DataFrame:
        """""Calculate novelty values.

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
            # spark.read.parquet('test_association')
            intermediate_dataset
            # Marking peaks:
            .withColumn('peak', Association._get_peak(f.col('yearlyAssociationScore'), window_spec))
            .withColumnRenamed('year', 'associationYear')
            # Drawing around the window:
            .select('*', Association._windowing_around_peak(f.col('associationYear'), novelty_window))
            # Grouping data again:
            .groupBy([*groupby_columns, 'year'])
            .agg(
                f.max(
                    f.col('peak') / (1 + f.exp(novelty_scale * (f.col('year-peakYear') - novelty_shift))),
                ).alias('novelty')
            )
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
        intermediate_dataset = self._back_fill_missing_years(
            self.df.na.fill({"aggregationValue": "NA"}), self.GROUPBY_COLUMNS
        )

        # Calculate novelty based on subsequent years:
        novelty = self._get_novelty(
            intermediate_dataset, 
            self.GROUPBY_COLUMNS,
            novelty_window,
            novelty_shift,
            novelty_scale
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
                f.collect_list(
                    f.struct(
                        'year', 
                        'yearlyAssociationScore', 
                        'novelty', 
                        'yearlyEvidenceScores'
                    )
                ).alias('timeseries'),
            )
            # Adding current novelty, which allows for quick filtering:
            .withColumns(
                {
                    'currentNovelty': f.filter('timeseries', lambda x: x.year == datetime.now().year)[0].novelty,
                    'aggregationValue': f.when(f.col('aggregationValue')!= 'NA', f.col('aggregationValue'))
                } 
            )
        )
