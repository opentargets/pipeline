"""Definition of Evidence class."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Self

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql.window import Window

from pts.pyspark.timeseries_utils.association import Association
from pts.pyspark.timeseries_utils.dataset import Dataset


@dataclass
class Evidence(Dataset):
    """Methods applied on evidence data."""

    # This is the default set of columns we group evidence by, used for association calculation:
    GROUPBY_COLUMNS: list[str] = field(
        default_factory=lambda: ['diseaseId', 'targetId', 'aggregationType', 'aggregationValue']
    )

    # The maximum harmonic sum is hard coded:
    MAX_SCORE: float = 1.64493

    # These are the columns we are working on:
    MANDATORY_COLUMNS: ClassVar[list[str]] = [
        'targetId',
        'diseaseId',
        'score',
        'datasourceId',
        'datatypeId',
        'id',
        'year',
    ]

    # List of aggregation types:
    SUPPORTED_AGGREGATION_TYPES: list[str] = field(default_factory=lambda: ['datasourceId', 'datatypeId', 'overall'])

    @staticmethod
    def _evidence_date_to_year(evidence_date: Column) -> Column:
        """Convert evidence date to year when it is not available, use the last year +1.

        Args:
            evidence_date (Column): spark column with the evidence date

        Returns:
            Column: year, which should never be null.
        """
        return f.when(evidence_date.isNotNull(), f.split(evidence_date, '-')[0].cast(t.IntegerType()))

    @classmethod
    def from_raw_evidence(cls: type[Self], raw_evidence_input: DataFrame) -> Evidence:
        """Convert raw OT evidence to Evidence object.

        Args:
            raw_evidence_input (DataFrame): spark dataframe with the raw evidence.

        Returns:
            Evidence: formatted evidence data with relevant columns.
        """
        # Validation:
        raw_input_columns = [col for col in cls.MANDATORY_COLUMNS if col != 'year']
        cls.validate_columns(raw_evidence_input, [*raw_input_columns, 'evidenceDate'])

        # Covert and return evidence:
        return Evidence(
            raw_evidence_input.select(
                *raw_input_columns, cls._evidence_date_to_year(f.col('evidenceDate')).alias('year')
            )
        )

    def expand_disease(self: Evidence, disease_index: DataFrame) -> Evidence:
        """Calculate indirect evidence based on the provided disease ontology.

        Args:
            disease_index (DataFrame): Open Targets disease index.

        Returns:
            Evidence: evidence exploded to indirect evidence.
        """
        # Exploding disease ancestors:
        processed_disease = disease_index.select(
            f.col('id').alias('diseaseId'),
            f.explode(f.array_union(f.array(f.col('id')), f.col('ancestors'))).alias('specificDiseaseId'),
        )

        # Exploding evidence for all ancestors:
        return Evidence(
            self.df.join(processed_disease, on='diseaseId', how='inner')
            .drop('diseaseId')
            .withColumnRenamed('specificDiseaseId', 'diseaseId')
        )

    @staticmethod
    def _get_top_high_scores(collected_scores: Column, max_array_size: int = 50) -> Column:
        """Get the TOP n scores in an ordered array.

        Args:
            collected_scores (Column): array column with scores.
            max_array_size (int): the maximum size of the returned array.

        Returns:
            Column: ordered array of the top scores, size capped.

        Examples:
            >>> df = (
            ...     spark.createDataFrame([([0.1, 0.5, 0.1, 0.1, 0.9, 0.85],)], ['value'])
            ...     .select(f.explode(Evidence._get_top_high_scores(f.col("value"), 3).alias('returned')))
            ...     .persist()
            ... )
            >>> assert df.first() == t.Row(scores=0.9)
            >>> assert df.count() == 3
        """
        return f.slice(f.reverse(f.array_sort(collected_scores)), 1, max_array_size).alias('scores')

    @staticmethod
    def _get_score_indices(sorted_filtered_scores: Column) -> Column:
        """Get index array equal in size to the score array.

        Args:
            sorted_filtered_scores (Column): Top, ordered scores.

        Returns:
            Column: array of a sequence of intergers from 1, size as the corresponding scores array

        Examples:
            >>> df = (
            ...     spark.createDataFrame([([0.1, 0.5, 0.1, 0.1, 0.9, 0.85],)], ['value'])
            ...     .select(f.explode(Evidence._get_top_high_scores(f.col("value"), 3).alias('returned')))
            ...     .persist()
            ... )
            >>> assert df.first() == t.Row(scores=0.9)
            >>> assert df.count() == 3
        """
        return f.sequence(f.lit(1), f.size(sorted_filtered_scores)).alias('indices')

    @staticmethod
    def _get_harmonic_sum(collected_scores: Column, max_value: float) -> Column:
        """Calculating harmonic sum on an array of score values.

        Process:
            1. Sorting collected scores
            2. Get top 50 scores
            3. get weighted indices
            4. compute harmonic sum of scores
            5. Normalise harmonic sum

        Args:
            collected_scores (Column): list of floats in a spark columns
            max_value (float): maximum value of harmonic sum.

        Returns:
            Column: float normalised harmonic sum of the input array

        Examples:
            >>> (
            ...     spark.createDataFrame([([0.1, 0.5, 0.1, 0.1, 0.9, 0.85],)], ['evidenceScores'])
            ...     .select(
            ...         Evidence._get_harmonic_sum(f.col("evidenceScores"),
            ...         Evidence.MAX_SCORE).alias('associationScore'))
            ...     .show()
            ... )
            +------------------+
            |  associationScore|
            +------------------+
            |0.7180143430622176|
            +------------------+
            <BLANKLINE>
        """
        # For the harmonic sum calculation we only use the highest 50 scores in an ordered fashion:
        sorted_filtered_scores = Evidence._get_top_high_scores(collected_scores)

        # For each of the score values, we generate the corresponding index:
        indices = Evidence._get_score_indices(sorted_filtered_scores)

        weighted_scores = f.transform(
            f.arrays_zip(sorted_filtered_scores, indices), lambda pair: pair.scores / f.pow(pair.indices, f.lit(2))
        )

        return f.aggregate(weighted_scores, f.lit(0.0), lambda acc, x: acc + x) / max_value  # noqa: FURB118  # PySpark requires a lambda here

    def apply_datasource_weight(self: Evidence, datasource_weights: DataFrame) -> Evidence:
        """Apply weight on scores based on a provided weight for each datasourceId.

        Args:
            datasource_weights (DataFrame): Dataframe with two columns: datasourceId and weight.

        Returns:
            Evidence: evidence dataset with applied weight
        """
        return Evidence(
            self.df.join(datasource_weights, on='datasourceId', how='inner')
            .withColumn('score', f.col('score') * f.col('weight'))
            .drop('weight')
        )

    def aggregate_evidence(
        self: Evidence, aggregation_type: str = 'overall', last_year_only: bool = False
    ) -> Association:
        """Compute association scores from the evidence scores.

        If aggregate column is not provided overall assocation score is calculated.

        Args:
            aggregation_type (str): name of the column to association by
            last_year_only (bool): all years will be collapsed into a single year

        Returns:
            Association: Calculated association dataset

        Raises:
            ValueError: provided aggregation type is not supported (accepted aggregation: )
        """
        # Make sure the right aggregation type is requested:
        if aggregation_type not in self.SUPPORTED_AGGREGATION_TYPES:
            allowed = ', '.join(self.SUPPORTED_AGGREGATION_TYPES)
            raise ValueError(f'unsupported aggregation_type: {aggregation_type!r}. Accepted aggregations: {allowed}')

        # For overall association score calculation a fake column needs to be added:
        if aggregation_type == 'overall':
            self._df = self.df.withColumn('overall', f.lit(None).cast(t.StringType()))

        # Dropping historic data if no timeseries are interesting:
        if last_year_only:
            self._df = self.df.withColumn('year', f.lit(datetime.now().year))

        # Unifying data schema:
        self._df = self.df.withColumn('aggregationType', f.lit(aggregation_type)).withColumnRenamed(
            aggregation_type, 'aggregationValue'
        )

        # Partitioning: all evidence accumulated for each disease-target-datasource-year triplet until the given year
        window_for_collapsing_scores = (
            Window.partitionBy(*self.GROUPBY_COLUMNS).orderBy('year').rangeBetween(Window.unboundedPreceding, 0)
        )

        # return Association(
        association = (
            self.df
            # Grouping evidence by the relevant columns:
            .groupBy(*self.GROUPBY_COLUMNS, 'year')
            # Collect scores into an array:
            .agg(f.collect_list('score').alias('yearlyEvidenceScores'))
            # Collate scores for previous years:
            .withColumn(
                'retrospectiveEvidenceScores',
                f.flatten(f.collect_list('yearlyEvidenceScores').over(window_for_collapsing_scores)),
            )
            .withColumn(
                'yearlyAssociationScore',
                self._get_harmonic_sum(f.col('retrospectiveEvidenceScores'), self.MAX_SCORE),
            )
            # Dropping the combined evidence column, as we no longer need it:
            .drop('retrospectiveEvidenceScores')
            .orderBy(f.col('aggregationValue'), f.col('year'))
            # Reconverting Nulls in the aggregation column:
            .repartition(200, f.col('year'))
        )

        # Dropping historic data if no timeseries are interesting:
        if last_year_only:
            association = association.filter(f.col('year') == datetime.now().year)

        return Association(association)
