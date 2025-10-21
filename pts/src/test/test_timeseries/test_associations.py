"""Tests for timeseries association processing."""

from __future__ import annotations

from datetime import datetime

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.window import Window

from pts.pyspark.timeseries_utils.association import Association


class TestAssociation:
    """Testing suite for the Association dataset."""

    # Association dataset:
    DATASET = [
        ('d1', 't1', 2012, [0.2, 0.9], 0.7, 'agg', 'agg_1'),
        ('d1', 't1', 2022, [0.5], 0.8, 'agg', 'agg_1'),
        ('d1', 't2', 1980, [0.6, 0.6, 0.1], 0.5, 'agg', 'agg_1'),
        ('d1', 't2', 2012, [0.1], 0.51, 'agg', 'agg_1'),
    ]

    @pytest.fixture(autouse=True)
    def _setup(self: TestAssociation, spark: SparkSession) -> None:
        self.association = Association(
            spark.createDataFrame(
                self.DATASET,
                'diseaseId STRING, '
                'targetId STRING, '
                'year INTEGER, '
                'yearlyEvidenceScores ARRAY<FLOAT>, '
                'yearlyAssociationScore FLOAT, '
                'aggregationType STRING, '
                'aggregationValue STRING',
            )
        )

    def test_association_type(self: TestAssociation) -> None:
        """Testing if the dataset has the right type."""
        assert isinstance(self.association, Association)

    def test_association_columns_check(self: TestAssociation) -> None:
        """Testing if a missing mandatory column leads to expected error."""
        with pytest.raises(ValueError) as exception:
            # Dropping mandatory column from data:
            Association(self.association.df.drop('diseaseId'))
        assert str(exception.value) == 'Required column: diseaseId is missing from dataset!'

    @pytest.mark.parametrize(
        ('target_id', 'disease_id', 'first_year'),
        [
            ('t1', 'd1', 2007),
            ('t2', 'd1', 1975),
        ],
    )
    def test_date_explosion(self: TestAssociation, target_id, disease_id, first_year) -> None:
        this_year = datetime.now().year
        # Exploding years:
        exploded_filtered_df = (
            # Explode evidence for years:
            Association._create_yearly_view(self.association.df)
            .filter((f.col('diseaseId') == disease_id) & (f.col('targetId') == target_id))
            .orderBy(f.col('year').asc())
        )

        # Assert df is not empty:
        assert exploded_filtered_df.first() is not None

        # Assert the first year:
        assert exploded_filtered_df.first().year == first_year

        # Assert the number of years:
        assert exploded_filtered_df.count() == this_year - first_year + 1

    def test_data_propagation_to_missing_years__type(self: TestAssociation) -> None:
        """Test if the data propagation after explosion is good."""
        propagated_data = Association._back_fill_missing_years(self.association.df)

        assert isinstance(propagated_data, type(self.association.df))

    def test_data_propagation_to_missing_years__no_missing_association(self: TestAssociation) -> None:
        """Test if the data propagation after explosion is good."""
        propagated_data = Association._back_fill_missing_years(self.association.df)

        assert propagated_data.filter(f.col('yearlyAssociationScore').isNull()).count() == 0
        assert propagated_data.filter(f.col('yearlyEvidenceScores').isNull()).count() > 0

    def test_data_propagation_to_missing_years__association_score_always_grow(self: TestAssociation) -> None:
        """Test if yearlyAssociation score never decreases in subsequent years.  Within the same association window."""
        # Create window:
        association_window = Window.partitionBy(Association.GROUPBY_COLUMNS).orderBy('year')

        propagated_data = Association._back_fill_missing_years(self.association.df).withColumn(
            'scoreDiff',
            f.col('yearlyAssociationScore') - f.lag(f.col('yearlyAssociationScore'), offset=1).over(association_window),
        )

        assert propagated_data.filter(f.col('scoreDiff') < 0).count() == 0
