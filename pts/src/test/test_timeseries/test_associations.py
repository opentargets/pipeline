"""Tests for timeseries association processing."""
from __future__ import annotations

from datetime import datetime
from pts.pyspark.timeseries_utils import Association

import pytest
from pyspark.sql import types as t, SparkSession, functions as f


class TestAssociation:
    """Testing suite for the Association dataset."""

    # Association dataset:
    DATASET = [
        ('d1', 't1', 2012, [0.2, 0.9], [0.2, 0.9], 0.7, 'overall', None),
        ('d1', 't1', 2022, [0.5], [0.2, 0.9, 0.5], 0.8, 'overall', None),
        ('d1', 't2', 1980, [0.6, 0.6, 0.1], [0.6, 0.6, 0.1], 0.5, 'overall', None),
        ('d1', 't2', 2012, [0.1], [0.6, 0.6, 0.1, 0.1], 0.51, 'overall', None),
    ]

    @pytest.fixture(autouse=True)
    def _setup(self: TestAssociation, spark: SparkSession) -> None:
        self.association = Association(
            spark.createDataFrame(
                self.DATASET, 
                'diseaseId STRING, targetId STRING, year INTEGER, yearlyEvidenceScores ARRAY<FLOAT>, retrospectiveEvidenceScores  ARRAY<FLOAT>, yearlyAssociationScore FLOAT, aggregationType STRING, aggregationValue STRING'
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
        assert exception.value.__str__() == "Required column: diseaseId is missing from dataset!"

    @pytest.mark.parametrize(
        "target_id, disease_id, first_year",
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
            Association._create_yearly_view(self.association.df, ['diseaseId', 'targetId'])
            .filter(
                (f.col('diseaseId') == disease_id) &
                (f.col("targetId") == target_id)
            )
            .orderBy(f.col('year').asc())
        )

        # Assert df is not empty:
        assert exploded_filtered_df.first() is not None
        
        # Assert the first year:
        assert exploded_filtered_df.first().year == first_year

        # Assert the number of years:
        assert exploded_filtered_df.count() == this_year - first_year + 1

