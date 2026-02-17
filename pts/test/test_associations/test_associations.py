"""Tests for timeseries association processing."""

from __future__ import annotations

from datetime import datetime

import pytest
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f
from pyspark.sql.window import Window

from pts.pyspark.associations_utils.association import Association


@pytest.mark.slow
class TestAssociation:
    """Testing suite for the Association dataset."""

    # Association dataset:
    DATASET = [
        ('d1', 't1', 2012, [0.2, 0.9], 0.7, 'datasourceId', 'ds1', 2),
        ('d1', 't1', 2022, [0.5], 0.8, 'datasourceId', 'ds1', 1),
        ('d1', 't2', 1980, [0.6, 0.6, 0.1], 0.5, 'datasourceId', 'ds1', 3),
        ('d1', 't2', 2012, [0.1], 0.51, 'datasourceId', 'ds2', 1),
    ]

    # Some weights for the datasources:
    DATASOURCE_WEIGHTS = [
        ('ds1', 'dt1', 1.0),
        ('ds2', 'dt1', 0.5),
        ('ds3', 'dt2', 0.2),
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
                'associationScore FLOAT, '
                'aggregationType STRING, '
                'aggregationValue STRING, '
                'yearlyEvidenceCount INTEGER',
            )
        )

        self.datasource_weights = spark.createDataFrame(
            self.DATASOURCE_WEIGHTS, ['datasourceId', 'datatypeId', 'weight']
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
        ('target_id', 'disease_id', 'datasource_id', 'first_year'),
        [
            ('t1', 'd1', 'ds1', 2007),
            ('t2', 'd1', 'ds2', 1975),
        ],
    )
    def test_date_explosion(self: TestAssociation, target_id, disease_id, datasource_id, first_year) -> None:
        """Test if the process that explodes data to continuous yearly records works fine."""
        this_year = datetime.now().year
        # Exploding years:
        exploded_filtered_df = (
            # Explode evidence for years:
            Association._create_yearly_view(self.association.df)
            .filter(
                (f.col('diseaseId') == disease_id)
                & (f.col('targetId') == target_id)
                & (f.col('aggregationValue') == datasource_id)
            )
            .orderBy(f.col('year').asc())
        )

        # Assert df is not empty:
        assert exploded_filtered_df.first() is not None

        # Assert the first year:
        assert exploded_filtered_df.first().year == first_year  # pyright: ignore[reportOptionalMemberAccess]

        # Assert the number of years:
        assert exploded_filtered_df.count() == this_year - first_year + 2

    def test_data_propagation_to_missing_years__type(self: TestAssociation) -> None:
        """Test if the data propagation after explosion is good."""
        propagated_data = Association._back_fill_missing_years(self.association.df)

        assert isinstance(propagated_data, type(self.association.df))

    def test_data_propagation_to_missing_years__no_missing_association(self: TestAssociation) -> None:
        """Test if the data propagation after explosion is good."""
        propagated_data = Association._back_fill_missing_years(self.association.df)

        assert propagated_data.filter(f.col('associationScore').isNull()).count() == 0
        assert propagated_data.filter(f.col('yearlyEvidenceScores').isNull()).count() > 0

    def test_data_propagation_to_missing_years__association_score_always_grow(self: TestAssociation) -> None:
        """Test if yearlyAssociation score never decreases in subsequent years.  Within the same association window."""
        # Create window:
        association_window = Window.partitionBy(Association.GROUPBY_COLUMNS).orderBy('year')

        propagated_data = Association._back_fill_missing_years(self.association.df).withColumn(
            'scoreDiff',
            f.col('associationScore') - f.lag(f.col('associationScore'), offset=1).over(association_window),
        )

        assert propagated_data.filter(f.col('scoreDiff') < 0).count() == 0

    def test_aggregate_overall__return_type(self: TestAssociation) -> None:
        """Test if the overall aggregation returns the right data type."""
        assert isinstance(self.association.aggregate_overall(self.datasource_weights), Association)

    def test_aggregate_by_datatype__return_type(self: TestAssociation) -> None:
        """Test if the aggregation by datatype returns the right data type."""
        assert isinstance(self.association.aggregate_by_datatype(self.datasource_weights), Association)

    def test_aggregate_overall__returned_counte(self: TestAssociation) -> None:
        """Test if the overall aggregation returns the right amount of rows."""
        aggregated_count = self.association.aggregate_overall(self.datasource_weights).df.count()

        assert aggregated_count == self.association.df.select('diseaseId', 'targetId', 'year').distinct().count()

    def test_aggregate_by_datatype__returned_count(self: TestAssociation) -> None:
        """Test if the weighting function returns the right data type."""
        aggregated_count = self.association.aggregate_by_datatype(self.datasource_weights).df.count()

        assert (
            aggregated_count
            == self.association.df.select('diseaseId', 'targetId', 'year', 'aggregationValue').distinct().count()
        )

    def test_timeseries_by_datasource__return_type(self: TestAssociation) -> None:
        """Test if the aggregation by datatype returns the right data type."""
        assert isinstance(self.association.compute_novelty(), DataFrame)

    def test_timeseries_by_datasource__returned_counte(self: TestAssociation) -> None:
        """Test if the overall aggregation returns the right amount of rows."""
        aggregated_count = self.association.compute_novelty().count()

        assert (
            aggregated_count
            == self.association.df.select('diseaseId', 'targetId', 'aggregationValue').distinct().count()
        )
