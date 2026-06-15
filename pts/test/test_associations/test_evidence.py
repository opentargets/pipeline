"""Tests for timeseries evidence logic."""

from __future__ import annotations

from math import isclose

import pytest
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql import functions as f

from pts.pyspark.associations_utils.association import Association
from pts.pyspark.associations_utils.dataset import Dataset
from pts.pyspark.associations_utils.evidence import Evidence


@pytest.mark.slow
class TestEvidence:
    """Testing suite for the Evidence dataset."""

    DATASET = [
        ('t1', 'd1', 1991, 0.5, 'ds1', 'dt1', 1),
        ('t1', 'd1', 1991, 0.8, 'ds1', 'dt1', 2),
        ('t1', 'd1', 2010, 0.5, 'ds1', 'dt1', 3),
        ('t1', 'd1', 1990, 0.5, 'ds2', 'dt1', 4),
        ('t1', 'd1', 1994, 0.5, 'ds3', 'dt1', 5),
        ('t1', 'd2', 1989, 0.5, 'ds3', 'dt1', 6),
        ('t2', 'd1', 2010, 0.5, 'ds1', 'dt1', 7),
        ('t3', 'd2', 2010, 0.5, 'ds1', 'dt1', 7),
        ('t1', 'd1', 1990, 0.5, 'ds2', 'dt1', 8),
        ('t1', 'd1', 1990, 0.2, 'ds2', 'dt1', 9),
    ]

    DISEASE_DATA = [('d1', ['d3', 'd4']), ('d3', ['d4']), ('d2', ['d4'])]

    # ds1 and ds2 are propagated; ds3 is not
    DATASOURCE_WEIGHT_DATA = [('ds1', True), ('ds2', True), ('ds3', False)]

    @pytest.fixture(autouse=True)
    def _setup(self: TestEvidence, spark: SparkSession) -> None:
        """Setting up input datasets."""
        self.evidence = Evidence(
            spark.createDataFrame(
                self.DATASET,
                'targetId STRING, diseaseId STRING, year INTEGER, score FLOAT,'
                'datasourceId STRING, datatypeId STRING, id INTEGER',
            )
        )

        # Create disease index:
        self.disease_df = spark.createDataFrame(self.DISEASE_DATA, 'id STRING, ancestors ARRAY<STRING>')

        # Create datasource weight lookup:
        self.datasource_weight_df = spark.createDataFrame(
            self.DATASOURCE_WEIGHT_DATA, 'datasourceId STRING, toPropagate BOOLEAN'
        )

        # Test return type:
        assert isinstance(self.evidence, Evidence)

    def test_length(self: TestEvidence) -> None:
        """Testing if the creation of the Evidence object leads to data loss."""
        assert self.evidence.df.count() == len(self.DATASET)

    def test_filter__type(self: TestEvidence) -> None:
        """Testing if filter method yields the right type."""
        assert isinstance(self.evidence.filter(f.col('diseaseId') == 'd1'), Evidence)

    def test_filter__count(self: TestEvidence) -> None:
        """Testing if filter method yields the correct rows."""
        filtered_df = self.evidence.filter(f.col('diseaseId') == 'd1').df

        # We have the right number of rows:
        assert filtered_df.count() == 8

        # We don't have other rows than what we are expecting:
        assert filtered_df.filter(f.col('diseaseId') != 'd1').count() == 0

    def test_disease_expansion__type(self: TestEvidence) -> None:
        """Testing if diease expansion returns the same type."""
        assert isinstance(self.evidence.expand_disease(self.disease_df, self.datasource_weight_df), Evidence)

    def test_disease_expansion__explosion(self: TestEvidence) -> None:
        """Testing if diease expansion indeed explodes evidence."""
        assert (
            self.evidence.expand_disease(self.disease_df, self.datasource_weight_df).df.count()
            > self.evidence.df.count()
        )

    @pytest.mark.parametrize(
        ('target_id', 'disease_id', 'direct_count', 'expected_diseases'),
        [
            ('t2', 'd1', 1, ['d1', 'd3', 'd4']),  # ds1 is propagated -> expands to all ancestors
            ('t3', 'd2', 1, ['d2', 'd4']),  # ds1 is propagated -> expands to all ancestors
        ],
    )
    def test_disease_expansion__explosion_correct(
        self: TestEvidence, target_id: str, disease_id: str, direct_count: int, expected_diseases: list[str]
    ) -> None:
        """Testing if diease expansion yields the expected diseases."""
        exploded_evidence = self.evidence.expand_disease(self.disease_df, self.datasource_weight_df)

        # Make sure there's only one evidence before explosion:
        assert self.evidence.filter(f.col('targetId') == target_id).df.count() == direct_count

        # Make sure the explosion yields the expected diseases:
        exploded_diseases = [
            row['diseaseId'] for row in exploded_evidence.filter(f.col('targetId') == target_id).df.collect()
        ]
        assert set(exploded_diseases) == set(expected_diseases)

    def test_disease_expansion__non_propagated_stays_direct(self: TestEvidence) -> None:
        """Testing that evidence from non-propagated datasources is not expanded to ancestor diseases.

        ds3 is toPropagate=False. The dataset has evidence for t1/d2/ds3 (row 6).
        d2 has ancestor d4, so if propagation were applied it would also appear as t1/d4/ds3.
        After expansion, only d2 should be present for that target/datasource combination.
        """
        exploded_evidence = self.evidence.expand_disease(self.disease_df, self.datasource_weight_df)

        ds3_diseases = {
            row['diseaseId']
            for row in exploded_evidence.df.filter(
                (f.col('targetId') == 't1') & (f.col('datasourceId') == 'ds3')
            ).collect()
        }

        # d1 and d2 are the direct diseases for t1/ds3; d4 (ancestor of both) must not appear
        assert 'd4' not in ds3_diseases
        assert ds3_diseases == {'d1', 'd2'}

    def test_disease_expansion__propagated_does_not_lose_direct(self: TestEvidence) -> None:
        """Testing that the direct disease is still present after propagation for propagated datasources."""
        exploded_evidence = self.evidence.expand_disease(self.disease_df, self.datasource_weight_df)

        # t2/ds1 has direct evidence on d1; after expansion d1 must still be present alongside ancestors
        ds1_diseases_t2 = {
            row['diseaseId']
            for row in exploded_evidence.df.filter(
                (f.col('targetId') == 't2') & (f.col('datasourceId') == 'ds1')
            ).collect()
        }
        assert 'd1' in ds1_diseases_t2

    def test_disease_expansion__unknown_datasource_defaults_to_propagated(
        self: TestEvidence, spark: SparkSession
    ) -> None:
        """Testing that evidence with a datasourceId absent from datasource_weight is not silently dropped.

        After a left join, unknown datasources have toPropagate=null.  The coalesce defaults
        null to True, so the evidence should be fully expanded to ancestor diseases.
        """
        unknown_ds_evidence = Evidence(
            spark.createDataFrame(
                [('t9', 'd1', 2010, 0.5, 'ds_unknown', 'dt1')],
                'targetId STRING, diseaseId STRING, year INTEGER, score FLOAT, datasourceId STRING, datatypeId STRING',
            )
        )

        expanded = unknown_ds_evidence.expand_disease(self.disease_df, self.datasource_weight_df)
        expanded_diseases = {row['diseaseId'] for row in expanded.df.collect()}

        # d1 has ancestors d3, d4 — all three must appear because unknown defaults to propagated=True
        assert expanded_diseases == {'d1', 'd3', 'd4'}

    def test_aggregation__output_type(self: TestEvidence) -> None:
        """Testing if the evidence aggregation yields the right dataset."""
        assert isinstance(self.evidence.aggregate_evidence_by_datasource(), Association)

    def test_aggregation__count(self: TestEvidence) -> None:
        """Testing if the evidence aggregation yields the right dataset."""
        assert self.evidence.aggregate_evidence_by_datasource().df.count() == 7

    def test_aggregation__evidence_count(self: TestEvidence) -> None:
        """Testing if the number of evidece is returned as evidence count."""
        association = self.evidence.aggregate_evidence_by_datasource().df

        assert isinstance(association, DataFrame)
        assert isinstance(association.agg(f.sum('yearlyEvidenceCount').alias('col')).first(), Row)

        assert association.agg(f.sum('yearlyEvidenceCount').alias('col')).first()['col'] == self.evidence.df.count()  # ty:ignore[not-subscriptable]

    def test_score_calculation(self: TestEvidence) -> None:
        """Testing if the normalised harmoinc score is properly calculated.

        Row to test:
        +--------+---------+----------------+----+------------------+-------------------+---------------+
        |targetId|diseaseId|aggregationValue|year|  associationScore|yearlyEvidenceCount|aggregationType|
        +--------+---------+----------------+----+------------------+-------------------+---------------+
        |      t1|       d1|             ds2|1990|0.3958584683801278|                  3|   datasourceId|
        +--------+---------+----------------+----+------------------+-------------------+---------------+

        Runderlying evidence:
        +--------+---------+----+-----+------------+----------+---+
        |targetId|diseaseId|year|score|datasourceId|datatypeId| id|
        +--------+---------+----+-----+------------+----------+---+
        |      t1|       d1|1990|  0.5|         ds2|       dt1|  4|
        |      t1|       d1|1990|  0.5|         ds2|       dt1|  8|
        |      t1|       d1|1990|  0.2|         ds2|       dt1|  9|
        +--------+---------+----+-----+------------+----------+---+
        """
        target_id = 't1'
        disease_id = 'd1'
        year = 1990

        # get scores:
        harmonic_sum = (
            self.evidence.df
            .filter((f.col('targetId') == target_id) & (f.col('diseaseId') == disease_id) & (f.col('year') == year))
            .groupBy(f.lit('x'))
            .agg(Dataset._get_harmonic_sum(f.collect_list('score')).alias('sum'))
            .first()['sum']  # ty:ignore[not-subscriptable]
        )

        normalised_harmonic_sum = (
            self.evidence
            .filter((f.col('targetId') == target_id) & (f.col('diseaseId') == disease_id) & (f.col('year') == year))
            .aggregate_evidence_by_datasource()
            .df.first()['associationScore']  # ty:ignore[not-subscriptable]
        )

        assert isclose(harmonic_sum, 0.6472222225533591)
        assert harmonic_sum / self.evidence.MAX_HARMONIC_SUM == normalised_harmonic_sum
