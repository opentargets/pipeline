"""Tests for timeseries evidence logic."""
from __future__ import annotations

from datetime import datetime
from sre_compile import isstring
from pts.pyspark.timeseries_utils.evidence import Evidence

import pytest
from pyspark.sql import types as t, SparkSession, functions as f


class TestEvidence:
    """Testing suite for the Evidence dataset."""
    DATASET = [
        ('t1', 'd1', 1991, 0.5, 'ds1', 'dt1', 1),
        ('t1', 'd1', 1991, 0.8, 'ds1', 'dt1', 2),
        ('t1', 'd1', 2010, 0.5, 'ds1', 'dt1', 3),
        ('t1', 'd1', 1990, 0.5, 'ds2', 'dt1', 4),
        ('t1', 'd1', 1994, 0.5, 'ds3', 'dt1', 5),
        ('t1', 'd2', 1989, 0.5, 'ds3', 'dt1', 6),
    ]

    DATASOURCE_WEIGHTS = [
        ('ds1', 1.0),
        ('ds2', 0.5),
        ('ds3', 0.2),
    ]

    @pytest.fixture(autouse=True)
    def _setup(self: TestEvidence, spark: SparkSession) -> None:
        """Setting up input datasets."""
        self.evidence = Evidence(
            spark.createDataFrame(
                self.DATASET, 
                'targetId STRING, diseaseId STRING, year INTEGER, score FLOAT, datasourceId STRING, datatypeId STRING, id INTEGER'
            )
        )

        # Test return type:
        assert isinstance(self.evidence, Evidence)

        self.datasource_weights = spark.createDataFrame(
            self.DATASOURCE_WEIGHTS, 'datasourceId STRING, weight FLOAT'
        )

    def test_lenght(self: TestEvidence) -> None:
        """Testing if the creation of the Evidence object leads to data loss."""
        assert self.evidence.df.count() == len(self.DATASET)

    def test_weighting_type(self: TestEvidence) -> None:
        """Testing if applying weight results in evidence object."""
        assert isinstance(
            self.evidence.apply_datasource_weight(self.datasource_weights), Evidence
        )

    def test_weighting_columns(self: TestEvidence) -> None:
        """Testing if applyting weights results in chaning list of columns."""
        assert set(self.evidence.df.columns) == set(self.evidence.apply_datasource_weight(self.datasource_weights).df.columns)

    def test_weighting_no_explosion(self: TestEvidence) -> None:
        """Testing if the resulting scores are correct."""
        assert self.evidence.apply_datasource_weight(self.datasource_weights).df.count() == self.evidence.df.count()

