"""Tests for vectors step."""

import pytest
from pyspark.ml.linalg import Vectors as MLVectors
from pyspark.sql import Row


class TestCategoryAssignment:
    """Test _compute_vectors category logic."""

    def test_target_category(self, spark):
        from pts.pyspark.vectors import _compute_vectors

        df = spark.createDataFrame(
            [Row(word='ENSG00000139618', vector=MLVectors.dense([1.0, 0.0, 0.0]))],
        )
        result = _compute_vectors(df)
        row = result.collect()[0]
        assert row['category'] == 'target'

    def test_drug_category(self, spark):
        from pts.pyspark.vectors import _compute_vectors

        df = spark.createDataFrame(
            [Row(word='CHEMBL25', vector=MLVectors.dense([0.0, 1.0, 0.0]))],
        )
        result = _compute_vectors(df)
        row = result.collect()[0]
        assert row['category'] == 'drug'

    def test_disease_category(self, spark):
        from pts.pyspark.vectors import _compute_vectors

        df = spark.createDataFrame(
            [Row(word='EFO_0000311', vector=MLVectors.dense([0.0, 0.0, 1.0]))],
        )
        result = _compute_vectors(df)
        row = result.collect()[0]
        assert row['category'] == 'disease'


class TestNormComputation:
    """Test L2 norm calculation."""

    def test_unit_vector_norm(self, spark):
        from pts.pyspark.vectors import _compute_vectors

        df = spark.createDataFrame(
            [Row(word='ENSG001', vector=MLVectors.dense([1.0, 0.0, 0.0]))],
        )
        result = _compute_vectors(df)
        row = result.collect()[0]
        assert row['norm'] == pytest.approx(1.0)

    def test_known_norm(self, spark):
        from pts.pyspark.vectors import _compute_vectors

        df = spark.createDataFrame(
            [Row(word='ENSG001', vector=MLVectors.dense([3.0, 4.0]))],
        )
        result = _compute_vectors(df)
        row = result.collect()[0]
        assert row['norm'] == pytest.approx(5.0)


class TestOutputSchema:
    """Test output column structure."""

    def test_output_columns(self, spark):
        from pts.pyspark.vectors import _compute_vectors

        df = spark.createDataFrame(
            [Row(word='ENSG001', vector=MLVectors.dense([1.0, 2.0]))],
        )
        result = _compute_vectors(df)
        assert result.columns == ['category', 'word', 'norm', 'vector']

    def test_vector_is_array(self, spark):
        from pts.pyspark.vectors import _compute_vectors

        df = spark.createDataFrame(
            [Row(word='ENSG001', vector=MLVectors.dense([1.0, 2.0]))],
        )
        result = _compute_vectors(df)
        row = result.collect()[0]
        assert isinstance(row['vector'], list)
        assert len(row['vector']) == 2
