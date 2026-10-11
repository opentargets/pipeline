"""Tests for literature_vector step."""

import random

import numpy as np
import pytest
from pyspark.ml.linalg import Vectors as MLVectors
from pyspark.sql import Row


class TestCategoryAssignment:
    """Test _compute_vectors category logic."""

    def test_target_category(self, spark):
        from pts.pyspark.literature_vector import _compute_vectors

        df = spark.createDataFrame(
            [Row(word='ENSG00000139618', vector=MLVectors.dense([1.0, 0.0, 0.0]))],
        )
        result = _compute_vectors(df)
        row = result.collect()[0]
        assert row['category'] == 'target'

    def test_drug_category(self, spark):
        from pts.pyspark.literature_vector import _compute_vectors

        df = spark.createDataFrame(
            [Row(word='CHEMBL25', vector=MLVectors.dense([0.0, 1.0, 0.0]))],
        )
        result = _compute_vectors(df)
        row = result.collect()[0]
        assert row['category'] == 'drug'

    def test_disease_category(self, spark):
        from pts.pyspark.literature_vector import _compute_vectors

        df = spark.createDataFrame(
            [Row(word='EFO_0000311', vector=MLVectors.dense([0.0, 0.0, 1.0]))],
        )
        result = _compute_vectors(df)
        row = result.collect()[0]
        assert row['category'] == 'disease'


class TestNormComputation:
    """Test L2 norm calculation."""

    def test_unit_vector_norm(self, spark):
        from pts.pyspark.literature_vector import _compute_vectors

        df = spark.createDataFrame(
            [Row(word='ENSG001', vector=MLVectors.dense([1.0, 0.0, 0.0]))],
        )
        result = _compute_vectors(df)
        row = result.collect()[0]
        assert row['norm'] == pytest.approx(1.0)

    def test_known_norm(self, spark):
        from pts.pyspark.literature_vector import _compute_vectors

        df = spark.createDataFrame(
            [Row(word='ENSG001', vector=MLVectors.dense([3.0, 4.0]))],
        )
        result = _compute_vectors(df)
        row = result.collect()[0]
        assert row['norm'] == pytest.approx(5.0)

    def test_matches_python_norm(self, spark):
        from pts.pyspark.literature_vector import _compute_vectors

        rng = random.Random(0)
        # Word2Vec keeps float32 weights
        vectors = [[float(np.float32(rng.gauss(0.0, 0.3))) for _ in range(100)] for _ in range(2000)]
        vectors += [[0.0] * 100, [1e-30] * 100, [-2.5], [float(np.float32(1.1))] * 7]
        df = spark.createDataFrame([Row(word=f'w{i}', vector=MLVectors.dense(v)) for i, v in enumerate(vectors)])

        rows = _compute_vectors(df).collect()
        # the former `Vectors.norm(v, 2.0)` UDF
        expected = {f'w{i}': float(MLVectors.norm(MLVectors.dense(v), 2.0)) for i, v in enumerate(vectors)}

        # numpy's dot product sums in a different order, so the last few bits may differ
        for row in rows:
            assert row['norm'] == pytest.approx(expected[row['word']], rel=1e-14, abs=0.0)


class TestOutputSchema:
    """Test output column structure."""

    def test_output_columns(self, spark):
        from pts.pyspark.literature_vector import _compute_vectors

        df = spark.createDataFrame(
            [Row(word='ENSG001', vector=MLVectors.dense([1.0, 2.0]))],
        )
        result = _compute_vectors(df)
        assert result.columns == ['category', 'word', 'norm', 'vector']

    def test_vector_is_array(self, spark):
        from pts.pyspark.literature_vector import _compute_vectors

        df = spark.createDataFrame(
            [Row(word='ENSG001', vector=MLVectors.dense([1.0, 2.0]))],
        )
        result = _compute_vectors(df)
        row = result.collect()[0]
        assert isinstance(row['vector'], list)
        assert len(row['vector']) == 2
