"""Tests for the reactome PySpark module."""

from __future__ import annotations

from pyspark.sql import SparkSession
from pyspark.sql import types as t

from pts.pyspark.reactome import _build_graph_documents, _clean_pathways


class TestCleanPathways:
    SCHEMA = t.StructType([
        t.StructField('_c0', t.StringType()),
        t.StructField('_c1', t.StringType()),
        t.StructField('_c2', t.StringType()),
    ])

    def test_keeps_only_homo_sapiens(self, spark: SparkSession) -> None:
        data = [('R-HSA-1', 'Pathway A', 'Homo sapiens'), ('R-MMU-1', 'Pathway B', 'Mus musculus')]
        df = spark.createDataFrame(data, schema=self.SCHEMA)
        result = _clean_pathways(df)
        assert result.count() == 1
        assert result.first()['id'] == 'R-HSA-1'

    def test_drops_species_column(self, spark: SparkSession) -> None:
        data = [('R-HSA-1', 'Pathway A', 'Homo sapiens')]
        df = spark.createDataFrame(data, schema=self.SCHEMA)
        result = _clean_pathways(df)
        assert set(result.columns) == {'id', 'name'}


class TestBuildGraphDocuments:
    def test_root_node_has_no_ancestors(self, spark: SparkSession) -> None:
        vertices = spark.createDataFrame([('R-1', 'Root')], ['id', 'name'])
        edges = spark.createDataFrame(
            [],
            t.StructType([
                t.StructField('src', t.StringType()),
                t.StructField('dst', t.StringType()),
            ]),
        )
        result = _build_graph_documents(spark, vertices, edges)
        row = result.filter(result.id == 'R-1').first()
        assert row['ancestors'] == []
        assert row['children'] == []

    def test_child_has_parent_as_ancestor(self, spark: SparkSession) -> None:
        vertices = spark.createDataFrame([('R-1', 'Root'), ('R-2', 'Child')], ['id', 'name'])
        edges = spark.createDataFrame([('R-1', 'R-2')], ['src', 'dst'])
        result = _build_graph_documents(spark, vertices, edges)
        child = result.filter(result.id == 'R-2').first()
        root = result.filter(result.id == 'R-1').first()
        assert 'R-1' in child['ancestors']
        assert 'R-1' in child['parents']
        assert 'R-2' in root['descendants']
        assert any('R-1' in p and 'R-2' in p for p in child['path'])

    def test_cycle_guard_removes_back_edge(self, spark: SparkSession) -> None:
        vertices = spark.createDataFrame([('R-1', 'A'), ('R-2', 'B')], ['id', 'name'])
        # R-1 -> R-2 -> R-1 is a cycle
        edges = spark.createDataFrame([('R-1', 'R-2'), ('R-2', 'R-1')], ['src', 'dst'])
        # Should not raise, should complete successfully
        result = _build_graph_documents(spark, vertices, edges)
        assert result.count() == 2
