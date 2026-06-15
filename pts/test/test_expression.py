"""Tests for the expression PySpark module."""

from __future__ import annotations

from pyspark.sql import SparkSession
from pyspark.sql import types as t

from pts.pyspark.expression import _efo_tissue_mapping, _transform_normal_tissue, _transpose_baseline


class TestTransformNormalTissue:
    SCHEMA = t.StructType([
        t.StructField('Gene', t.StringType()),
        t.StructField('Tissue', t.StringType()),
        t.StructField('Cell_type', t.StringType()),
        t.StructField('Level', t.StringType()),
        t.StructField('Reliability', t.StringType()),
    ])

    def test_filters_na_level(self, spark: SparkSession) -> None:
        data = [
            ('G1', 'Liver', None, 'High', 'Supportive'),
            ('G2', 'Lung', None, 'N/A', 'Uncertain'),
        ]
        df = spark.createDataFrame(data, schema=self.SCHEMA)
        result = _transform_normal_tissue(df)
        assert result.count() == 1
        assert result.first()['Gene'] == 'G1'

    def test_maps_reliability_supportive_to_true(self, spark: SparkSession) -> None:
        data = [('G1', 'Liver', None, 'High', 'Supportive')]
        df = spark.createDataFrame(data, schema=self.SCHEMA)
        result = _transform_normal_tissue(df)
        assert result.first()['ReliabilityMap'] is True

    def test_maps_level_high_to_3(self, spark: SparkSession) -> None:
        data = [('G1', 'Liver', None, 'High', 'Supportive')]
        df = spark.createDataFrame(data, schema=self.SCHEMA)
        result = _transform_normal_tissue(df)
        assert result.first()['LevelMap'] == 3

    def test_maps_reliability_uncertain_to_false(self, spark: SparkSession) -> None:
        data = [('G1', 'Liver', None, 'Low', 'Uncertain')]
        df = spark.createDataFrame(data, schema=self.SCHEMA)
        result = _transform_normal_tissue(df)
        assert result.first()['ReliabilityMap'] is False

    def test_normalises_column_names_with_spaces(self, spark: SparkSession) -> None:
        schema = t.StructType([
            t.StructField('Gene', t.StringType()),
            t.StructField('Tissue', t.StringType()),
            t.StructField('Cell type', t.StringType()),
            t.StructField('Level', t.StringType()),
            t.StructField('Reliability', t.StringType()),
        ])
        data = [('G1', 'Liver', 'Hepatocyte', 'High', 'Supportive')]
        df = spark.createDataFrame(data, schema=schema)
        result = _transform_normal_tissue(df)
        assert 'Cell_type' in result.columns


class TestTransposeBaseline:
    def test_produces_gene_tissue_val_unit_columns(self, spark: SparkSession) -> None:
        schema = t.StructType([
            t.StructField('ID', t.StringType()),
            t.StructField('ENSG001', t.StringType()),
            t.StructField('ENSG002', t.StringType()),
        ])
        data = [('Liver', '1.5', '2.0'), ('Lung', '0.3', '4.1')]
        df = spark.createDataFrame(data, schema=schema)
        result = _transpose_baseline(df, 'TPM')
        assert set(result.columns) == {'Gene', 'Tissue', 'val', 'unit'}

    def test_creates_one_row_per_gene_tissue(self, spark: SparkSession) -> None:
        schema = t.StructType([
            t.StructField('ID', t.StringType()),
            t.StructField('ENSG001', t.StringType()),
        ])
        data = [('Liver', '1.5'), ('Lung', '0.3')]
        df = spark.createDataFrame(data, schema=schema)
        result = _transpose_baseline(df, 'TPM')
        assert result.count() == 2  # 1 gene x 2 tissues


class TestEfoTissueMapping:
    def test_maps_efo_code_to_tissue(self, spark: SparkSession) -> None:
        efomap = spark.createDataFrame(
            [('liver', 'EFO:001', 'Liver', None, None)],
            'tissue_id STRING, efo_code STRING, label STRING, anatomical_systems STRING, organs STRING',
        )
        hierarchy = spark.createDataFrame(
            [('T001', 'liver')],
            '_c0 STRING, _c1 STRING',
        )
        result = _efo_tissue_mapping(efomap, hierarchy)
        row = result.first()
        assert row['efoId'] == 'EFO:001'
        assert row['labelNew'] == 'Liver'

    def test_uses_name_when_efo_code_is_null(self, spark: SparkSession) -> None:
        efomap = spark.createDataFrame(
            [('liver', None, None, None, None)],
            'tissue_id STRING, efo_code STRING, label STRING, anatomical_systems STRING, organs STRING',
        )
        hierarchy = spark.createDataFrame(
            [('T001', 'liver')],
            '_c0 STRING, _c1 STRING',
        )
        result = _efo_tissue_mapping(efomap, hierarchy)
        row = result.first()
        assert row['efoId'] == result.filter(result.efoId.isNotNull()).first()['efoId']
