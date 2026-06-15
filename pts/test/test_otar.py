"""Tests for the otar PySpark module."""

from __future__ import annotations

from pyspark.sql import SparkSession

from pts.pyspark.otar import _generate_otar_info


class TestGenerateOtarInfo:
    def test_propagates_to_ancestors(self, spark: SparkSession) -> None:
        disease = spark.createDataFrame(
            [('EFO:0001', ['EFO:0001', 'EFO:ROOT'])],
            'id STRING, ancestors ARRAY<STRING>',
        )
        meta = spark.createDataFrame(
            [('OTAR001', 'Project Alpha', 'Active', 'false')],
            'otar_code STRING, project_name STRING, project_status STRING, integrates_in_PPP STRING',
        )
        lookup = spark.createDataFrame(
            [('OTAR001', 'EFO:0001')],
            'otar_code STRING, efo_disease_id STRING',
        )
        result = _generate_otar_info(disease, meta, lookup)
        efo_ids = {r['efo_id'] for r in result.collect()}
        assert 'EFO:ROOT' in efo_ids  # propagated to ancestor

    def test_includes_reference_url(self, spark: SparkSession) -> None:
        disease = spark.createDataFrame(
            [('EFO:0001', ['EFO:0001'])],
            'id STRING, ancestors ARRAY<STRING>',
        )
        meta = spark.createDataFrame(
            [('OTAR001', 'Project Alpha', 'Active', 'false')],
            'otar_code STRING, project_name STRING, project_status STRING, integrates_in_PPP STRING',
        )
        lookup = spark.createDataFrame(
            [('OTAR001', 'EFO:0001')],
            'otar_code STRING, efo_disease_id STRING',
        )
        result = _generate_otar_info(disease, meta, lookup)
        project = result.collect()[0]['projects'][0]
        assert project['reference'] == 'http://home.opentargets.org/OTAR001'
