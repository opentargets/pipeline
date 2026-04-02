from __future__ import annotations

import polars as pl

from pts.transformers.utils import load_spark_schema_as_polars


def test_load_spark_schema_as_polars_evidence() -> None:
    schema = load_spark_schema_as_polars('evidence.json')

    assert schema['datasourceId'] == pl.String
    assert schema['score'] == pl.Float64
    assert str(schema['literature']) == 'List(String)'
    assert str(schema['biomarkers']).startswith('Struct(')
