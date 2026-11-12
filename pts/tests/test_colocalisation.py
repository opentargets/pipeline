"""Tests for the colocalisation entry point."""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl
import pytest

from pts.pyspark import colocalisation as step
from pts.schemas.colocalisation import colocalisation_schema
from tests.credible_set_fixtures import published_credible_set, published_credible_set_variant


def _write_inputs(tmp_path):
    published_credible_set(
        {'id': 'g', 'studyType': 'gwas', 'chromosome': '1'},
        {'id': 'q', 'studyType': 'eqtl', 'chromosome': '1'},
    ).write_parquet(tmp_path / 'credible_set.parquet')
    published_credible_set_variant(
        {'credibleSetId': 'g', 'variantId': 'v1', 'posteriorProbability': 0.5, 'reportedBeta': 0.2},
        {'credibleSetId': 'g', 'variantId': 'v2', 'posteriorProbability': 0.5, 'reportedBeta': 0.1},
        {'credibleSetId': 'q', 'variantId': 'v1', 'posteriorProbability': 1.0, 'reportedBeta': -0.3},
    ).write_parquet(tmp_path / 'credible_set_variant.parquet')


def test_colocalisation_end_to_end(spark, tmp_path, monkeypatch):
    _write_inputs(tmp_path)
    sessions = []

    def _session(**kwargs):
        sessions.append(kwargs)
        return SimpleNamespace(load_data=lambda path, schema=None: spark.read.schema(schema).parquet(path))

    monkeypatch.setattr(step, 'Session', _session)

    step.colocalisation(
        {
            'credible_set': str(tmp_path / 'credible_set.parquet'),
            'credible_set_variant': str(tmp_path / 'credible_set_variant.parquet'),
        },
        {'colocalisation': str(tmp_path / 'out')},
        {'output_partitions': 1},
        {'spark.executor.cores': '4'},
    )

    assert sessions == [{'app_name': 'colocalisation', 'properties': {'spark.executor.cores': '4'}}]
    written = pl.read_parquet(tmp_path / 'out' / '*.parquet')
    assert written.columns == list(colocalisation_schema)
    [row] = written.to_dicts()
    assert row['leftLocusId'] == 'g'
    assert row['rightLocusId'] == 'q'
    assert row['rightStudyType'] == 'eqtl'
    assert row['chromosome'] == '1'
    assert row['colocalisationMethod'] == 'COLOC_PIP_ECAVIAR'
    assert row['numberColocalisingVariants'] == 1
    assert row['clpp'] == pytest.approx(0.5, rel=1e-12)
    assert row['h4'] == pytest.approx(5e-6 / (5e-6 + 5e-9), rel=1e-9)
    assert row['h3'] == pytest.approx(5e-9 / (5e-6 + 5e-9), rel=1e-9)
    assert row['betaRatioSignAverage'] == -1.0
