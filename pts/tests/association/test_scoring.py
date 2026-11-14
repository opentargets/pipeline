"""Tests for the association by-datasource aggregation."""

from __future__ import annotations

from math import isclose

import polars as pl

from pts.transformers.association.harmonic import MAX_HARMONIC_SUM
from pts.transformers.association.scoring import aggregate_by_datasource


def _evidence(rows: list[tuple]) -> pl.LazyFrame:
    return pl.LazyFrame(
        rows,
        schema={
            'targetId': pl.String,
            'diseaseId': pl.String,
            'datasourceId': pl.String,
            'datatypeId': pl.String,
            'score': pl.Float64,
            'year': pl.Int32,
        },
        orient='row',
    )


def _run(rows: list[tuple]) -> pl.DataFrame:
    return aggregate_by_datasource(_evidence(rows)).sort('targetId', 'diseaseId', 'aggregationValue', 'year').collect()


def test_by_datasource_score_spark_case() -> None:
    out = _run([('t1', 'd1', 'ds2', 'dt', s, 1990) for s in (0.5, 0.5, 0.2)])
    assert out.columns == [
        'targetId',
        'diseaseId',
        'aggregationType',
        'aggregationValue',
        'datatypeId',
        'year',
        'associationScore',
        'yearlyEvidenceCount',
    ]
    assert out.schema['year'] == pl.Int32
    assert out.schema['yearlyEvidenceCount'] == pl.Int32
    row = out.row(0, named=True)
    assert row['aggregationType'] == 'datasourceId'
    assert row['aggregationValue'] == 'ds2'
    assert row['yearlyEvidenceCount'] == 3
    assert isclose(row['associationScore'], 0.6472222225533591 / MAX_HARMONIC_SUM, abs_tol=1e-9)


def test_by_datasource_pools_scores_cumulatively_over_years() -> None:
    out = _run([('t', 'd', 'ds', 'dt', 0.5, 1990), ('t', 'd', 'ds', 'dt', 0.9, 1992)])
    assert out['year'].to_list() == [1990, 1992]
    assert out['yearlyEvidenceCount'].to_list() == [1, 1]
    expected = [0.5 / MAX_HARMONIC_SUM, (0.9 + 0.5 / 4) / MAX_HARMONIC_SUM]
    for got, want in zip(out['associationScore'].to_list(), expected, strict=True):
        assert isclose(got, want, abs_tol=1e-12)


def test_by_datasource_same_year_rows_pool_together() -> None:
    out = _run([('t', 'd', 'ds', 'dt', 0.5, 1990), ('t', 'd', 'ds', 'dt', 0.9, 1990)])
    assert out.height == 1
    assert isclose(out['associationScore'][0], (0.9 + 0.5 / 4) / MAX_HARMONIC_SUM, abs_tol=1e-12)


def test_by_datasource_keys_do_not_leak() -> None:
    out = _run([('t', 'd', 'a', 'dt', 0.5, 1990), ('t', 'd', 'b', 'dt', 0.9, 1991)])
    assert out['aggregationValue'].to_list() == ['a', 'b']
    assert isclose(out['associationScore'][1], 0.9 / MAX_HARMONIC_SUM, abs_tol=1e-12)


def test_by_datasource_cap_survives_carrying_across_years() -> None:
    rows = [('t', 'd', 'ds', 'dt', 1.0, 1990 + i % 3) for i in range(120)]
    out = _run(rows)
    assert out['yearlyEvidenceCount'].to_list() == [40, 40, 40]
    expected = sum(1 / i**2 for i in range(1, 51)) / MAX_HARMONIC_SUM
    assert isclose(out['associationScore'][2], expected, abs_tol=1e-12)


def test_by_datasource_total_count_equals_evidence_rows() -> None:
    rows = [('t', 'd', 'ds', 'dt', 0.1 * (i % 9 + 1), 2000 + i % 4) for i in range(37)]
    out = _run(rows)
    assert out['yearlyEvidenceCount'].sum() == 37
