"""Tests for the association by-datasource aggregation."""

from __future__ import annotations

from math import isclose

import polars as pl

from pts.transformers.association.harmonic import MAX_HARMONIC_SUM
from pts.transformers.association.scoring import aggregate_by_datasource, aggregate_by_datatype, aggregate_overall


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


def _weights(rows: list[tuple[str, float, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        [(ds, w, dt, False) for ds, w, dt in rows],
        schema={'datasourceId': pl.String, 'weight': pl.Float64, 'datatypeId': pl.String, 'toPropagate': pl.Boolean},
        orient='row',
    )


def _by_ds(rows: list[tuple]) -> pl.LazyFrame:
    return pl.LazyFrame(
        [(t, d, 'datasourceId', ds, dt, y, s, c) for t, d, ds, dt, y, s, c in rows],
        schema={
            'targetId': pl.String,
            'diseaseId': pl.String,
            'aggregationType': pl.String,
            'aggregationValue': pl.String,
            'datatypeId': pl.String,
            'year': pl.Int32,
            'associationScore': pl.Float64,
            'yearlyEvidenceCount': pl.Int32,
        },
        orient='row',
    )


def _agg(fn, rows: list[tuple], weights: list[tuple[str, float, str]]) -> pl.DataFrame:
    out = fn(_by_ds(rows), _weights(weights))
    return out.sort('targetId', 'diseaseId', 'aggregationValue', 'year').collect()


def test_overall_weights_and_keeps_latest_max_per_datasource() -> None:
    out = _agg(
        aggregate_overall,
        [
            ('t', 'd', 'ds1', 'dt', 1990, 0.4, 1),
            ('t', 'd', 'ds1', 'dt', 1992, 0.6, 1),
            ('t', 'd', 'ds2', 'dt', 1991, 0.8, 1),
        ],
        [('ds1', 1.0, 'dt'), ('ds2', 0.5, 'dt')],
    )
    assert out['year'].to_list() == [1990, 1991, 1992]
    expected = [
        0.4 / MAX_HARMONIC_SUM,
        (0.4 + 0.4 / 4) / MAX_HARMONIC_SUM,
        (0.6 + 0.4 / 4) / MAX_HARMONIC_SUM,
    ]
    for got, want in zip(out['associationScore'].to_list(), expected, strict=True):
        assert isclose(got, want, abs_tol=1e-12)


def test_overall_aggregation_value_is_string_none() -> None:
    out = _agg(aggregate_overall, [('t', 'd', 'ds1', 'dt', 1990, 0.4, 2)], [('ds1', 1.0, 'dt')])
    assert out.columns == [
        'targetId',
        'diseaseId',
        'aggregationType',
        'aggregationValue',
        'year',
        'associationScore',
        'yearlyEvidenceCount',
    ]
    assert out['aggregationType'].to_list() == ['overall']
    assert out['aggregationValue'].to_list() == ['None']
    assert out.schema['yearlyEvidenceCount'] == pl.Int64


def test_by_datatype_normalises_by_max_harmonic_of_n_ones() -> None:
    out = _agg(
        aggregate_by_datatype,
        [('t', 'd', 'ds1', 'dt', 1990, 0.6, 1), ('t', 'd', 'ds2', 'dt', 1991, 0.4, 1)],
        [('ds1', 0.1, 'dt'), ('ds2', 0.1, 'dt')],
    )
    assert out['aggregationType'].to_list() == ['datatypeId', 'datatypeId']
    assert out['aggregationValue'].to_list() == ['dt', 'dt']
    assert isclose(out['associationScore'][0], 0.6, abs_tol=1e-12)
    assert isclose(out['associationScore'][1], (0.6 + 0.4 / 4) / (1 + 1 / 4), abs_tol=1e-12)


def test_by_datatype_year_score_is_max_over_datasource_prefixes() -> None:
    out = _agg(
        aggregate_by_datatype,
        [('t', 'd', 'dsA', 'dt', 2000, 0.9, 1), ('t', 'd', 'dsB', 'dt', 2000, 0.1, 1)],
        [('dsA', 1.0, 'dt'), ('dsB', 1.0, 'dt')],
    )
    assert out.height == 1
    assert isclose(out['associationScore'][0], 0.9, abs_tol=1e-12)
    assert out['yearlyEvidenceCount'][0] == 2


def test_by_datatype_prior_year_state_carries() -> None:
    out = _agg(
        aggregate_by_datatype,
        [('t', 'd', 'dsA', 'dt', 1999, 0.9, 1), ('t', 'd', 'dsB', 'dt', 2000, 0.1, 1)],
        [('dsA', 1.0, 'dt'), ('dsB', 1.0, 'dt')],
    )
    assert isclose(out['associationScore'][1], (0.9 + 0.1 / 4) / (1 + 1 / 4), abs_tol=1e-12)


def test_datasource_missing_from_weights_is_dropped() -> None:
    out = _agg(
        aggregate_overall,
        [('t', 'd', 'ds1', 'dt', 1990, 0.4, 1), ('t', 'd', 'ghost', 'dt', 1990, 0.9, 5)],
        [('ds1', 1.0, 'dt')],
    )
    assert out.height == 1
    assert out['yearlyEvidenceCount'][0] == 1
    assert isclose(out['associationScore'][0], 0.4 / MAX_HARMONIC_SUM, abs_tol=1e-12)


def test_yearly_evidence_count_summed_per_year() -> None:
    out = _agg(
        aggregate_overall,
        [
            ('t', 'd', 'ds1', 'dt', 1990, 0.4, 3),
            ('t', 'd', 'ds2', 'dt', 1990, 0.2, 4),
            ('t', 'd', 'ds1', 'dt', 1991, 0.5, 1),
        ],
        [('ds1', 1.0, 'dt'), ('ds2', 1.0, 'dt')],
    )
    assert out['yearlyEvidenceCount'].to_list() == [7, 1]


def test_groups_are_independent() -> None:
    out = _agg(
        aggregate_by_datatype,
        [
            ('t1', 'd', 'ds1', 'dt1', 1990, 0.5, 1),
            ('t1', 'd', 'ds2', 'dt2', 1990, 0.2, 1),
            ('t2', 'd', 'ds1', 'dt1', 1990, 0.7, 1),
        ],
        [('ds1', 1.0, 'dt1'), ('ds2', 1.0, 'dt2')],
    )
    assert out.select('targetId', 'aggregationValue', 'associationScore').rows() == [
        ('t1', 'dt1', 0.5),
        ('t1', 'dt2', 0.2),
        ('t2', 'dt1', 0.7),
    ]
