"""Tests for the association novelty computation."""

from __future__ import annotations

from math import exp, isclose

import polars as pl
import pytest

from pts.schemas.association import association_schema
from pts.transformers.association.novelty import compute_novelty

CURRENT_YEAR = 2026


def _association(rows: list[tuple], dtype: type[pl.DataType] = pl.Int32) -> pl.LazyFrame:
    return pl.LazyFrame(
        rows,
        schema={
            'targetId': pl.String,
            'diseaseId': pl.String,
            'aggregationType': pl.String,
            'aggregationValue': pl.String,
            'year': pl.Int32,
            'associationScore': pl.Float64,
            'yearlyEvidenceCount': dtype,
        },
        orient='row',
    )


def _run(
    rows: list[tuple], *, window: int = 10, scale: float = 2, shift: float = 2, dtype: type[pl.DataType] = pl.Int32
) -> pl.DataFrame:
    return compute_novelty(
        _association(rows, dtype),
        current_year=CURRENT_YEAR,
        window=window,
        scale=scale,
        shift=shift,
        evidence_count_dtype=dtype,
    ).collect()


def _row(df: pl.DataFrame, target: str, value: str | None = None) -> dict:
    return df.filter((pl.col('targetId') == target) & (pl.col('aggregationValue').eq_missing(value))).row(0, named=True)


def test_logistic_decay_values_after_a_single_peak():
    out = _run([('t1', 'd1', 'datasourceId', 'ds1', 2020, 0.99, 1)], window=10, scale=0.5, shift=1.0)
    novelty = {e['year']: e['novelty'] for e in _row(out, 't1', 'ds1')['timeseries']}
    expected = {2020: 0.616234737889836, 2021: 0.495, 2022: 0.373765262110164, 2023: 0.26625200715629516}
    for year, value in expected.items():
        assert isclose(novelty[year], value, rel_tol=1e-12)
    assert isclose(novelty[2019], 0.0)


def test_novelty_is_zero_outside_the_window():
    out = _run([('t1', 'd1', 'datasourceId', 'ds1', 2010, 0.5, 1)])
    novelty = {e['year']: e['novelty'] for e in _row(out, 't1', 'ds1')['timeseries']}
    assert isclose(novelty[2010], 0.5 / (1 + exp(-4)))
    assert novelty[2020] > 0
    assert novelty[2021] == 0.0
    assert _row(out, 't1', 'ds1')['currentNovelty'] == 0.0


def test_two_peaks_take_the_larger_contribution():
    out = _run([
        ('t2', 'd2', 'datasourceId', 'ds1', 2010, 0.4, 1),
        ('t2', 'd2', 'datasourceId', 'ds1', 2014, 0.7, 1),
    ])
    novelty = {e['year']: e['novelty'] for e in _row(out, 't2', 'ds1')['timeseries']}
    first = 0.4 / (1 + exp(2 * (4 - 2)))
    second = 0.3 / (1 + exp(2 * (0 - 2)))
    assert isclose(novelty[2014], max(first, second))
    assert isclose(novelty[2016], max(0.4 / (1 + exp(2 * (6 - 2))), 0.3 / (1 + exp(0))))


def test_peak_only_positive_increases_within_window():
    out = _run([
        ('t1', 'd1', 'datasourceId', 'ds1', 2010, 0.5, 1),
        ('t1', 'd1', 'datasourceId', 'ds1', 2012, 0.5, 1),
        ('t1', 'd1', 'datasourceId', 'ds1', 2013, 0.4, 1),
    ])
    novelty = {e['year']: e['novelty'] for e in _row(out, 't1', 'ds1')['timeseries']}
    assert isclose(novelty[2012], 0.5 / (1 + exp(0)))
    assert isclose(novelty[2013], 0.5 / (1 + exp(2 * (3 - 2))))


def test_backfill_starts_five_years_before_pair_minimum_across_aggregation_values():
    out = _run([
        ('t1', 'd1', 'datasourceId', 'ds1', 2010, 0.5, 1),
        ('t1', 'd1', 'datasourceId', 'ds2', 2015, 0.3, 1),
    ])
    years = [e['year'] for e in _row(out, 't1', 'ds2')['timeseries']]
    assert min(y for y in years if y is not None) == 2005
    scores = {e['year']: e['associationScore'] for e in _row(out, 't1', 'ds2')['timeseries']}
    assert scores[2014] == 0.0
    assert scores[2015] == 0.3
    assert scores[2020] == 0.3


def test_backfill_ends_at_current_year_plus_one_and_drops_later_years():
    out = _run([
        ('t1', 'd1', 'datasourceId', 'ds1', 2010, 0.5, 1),
        ('t1', 'd1', 'datasourceId', 'ds1', 2027, 0.8, 2),
        ('t1', 'd1', 'datasourceId', 'ds1', 2030, 0.9, 4),
    ])
    row = _row(out, 't1', 'ds1')
    assert len(row['timeseries']) == CURRENT_YEAR + 1 - 2005 + 1
    assert row['evidenceCount'] == 3
    assert row['associationScore'] == 0.8
    assert [e['year'] for e in row['timeseries']][-1] is None


def test_backfilled_years_keep_a_null_count():
    out = _run([('t1', 'd1', 'datasourceId', 'ds1', 2010, 0.5, 3)])
    counts = {e['year']: e['yearlyEvidenceCount'] for e in _row(out, 't1', 'ds1')['timeseries']}
    assert counts[2010] == 3
    assert counts[2011] is None
    assert counts[2005] is None


def test_current_novelty_is_novelty_at_current_year():
    out = _run([('t1', 'd1', 'datasourceId', 'ds1', 2025, 0.5, 1)])
    row = _row(out, 't1', 'ds1')
    assert isclose(row['currentNovelty'], 0.5 / (1 + exp(2 * (1 - 2))))


def test_association_score_is_the_null_year_entry():
    out = _run([
        ('t1', 'd1', 'datasourceId', 'ds1', 2010, 0.5, 1),
        ('t1', 'd1', 'datasourceId', 'ds1', 2020, 0.7, 1),
    ])
    row = _row(out, 't1', 'ds1')
    assert row['associationScore'] == 0.7
    undated = [e for e in row['timeseries'] if e['year'] is None]
    assert len(undated) == 1
    assert undated[0]['novelty'] is None


def test_na_aggregation_value_round_trips_to_null():
    out = _run([
        ('t3', 'd3', 'overall', None, 2010, 0.6, 1),
        ('t4', 'd3', 'overall', 'None', 2010, 0.6, 1),
    ])
    assert _row(out, 't3', None)['aggregationValue'] is None
    assert _row(out, 't4', 'None')['aggregationValue'] == 'None'


def test_timeseries_sorted_by_year_nulls_last():
    out = _run([
        ('t1', 'd1', 'datasourceId', 'ds1', 2020, 0.7, 1),
        ('t1', 'd1', 'datasourceId', 'ds1', 2010, 0.5, 1),
    ])
    years = [e['year'] for e in _row(out, 't1', 'ds1')['timeseries']]
    assert years[:-1] == sorted(years[:-1])
    assert years[-1] is None


@pytest.mark.parametrize('dtype', [pl.Int32, pl.Int64])
def test_output_schema_matches(dtype):
    frame = compute_novelty(
        _association([('t1', 'd1', 'datasourceId', 'ds1', 2010, 0.5, 1)], dtype),
        current_year=CURRENT_YEAR,
        window=10,
        scale=2,
        shift=2,
        evidence_count_dtype=dtype,
    )
    assert frame.collect_schema() == association_schema(dtype)
    assert frame.collect().schema == association_schema(dtype)
