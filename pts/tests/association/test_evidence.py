"""Tests for association evidence preparation."""

from __future__ import annotations

import polars as pl
import pytest

from pts.transformers.association.evidence import evidence_year, expand_disease, prepare_evidence

NEXT_YEAR = 2027


@pytest.mark.parametrize(
    ('date', 'expected'),
    [
        ('1999-12-31', 1999),
        ('2021-02-11', 2021),
        (None, NEXT_YEAR),
        ('Batman-Robin', NEXT_YEAR),
        ('2021', 2021),
        ('\uff12\uff10\uff12\uff11-01-01', NEXT_YEAR),
        ('20211', NEXT_YEAR),
    ],
)
def test_evidence_year(date: str | None, expected: int) -> None:
    df = pl.DataFrame({'date': [date]}, schema={'date': pl.String}).select(
        evidence_year(pl.col('date'), NEXT_YEAR).alias('year')
    )
    assert df.schema['year'] == pl.Int32
    assert df['year'].to_list() == [expected]


def _raw(rows: list[tuple]) -> pl.LazyFrame:
    return pl.LazyFrame(
        rows,
        schema={
            'targetId': pl.String,
            'diseaseId': pl.String,
            'datasourceId': pl.String,
            'datatypeId': pl.String,
            'score': pl.Float64,
            'evidenceDate': pl.String,
        },
        orient='row',
    )


def test_prepare_evidence_drops_non_positive_scores() -> None:
    raw = _raw([
        ('t1', 'd1', 'ds', 'dt', 0.0, '2020-01-01'),
        ('t1', 'd1', 'ds', 'dt', -1.0, '2020-01-01'),
        ('t1', 'd1', 'ds', 'dt', 0.1, '2020-01-01'),
    ])
    out = prepare_evidence(raw, NEXT_YEAR).collect()
    assert out.columns == ['targetId', 'diseaseId', 'datasourceId', 'datatypeId', 'score', 'year']
    assert out['score'].to_list() == [0.1]
    assert out['year'].to_list() == [2020]
    assert out.schema['year'] == pl.Int32


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


DISEASE = pl.LazyFrame(
    {'id': ['d1', 'd2'], 'ancestors': [[], ['d1', 'd1', 'd2']]},
    schema={'id': pl.String, 'ancestors': pl.List(pl.String)},
)
WEIGHTS = pl.DataFrame({
    'datasourceId': ['ds', 'expression_atlas'],
    'weight': [1.0, 0.2],
    'datatypeId': ['dt', 'rna_expression'],
    'toPropagate': [True, False],
})


def _ids(out: pl.LazyFrame) -> list[str]:
    return sorted(out.collect()['diseaseId'].to_list())


def test_expand_disease_explodes_to_self_and_ancestors_deduplicated() -> None:
    ev = _evidence([('t1', 'd2', 'ds', 'dt', 0.5, 2020)])
    out = expand_disease(ev, DISEASE, WEIGHTS)
    assert _ids(out) == ['d1', 'd2']
    assert out.collect().columns == ['targetId', 'diseaseId', 'datasourceId', 'datatypeId', 'score', 'year']


def test_expand_disease_keeps_non_propagated_direct() -> None:
    ev = _evidence([('t1', 'd2', 'expression_atlas', 'rna_expression', 0.5, 2020)])
    assert _ids(expand_disease(ev, DISEASE, WEIGHTS)) == ['d2']


def test_expand_disease_unknown_datasource_propagates() -> None:
    ev = _evidence([('t1', 'd2', 'mystery', 'dt', 0.5, 2020)])
    assert _ids(expand_disease(ev, DISEASE, WEIGHTS)) == ['d1', 'd2']


def test_expand_disease_drops_propagated_evidence_on_unknown_disease() -> None:
    ev = _evidence([('t1', 'dX', 'ds', 'dt', 0.5, 2020)])
    assert expand_disease(ev, DISEASE, WEIGHTS).collect().height == 0
