"""Tests for the search_ebi transformer."""

from pathlib import Path

import polars as pl

from pts.transformers.search_ebi import generate_datasets, search_ebi

COLUMNS = ['targetId', 'diseaseId', 'approvedSymbol', 'name', 'score']


def disease(rows: list[tuple]) -> pl.LazyFrame:
    return pl.LazyFrame(rows, schema={'diseaseId': pl.String, 'name': pl.String}, orient='row')


def target(rows: list[tuple]) -> pl.LazyFrame:
    return pl.LazyFrame(rows, schema={'targetId': pl.String, 'approvedSymbol': pl.String}, orient='row')


def association(rows: list[tuple]) -> pl.LazyFrame:
    schema = {'targetId': pl.String, 'diseaseId': pl.String, 'associationScore': pl.Float64}
    return pl.LazyFrame(rows, schema=schema, orient='row')


def evidence(rows: list[tuple]) -> pl.LazyFrame:
    return pl.LazyFrame(rows, schema={'targetId': pl.String, 'diseaseId': pl.String, 'score': pl.Float64}, orient='row')


def test_associations_output_columns() -> None:
    out = generate_datasets(
        disease([('D1', 'Cancer')]),
        target([('T1', 'BRCA1')]),
        association([('T1', 'D1', 0.9)]),
        evidence([]),
    )
    assert out['associations'].collect_schema().names() == COLUMNS


def test_associations_inner_join() -> None:
    out = generate_datasets(
        disease([('D1', 'Cancer')]),
        target([('T1', 'BRCA1')]),
        association([('T1', 'D1', 0.9), ('T2', 'D1', 0.5)]),
        evidence([]),
    )
    assert out['associations'].collect().rows() == [('T1', 'D1', 'BRCA1', 'Cancer', 0.9)]


def test_associations_score_alias_keeps_float64() -> None:
    out = generate_datasets(
        disease([('D1', 'Cancer')]),
        target([('T1', 'BRCA1')]),
        association([('T1', 'D1', 0.8)]),
        evidence([]),
    )
    got = out['associations'].collect()
    assert got.schema['score'] == pl.Float64
    assert got['score'].to_list() == [0.8]


def test_evidence_inner_join() -> None:
    out = generate_datasets(
        disease([('D1', 'Cancer')]),
        target([('T1', 'BRCA1')]),
        association([]),
        evidence([('T1', 'D1', 0.7), ('T1', 'D2', 0.5)]),
    )
    assert out['evidence'].collect().rows() == [('T1', 'D1', 'BRCA1', 'Cancer', 0.7)]


def test_evidence_duplicates_kept_unknown_ids_dropped() -> None:
    d = pl.LazyFrame({'diseaseId': ['D1'], 'name': ['dis']})
    t = pl.LazyFrame({'targetId': ['T1'], 'approvedSymbol': ['SYM']})
    a = pl.LazyFrame({'targetId': ['T1', 'T9'], 'diseaseId': ['D1', 'D1'], 'associationScore': [0.5, 0.1]})
    e = pl.LazyFrame({'targetId': ['T1', 'T1'], 'diseaseId': ['D1', 'D1'], 'score': [0.2, 0.2]})
    out = generate_datasets(d, t, a, e)
    assert out['associations'].collect().rows() == [('T1', 'D1', 'SYM', 'dis', 0.5)]
    assert out['evidence'].collect().height == 2


def test_extra_columns_do_not_leak() -> None:
    d = disease([('D1', 'Cancer')]).with_columns(pl.lit('x').alias('score'))
    t = target([('T1', 'BRCA1')]).with_columns(pl.lit('y').alias('name'))
    out = generate_datasets(d, t, association([('T1', 'D1', 0.4)]), evidence([('T1', 'D1', 0.3)]))
    assert out['associations'].collect().rows() == [('T1', 'D1', 'BRCA1', 'Cancer', 0.4)]
    assert out['evidence'].collect().rows() == [('T1', 'D1', 'BRCA1', 'Cancer', 0.3)]


def test_search_ebi_end_to_end_two_chunks_deterministic(tmp_path: Path) -> None:
    pl.DataFrame({'id': ['D1', 'D2'], 'name': ['a', 'b'], 'extra': [1, 2]}).write_parquet(tmp_path / 'disease.parquet')
    pl.DataFrame({'id': ['T1', 'T2'], 'approvedSymbol': ['S1', 'S2'], 'extra': [1, 2]}).write_parquet(
        tmp_path / 'target.parquet'
    )
    assoc = pl.concat(
        [
            pl.DataFrame({'targetId': ['T2'], 'diseaseId': ['D2'], 'associationScore': [0.1]}),
            pl.DataFrame({'targetId': ['T1', 'T9'], 'diseaseId': ['D1', 'D1'], 'associationScore': [0.9, 0.2]}),
        ],
        rechunk=False,
    )
    assoc.write_parquet(tmp_path / 'assoc.parquet')
    ev = pl.concat(
        [
            pl.DataFrame({'targetId': ['T2', 'T1'], 'diseaseId': ['D1', 'D1'], 'score': [0.3, 0.6]}),
            pl.DataFrame({'targetId': ['T1'], 'diseaseId': ['D1'], 'score': [0.6]}),
        ],
        rechunk=False,
    )
    ev.write_parquet(tmp_path / 'ev.parquet')
    source = {
        'disease': str(tmp_path / 'disease.parquet'),
        'target': str(tmp_path / 'target.parquet'),
        'association': str(tmp_path / 'assoc.parquet'),
        'evidence': str(tmp_path / 'ev.parquet'),
    }
    blobs = []
    for run in ('a', 'b'):
        dest = {'associations': str(tmp_path / run / 'assoc'), 'evidence': str(tmp_path / run / 'ev')}
        search_ebi(source, dest, None, None)
        blobs.append({k: [p.read_bytes() for p in sorted(Path(v).glob('*.parquet'))] for k, v in dest.items()})
        got_a = pl.read_parquet(Path(dest['associations']) / '*.parquet')
        got_e = pl.read_parquet(Path(dest['evidence']) / '*.parquet')
        assert got_a.columns == COLUMNS
        assert got_a.rows() == [('T1', 'D1', 'S1', 'a', 0.9), ('T2', 'D2', 'S2', 'b', 0.1)]
        assert got_e.rows() == [
            ('T1', 'D1', 'S1', 'a', 0.6),
            ('T1', 'D1', 'S1', 'a', 0.6),
            ('T2', 'D1', 'S2', 'a', 0.3),
        ]
    assert blobs[0] == blobs[1]
