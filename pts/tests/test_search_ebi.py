"""Tests for the search_ebi transformer."""

from pathlib import Path

import polars as pl

from pts.transformers.search_ebi import generate_associations, search_ebi

COLUMNS = ['targetId', 'diseaseId', 'approvedSymbol', 'name', 'score']


def disease(rows: list[tuple]) -> pl.LazyFrame:
    return pl.LazyFrame(rows, schema={'diseaseId': pl.String, 'name': pl.String}, orient='row')


def target(rows: list[tuple]) -> pl.LazyFrame:
    return pl.LazyFrame(rows, schema={'targetId': pl.String, 'approvedSymbol': pl.String}, orient='row')


def association(rows: list[tuple]) -> pl.LazyFrame:
    schema = {'targetId': pl.String, 'diseaseId': pl.String, 'associationScore': pl.Float64}
    return pl.LazyFrame(rows, schema=schema, orient='row')


def test_associations_output_columns() -> None:
    out = generate_associations(
        disease([('D1', 'Cancer')]),
        target([('T1', 'BRCA1')]),
        association([('T1', 'D1', 0.9)]),
    )
    assert out.collect_schema().names() == COLUMNS


def test_associations_inner_join() -> None:
    out = generate_associations(
        disease([('D1', 'Cancer')]),
        target([('T1', 'BRCA1')]),
        association([('T1', 'D1', 0.9), ('T2', 'D1', 0.5)]),
    )
    assert out.collect().rows() == [('T1', 'D1', 'BRCA1', 'Cancer', 0.9)]


def test_associations_score_alias_keeps_float64() -> None:
    out = generate_associations(
        disease([('D1', 'Cancer')]),
        target([('T1', 'BRCA1')]),
        association([('T1', 'D1', 0.8)]),
    )
    got = out.collect()
    assert got.schema['score'] == pl.Float64
    assert got['score'].to_list() == [0.8]


def test_unknown_ids_dropped_duplicates_kept() -> None:
    d = pl.LazyFrame({'diseaseId': ['D1'], 'name': ['dis']})
    t = pl.LazyFrame({'targetId': ['T1'], 'approvedSymbol': ['SYM']})
    a = pl.LazyFrame(
        {'targetId': ['T1', 'T9', 'T1'], 'diseaseId': ['D1', 'D1', 'D1'], 'associationScore': [0.5, 0.1, 0.5]}
    )
    assert generate_associations(d, t, a).collect().height == 2


def test_extra_columns_do_not_leak() -> None:
    d = disease([('D1', 'Cancer')]).with_columns(pl.lit('x').alias('score'))
    t = target([('T1', 'BRCA1')]).with_columns(pl.lit('y').alias('name'))
    out = generate_associations(d, t, association([('T1', 'D1', 0.4)]))
    assert out.collect().rows() == [('T1', 'D1', 'BRCA1', 'Cancer', 0.4)]


def test_search_ebi_end_to_end_writes_one_json_file(tmp_path: Path) -> None:
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
    source = {
        'disease': str(tmp_path / 'disease.parquet'),
        'target': str(tmp_path / 'target.parquet'),
        'association': str(tmp_path / 'assoc.parquet'),
    }
    blobs = []
    for run in ('a', 'b'):
        dest = tmp_path / run / 'etc' / 'export' / 'search_ebi_associations' / 'search_ebi_associations.json'
        search_ebi(source, {'associations': str(dest)}, None, None)
        assert sorted(p.name for p in dest.parent.iterdir()) == [dest.name]
        blobs.append(dest.read_bytes())
        got = pl.read_ndjson(dest)
        assert got.columns == COLUMNS
        assert got.rows() == [('T1', 'D1', 'S1', 'a', 0.9), ('T2', 'D2', 'S2', 'b', 0.1)]
    assert blobs[0] == blobs[1]
