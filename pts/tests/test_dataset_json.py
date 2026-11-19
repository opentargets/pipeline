"""JSON-lines dataset writers."""

import datetime

import polars as pl
import pytest

from pts.transformers.utils.dataset import drop_nulls, write_json_dataset, write_json_file

# produced by pos.parquet2json.converter.Converter._serialize_rows on the same row
POS_LINE = (
    b'{"id":"ENSG1","score":0.00001,"day":"2026-11-19",'
    b'"name":"\xc3\x9cn\xc3\xafc\xc3\xb6d\xc3\xa9 \xe6\x97\xa5\xe6\x9c\xac",'
    b'"empty":[],"hits":[{"a":1},{"b":"x"}]}\n'
)


def test_drop_nulls_recursive():
    value = {'a': None, 'b': [None, {'c': None, 'd': 1}], 'e': {'f': None}}
    assert drop_nulls(value) == {'b': [{'d': 1}], 'e': {}}


def _pos_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            'id': ['ENSG1'],
            'score': [1e-05],
            'day': [datetime.date(2026, 11, 19)],
            'name': ['Ünïcödé 日本'],
            'empty': [[]],
            'hits': [[{'a': 1, 'b': None}, None, {'a': None, 'b': 'x'}]],
        },
        schema={
            'id': pl.String,
            'score': pl.Float64,
            'day': pl.Date,
            'name': pl.String,
            'empty': pl.List(pl.String),
            'hits': pl.List(pl.Struct({'a': pl.Int64, 'b': pl.String})),
        },
    )


def test_json_line_matches_pos_converter(tmp_path):
    write_json_file(_pos_frame(), str(tmp_path / 'out.json'))
    assert (tmp_path / 'out.json').read_bytes() == POS_LINE


def test_json_dataset_parts(tmp_path):
    frame = pl.DataFrame({'i': range(1200), 's': [None if i % 7 == 0 else str(i) for i in range(1200)]})
    write_json_dataset(frame, str(tmp_path / 'ds'), rows_per_file=500)
    parts = sorted((tmp_path / 'ds').iterdir())
    assert [p.name for p in parts] == ['part-00000.json', 'part-00001.json', 'part-00002.json']
    expected = b''.join(
        (b'{"i":%d}\n' % r['i']) if r['s'] is None else (b'{"i":%d,"s":"%s"}\n' % (r['i'], r['s'].encode()))
        for r in frame.iter_rows(named=True)
    )
    assert b''.join(p.read_bytes() for p in parts) == expected


def test_json_writers_refuse_existing(tmp_path):
    frame = pl.DataFrame({'a': [1]})
    directory = tmp_path / 'd'
    directory.mkdir()
    file = tmp_path / 'f.json'
    file.write_text('x')
    with pytest.raises(ValueError, match='already exists'):
        write_json_dataset(frame, str(directory))
    with pytest.raises(ValueError, match='already exists'):
        write_json_file(frame, str(file))
    assert file.read_text() == 'x'
