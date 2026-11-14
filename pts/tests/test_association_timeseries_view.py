"""Tests for the association timeseries view transformer."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pts.schemas.association import association_schema, association_timeseries_schema
from pts.transformers.association_timeseries_view import association_timeseries_view

NAMES = ['overall_direct', 'overall_indirect', 'by_datasource_direct', 'by_datasource_indirect']


def _entry(year: int, count: int = 1) -> dict:
    return {'year': year, 'associationScore': 0.5, 'novelty': 0.1, 'yearlyEvidenceCount': count}


def _row(target: str, disease: str, value: str | None, years: list[int] | None) -> dict:
    return {
        'diseaseId': disease,
        'targetId': target,
        'aggregationType': None if value is None else 'datasource',
        'aggregationValue': value,
        'associationScore': 0.5,
        'evidenceCount': 1,
        'timeseries': None if years is None else [_entry(y) for y in years],
        'currentNovelty': 0.1,
    }


def _write(tmp_path: Path, rows: dict[str, list[dict]]) -> dict[str, str]:
    source = {}
    for name in NAMES:
        dtype = pl.Int32() if name.startswith('by_datasource') else pl.Int64()
        path = tmp_path / f'association_{name}'
        path.mkdir()
        pl.DataFrame(rows.get(name, []), schema=association_schema(dtype)).write_parquet(path / 'part-0.parquet')
        source[f'association_{name}'] = str(path)
    return source


def _run(tmp_path: Path, rows: dict[str, list[dict]], file_count: int = 3) -> pl.DataFrame:
    destination = tmp_path / 'out'
    association_timeseries_view(_write(tmp_path, rows), str(destination), {'file_count': file_count}, None)
    return pl.read_parquet(destination / '*.parquet')


def test_partition_key_renders_booleans_like_spark(tmp_path: Path) -> None:
    out = _run(
        tmp_path,
        {
            'overall_direct': [_row('ENSG1', 'EFO_1', None, [2020])],
            'overall_indirect': [_row('ENSG1', 'EFO_1', None, [2020])],
        },
    )
    assert sorted(out['partitionKey']) == ['ENSG1_EFO_1_false', 'ENSG1_EFO_1_true']


def test_partition_key_skips_null_ids_like_concat_ws(tmp_path: Path) -> None:
    out = _run(tmp_path, {'overall_direct': [_row('ENSG1', None, None, [2020])]})
    assert out['partitionKey'].to_list() == ['ENSG1_true']


def test_explodes_every_timeseries_entry(tmp_path: Path) -> None:
    rows = {
        'overall_direct': [_row('T1', 'D1', None, [2018, 2019, 2020]), _row('T2', 'D1', None, [2020])],
        'by_datasource_indirect': [_row('T1', 'D2', 'chembl', [2019, 2020])],
    }
    out = _run(tmp_path, rows)
    assert out.height == 6
    assert out.schema == association_timeseries_schema()


def test_by_datasource_int32_counts_upcast_to_int64(tmp_path: Path) -> None:
    out = _run(tmp_path, {'by_datasource_direct': [_row('T1', 'D1', 'chembl', [2020])]})
    assert out.schema['yearlyEvidenceCount'] == pl.Int64()
    assert out['yearlyEvidenceCount'].to_list() == [1]


def test_empty_timeseries_rows_vanish(tmp_path: Path) -> None:
    out = _run(tmp_path, {'overall_direct': [_row('T1', 'D1', None, []), _row('T2', 'D1', None, None)]})
    assert out.height == 0
    assert out.schema == association_timeseries_schema()


def test_each_partition_key_in_exactly_one_file_and_files_sorted(tmp_path: Path) -> None:
    rows = {
        name: [_row(f'T{t}', f'D{d}', v, [2021, 2019, 2020]) for t in range(8) for d in range(3) for v in values]
        for name, values in (
            ('overall_direct', [None]),
            ('overall_indirect', [None]),
            ('by_datasource_direct', ['b', 'a']),
            ('by_datasource_indirect', ['b', 'a']),
        )
    }
    _run(tmp_path, rows)
    files = sorted((tmp_path / 'out').glob('part-*.parquet'))
    assert [f.name for f in files] == ['part-00000.parquet', 'part-00001.parquet', 'part-00002.parquet']
    seen: dict[str, str] = {}
    for file in files:
        frame = pl.read_parquet(file)
        assert frame.equals(frame.sort('partitionKey', 'aggregationValue', 'year'))
        for key in frame['partitionKey'].unique():
            assert seen.setdefault(key, file.name) == file.name
    assert len(seen) == 8 * 3 * 2


def test_refuses_existing_destination(tmp_path: Path) -> None:
    source = _write(tmp_path, {})
    (tmp_path / 'out').mkdir()
    with pytest.raises(ValueError, match='already exists'):
        association_timeseries_view(source, str(tmp_path / 'out'), {'file_count': 2}, None)


def test_rerun_is_byte_identical(tmp_path: Path) -> None:
    rows = {
        name: [_row(f'T{t}', f'D{d}', v, [2021, 2019, 2020, 2018]) for t in range(40) for d in range(5) for v in values]
        for name, values in (
            ('overall_direct', [None]),
            ('overall_indirect', [None]),
            ('by_datasource_direct', ['b', 'a', 'c']),
            ('by_datasource_indirect', ['b', 'a', 'c']),
        )
    }
    source = _write(tmp_path, rows)
    for run in ('one', 'two'):
        association_timeseries_view(source, str(tmp_path / run), {'file_count': 4}, None)
    first = sorted((tmp_path / 'one').glob('part-*.parquet'))
    second = sorted((tmp_path / 'two').glob('part-*.parquet'))
    assert [f.name for f in first] == [f.name for f in second]
    assert all(a.read_bytes() == b.read_bytes() for a, b in zip(first, second, strict=True))


def test_rerun_of_large_files_is_byte_identical(tmp_path: Path) -> None:
    rows = {'overall_direct': [_row(f'T{t}', 'D1', None, list(range(2010, 2020))) for t in range(100_000)]}
    source = _write(tmp_path, rows)
    for run in ('one', 'two'):
        association_timeseries_view(source, str(tmp_path / run), {'file_count': 1}, None)
    first, second = ((tmp_path / run / 'part-00000.parquet').read_bytes() for run in ('one', 'two'))
    assert first == second


def test_concurrency_does_not_change_bytes(tmp_path: Path) -> None:
    rows = {'overall_direct': [_row(f'T{t}', f'D{t % 7}', None, [2019, 2020]) for t in range(500)]}
    source = _write(tmp_path, rows)
    for run, concurrency in (('one', 1), ('four', 4)):
        association_timeseries_view(source, str(tmp_path / run), {'file_count': 6, 'concurrency': concurrency}, None)
    for name in [f'part-{i:05d}.parquet' for i in range(6)]:
        assert (tmp_path / 'one' / name).read_bytes() == (tmp_path / 'four' / name).read_bytes()
