"""End-to-end tests for the association transformer."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from pts.schemas.association import association_schema
from pts.transformers.association import association
from pts.transformers.utils.dataset import scan_dataset

OUTPUTS = [f'{kind}_{way}' for kind in ('overall', 'by_datasource', 'by_datatype') for way in ('direct', 'indirect')]
WEIGHTS = [
    {'datasourceId': 'ds1', 'weight': 1.0, 'datatypeId': 'dt1', 'toPropagate': True},
    {'datasourceId': 'ds2', 'weight': 0.5, 'datatypeId': 'dt2', 'toPropagate': False},
]
EVIDENCE_SCHEMA = {
    'targetId': pl.String,
    'diseaseId': pl.String,
    'datasourceId': pl.String,
    'datatypeId': pl.String,
    'score': pl.Float64,
    'evidenceDate': pl.String,
    'clinicalReportId': pl.String,
}
ROWS = [
    (f't{i}', d, ds, dt, 0.3 + 0.1 * (i % 5), f'{2015 + i % 6}-01-01', None)
    for i in range(150)
    for d in ('d_leaf', 'd_other')
    for ds, dt in (('ds1', 'dt1'), ('ds2', 'dt2'))
]


def _stage(root: Path, rows: list[tuple] = ROWS) -> tuple[dict, dict]:
    frame = pl.DataFrame(rows, schema=EVIDENCE_SCHEMA, orient='row')
    size = max(-(-len(rows) // 6), 1)
    chunks = [frame.slice(i, size) for i in range(0, max(len(rows), 1), size)]
    for i, part in enumerate(pl.concat(chunks, rechunk=False).iter_slices(size) if rows else [frame]):
        directory = root / 'output' / f'evidence_{i % 2}'
        directory.mkdir(parents=True, exist_ok=True)
        part.write_parquet(directory / f'part-{i}.parquet')
    disease = root / 'output' / 'disease'
    disease.mkdir()
    pl.DataFrame(
        {'id': ['d_leaf', 'd_other', 'd_root'], 'ancestors': [['d_root'], [], []]},
        schema={'id': pl.String, 'ancestors': pl.List(pl.String)},
    ).write_parquet(disease / 'part-0.parquet')
    source = {'evidence': str(root / 'output' / 'evidence_*'), 'disease': str(disease)}
    return source, {name: str(root / 'out' / name) for name in OUTPUTS}


def _settings(bucket_count: int) -> dict:
    return {
        'novelty_scale': 2,
        'novelty_shift': 3,
        'novelty_window': 10,
        'datasource_weights': WEIGHTS,
        'bucket_count': bucket_count,
    }


def _normalised(path: str) -> pl.DataFrame:
    df = scan_dataset(path).collect()
    keys = ['diseaseId', 'targetId', 'aggregationType', 'aggregationValue']
    return df.sort(keys).with_columns(pl.col('timeseries').list.eval(pl.element().sort_by(pl.element().struct['year'])))


def test_end_to_end_writes_six_outputs_with_schemas(tmp_path: Path) -> None:
    source, destination = _stage(tmp_path)
    association(source, destination, _settings(4), None)
    for name in OUTPUTS:
        df = scan_dataset(destination[name]).collect()
        dtype = pl.Int32() if name.startswith('by_datasource') else pl.Int64()
        assert df.schema == association_schema(dtype)
        assert df.height > 0
    assert scan_dataset(destination['overall_indirect']).filter(pl.col('diseaseId') == 'd_root').collect().height > 0
    assert scan_dataset(destination['overall_direct']).filter(pl.col('diseaseId') == 'd_root').collect().height == 0


def test_bucket_count_does_not_change_results(tmp_path: Path) -> None:
    one_source, one_dest = _stage(tmp_path / 'one')
    many_source, many_dest = _stage(tmp_path / 'many')
    association(one_source, one_dest, _settings(1), None)
    association(many_source, many_dest, _settings(7), None)
    for name in OUTPUTS:
        assert _normalised(one_dest[name]).equals(_normalised(many_dest[name])), name


def test_rerun_is_byte_identical(tmp_path: Path) -> None:
    a_source, a_dest = _stage(tmp_path / 'a')
    b_source, b_dest = _stage(tmp_path / 'b')
    association(a_source, a_dest, _settings(3), None)
    association(b_source, b_dest, _settings(3), None)
    for name in OUTPUTS:
        a_files = sorted(Path(a_dest[name]).glob('*.parquet'))
        b_files = sorted(Path(b_dest[name]).glob('*.parquet'))
        assert [f.name for f in a_files] == [f.name for f in b_files]
        assert [f.read_bytes() for f in a_files] == [f.read_bytes() for f in b_files]


def test_empty_bucket_is_handled(tmp_path: Path) -> None:
    source, destination = _stage(tmp_path / 'sparse')
    ref_source, ref_dest = _stage(tmp_path / 'ref')
    association(source, destination, _settings(200), None)
    association(ref_source, ref_dest, _settings(1), None)
    for name in OUTPUTS:
        assert scan_dataset(destination[name]).collect().height == scan_dataset(ref_dest[name]).collect().height


def test_no_evidence_gives_empty_outputs_with_schemas(tmp_path: Path) -> None:
    source, destination = _stage(tmp_path, rows=[])
    association(source, destination, _settings(3), None)
    for name in OUTPUTS:
        df = scan_dataset(destination[name]).collect()
        dtype = pl.Int32() if name.startswith('by_datasource') else pl.Int64()
        assert df.schema == association_schema(dtype)
        assert df.height == 0


def test_refuses_existing_destination(tmp_path: Path) -> None:
    source, destination = _stage(tmp_path)
    Path(destination['overall_direct']).mkdir(parents=True)
    with pytest.raises(ValueError, match='already exists'):
        association(source, destination, _settings(2), None)


def test_future_dated_evidence_counts_as_current_year(tmp_path: Path) -> None:
    rows = [('t_future', 'd_other', 'ds1', 'dt1', 0.8, '2040-01-01', None)]
    source, destination = _stage(tmp_path, rows=rows)
    association(source, destination, _settings(2), None)
    current_year = datetime.now().year
    out = scan_dataset(destination['overall_direct']).filter(pl.col('targetId') == 't_future').collect()
    assert out.height == 1
    assert out['associationScore'][0] > 0
    assert out['timeseries'][0].struct.field('year').drop_nulls().max() == current_year
    assert out['currentNovelty'][0] > 0
