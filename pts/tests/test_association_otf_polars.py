"""Tests for the association_otf transformer."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pts.schemas.association import association_otf_schema
from pts.transformers.association_otf import association_otf
from pts.transformers.utils.dataset import scan_dataset, sink_dataset

EVIDENCE = {'id': pl.String, 'diseaseId': pl.String, 'targetId': pl.String, 'datasourceId': pl.String,
            'datatypeId': pl.String, 'score': pl.Float64, 'extra': pl.String}  # fmt: skip


def _write(path: Path, frame: pl.DataFrame) -> str:
    path.mkdir(parents=True)
    frame.write_parquet(path / 'part-0.parquet')
    return str(path)


@pytest.fixture
def source(tmp_path: Path) -> dict[str, str]:
    evidence_a = pl.DataFrame(
        [('e1', 'EFO_1', 'ENSG1', 'ds1', 'dt1', 0.5, 'x'), ('e2', 'EFO_2', 'ENSG2', 'ds1', 'dt1', 0.7, 'x')],
        schema=EVIDENCE,
        orient='row',
    )
    evidence_b = pl.DataFrame(
        [('e3', 'EFO_1', 'ENSG2', 'ds2', 'dt2', 0.1, 'x'), ('e4', 'EFO_9', 'ENSG1', 'ds2', 'dt2', 0.9, 'x')],
        schema=EVIDENCE,
        orient='row',
    )
    disease = pl.DataFrame({'id': ['EFO_1', 'EFO_2'], 'name': ['name', 'other']})
    target = pl.DataFrame({
        'id': ['ENSG1', 'ENSG2'],
        'approvedName': ['approvedName', None],
        'approvedSymbol': ['SYMBOL', 'S2'],
    })
    novelty = pl.DataFrame({'targetId': ['ENSG1'], 'diseaseId': ['EFO_1'], 'currentNovelty': [1.5]})
    novelty_indirect = pl.DataFrame({
        'targetId': ['ENSG1', 'ENSG2'],
        'diseaseId': ['EFO_1', 'EFO_1'],
        'currentNovelty': [2.5, 3.5],
    })
    _write(tmp_path / 'evidence_a', evidence_a)
    _write(tmp_path / 'evidence_b', evidence_b)
    return {
        'evidence': str(tmp_path / 'evidence_*'),
        'disease': _write(tmp_path / 'disease', disease),
        'target': _write(tmp_path / 'target', target),
        'association_direct': _write(tmp_path / 'direct', novelty),
        'association_indirect': _write(tmp_path / 'indirect', novelty_indirect),
    }


def _run(source: dict[str, str], tmp_path: Path) -> pl.DataFrame:
    destination = str(tmp_path / 'out')
    association_otf(source, destination, None, None)
    return scan_dataset(destination).collect()


def test_schema_and_order(source: dict[str, str], tmp_path: Path) -> None:
    assert _run(source, tmp_path).schema == association_otf_schema()


def test_row_count_equals_evidence(source: dict[str, str], tmp_path: Path) -> None:
    assert _run(source, tmp_path).height == 4


def test_labels(source: dict[str, str], tmp_path: Path) -> None:
    row = _run(source, tmp_path).filter(pl.col('rowId') == 'e1').row(0, named=True)
    assert row['diseaseData'] == 'EFO_1 name'
    assert row['targetData'] == 'ENSG1 approvedName SYMBOL'
    assert row['noveltyDirect'] == 1.5
    assert row['noveltyIndirect'] == 2.5


def test_null_approved_name_gives_null_target_data(source: dict[str, str], tmp_path: Path) -> None:
    row = _run(source, tmp_path).filter(pl.col('rowId') == 'e2').row(0, named=True)
    assert row['targetData'] is None
    assert row['diseaseData'] == 'EFO_2 other'


def test_unmatched_gets_nulls(source: dict[str, str], tmp_path: Path) -> None:
    result = _run(source, tmp_path)
    e3 = result.filter(pl.col('rowId') == 'e3').row(0, named=True)
    assert e3['noveltyDirect'] is None
    assert e3['noveltyIndirect'] == 3.5
    e4 = result.filter(pl.col('rowId') == 'e4').row(0, named=True)
    assert e4['diseaseData'] is None
    assert e4['noveltyIndirect'] is None


def test_sorted_by_row_id(source: dict[str, str], tmp_path: Path) -> None:
    assert _run(source, tmp_path)['rowId'].to_list() == ['e1', 'e2', 'e3', 'e4']


def test_refuses_existing_destination(source: dict[str, str], tmp_path: Path) -> None:
    _run(source, tmp_path)
    with pytest.raises(ValueError, match='exist'):
        _run(source, tmp_path)


def test_rerun_is_byte_identical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('pts.transformers.association_otf.sink_dataset', _small_parts)
    rows = 200_000
    evidence = pl.DataFrame({
        'id': [f'e{i:07d}' for i in range(rows)],
        'diseaseId': [f'EFO_{i % 500}' for i in range(rows)],
        'targetId': [f'ENSG{i % 700}' for i in range(rows)],
        'datasourceId': ['ds'] * rows,
        'datatypeId': ['dt'] * rows,
        'score': [i / rows for i in range(rows)],
    })
    disease = pl.DataFrame({'id': [f'EFO_{i}' for i in range(500)], 'name': [f'name {i}' for i in range(500)]})
    target = pl.DataFrame({
        'id': [f'ENSG{i}' for i in range(700)],
        'approvedName': [f'n{i}' for i in range(700)],
        'approvedSymbol': [f's{i}' for i in range(700)],
    })
    novelty = pl.DataFrame({
        'targetId': [f'ENSG{i % 700}' for i in range(1000)],
        'diseaseId': [f'EFO_{i % 500}' for i in range(1000)],
        'currentNovelty': [i / 10 for i in range(1000)],
    }).unique(['targetId', 'diseaseId'])
    _write(tmp_path / 'evidence_big', evidence)
    paths = {
        'evidence': str(tmp_path / 'evidence_*'),
        'disease': _write(tmp_path / 'disease', disease),
        'target': _write(tmp_path / 'target', target),
        'association_direct': _write(tmp_path / 'direct', novelty),
        'association_indirect': _write(tmp_path / 'indirect', novelty),
    }
    for name in ('a', 'b'):
        association_otf(paths, str(tmp_path / name), None, None)
    files_a = sorted(p.name for p in (tmp_path / 'a').rglob('*.parquet'))
    assert len(files_a) > 1
    assert files_a == sorted(p.name for p in (tmp_path / 'b').rglob('*.parquet'))
    for name in files_a:
        assert (tmp_path / 'a' / name).read_bytes() == (tmp_path / 'b' / name).read_bytes()


def _small_parts(frame: pl.LazyFrame, path: str) -> pl.LazyFrame:
    return sink_dataset(frame, path, approximate_bytes_per_file=200_000)
