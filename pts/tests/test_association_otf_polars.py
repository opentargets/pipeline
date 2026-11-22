"""Tests for the association_otf transformer."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from ot_datamodels.polars import to_polars_schema
from ot_datamodels.views.association_otf import AssociationOtf

from pts.schemas.dataset.validate import validate_dataset
from pts.task_outputs import OutputValidation
from pts.transformers import association_otf as module
from pts.transformers.association_otf import association_otf
from pts.transformers.utils.dataset import scan_dataset, sink_dataset

EVIDENCE = {'id': pl.String, 'diseaseId': pl.String, 'targetId': pl.String, 'datasourceId': pl.String,
            'datatypeId': pl.String, 'score': pl.Float64, 'extra': pl.String}  # fmt: skip


def _write(path: Path, frame: pl.DataFrame) -> str:
    path.mkdir(parents=True)
    frame.write_parquet(path / 'part-0.parquet')
    return str(path)


E1, E2, E3, E4 = (f'{i:040x}' for i in range(1, 5))
T1, T2 = 'ENSG00000000001', 'ENSG00000000002'


@pytest.fixture
def source(tmp_path: Path) -> dict[str, str]:
    evidence_a = pl.DataFrame(
        [
            (E1, 'EFO_1', T1, 'eva', 'genetic_association', 0.5, 'x'),
            (E2, 'EFO_2', T2, 'eva', 'genetic_association', 0.7, 'x'),
        ],
        schema=EVIDENCE,
        orient='row',
    )
    evidence_b = pl.DataFrame(
        [
            (E3, 'EFO_1', T2, 'europepmc', 'literature', 0.1, 'x'),
            (E4, 'EFO_9', T1, 'europepmc', 'literature', 0.9, 'x'),
        ],
        schema=EVIDENCE,
        orient='row',
    )
    disease = pl.DataFrame({'id': ['EFO_1', 'EFO_2'], 'name': ['name', 'other']})
    target = pl.DataFrame({
        'id': [T1, T2],
        'approvedName': ['approvedName', None],
        'approvedSymbol': ['SYMBOL', 'S2'],
    })
    novelty = pl.DataFrame({'targetId': [T1], 'diseaseId': ['EFO_1'], 'currentNovelty': [0.15]})
    novelty_indirect = pl.DataFrame({
        'targetId': [T1, T2],
        'diseaseId': ['EFO_1', 'EFO_1'],
        'currentNovelty': [0.25, 0.35],
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
    assert _run(source, tmp_path).schema == to_polars_schema(AssociationOtf)


def test_writes_its_model(source: dict[str, str], tmp_path: Path) -> None:
    assert OutputValidation.of(association_otf, module).outputs == {None: AssociationOtf}
    association_otf(source, str(tmp_path / 'out'), None, None)
    validate_dataset(AssociationOtf, str(tmp_path / 'out'))


def test_evidence_as_pts_writes_it_gives_the_same_view(source: dict[str, str], tmp_path: Path) -> None:
    """Evidence stores its datasource and datatype as the shared Enums; the view keeps them."""
    strings = _run(source, tmp_path / 'strings')
    schema = to_polars_schema(AssociationOtf)
    for path in tmp_path.glob('evidence_*/part-0.parquet'):
        frame = pl.read_parquet(path)
        frame.cast({'datasourceId': schema['datasourceId'], 'datatypeId': schema['datatypeId']}).write_parquet(path)

    assert _run(source, tmp_path / 'typed').equals(strings)


def test_row_count_equals_evidence(source: dict[str, str], tmp_path: Path) -> None:
    assert _run(source, tmp_path).height == 4


def test_labels(source: dict[str, str], tmp_path: Path) -> None:
    row = _run(source, tmp_path).filter(pl.col('rowId') == E1).row(0, named=True)
    assert row['diseaseData'] == 'EFO_1 name'
    assert row['targetData'] == f'{T1} approvedName SYMBOL'
    assert row['noveltyDirect'] == 0.15
    assert row['noveltyIndirect'] == 0.25


def test_null_approved_name_gives_null_target_data(source: dict[str, str], tmp_path: Path) -> None:
    row = _run(source, tmp_path).filter(pl.col('rowId') == E2).row(0, named=True)
    assert row['targetData'] is None
    assert row['diseaseData'] == 'EFO_2 other'


def test_unmatched_gets_nulls(source: dict[str, str], tmp_path: Path) -> None:
    result = _run(source, tmp_path)
    e3 = result.filter(pl.col('rowId') == E3).row(0, named=True)
    assert e3['noveltyDirect'] is None
    assert e3['noveltyIndirect'] == 0.35
    e4 = result.filter(pl.col('rowId') == E4).row(0, named=True)
    assert e4['diseaseData'] is None
    assert e4['noveltyIndirect'] is None


def test_sorted_by_row_id(source: dict[str, str], tmp_path: Path) -> None:
    assert _run(source, tmp_path)['rowId'].to_list() == [E1, E2, E3, E4]


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
        'datasourceId': ['eva'] * rows,
        'datatypeId': ['genetic_association'] * rows,
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
