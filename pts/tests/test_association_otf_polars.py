"""Tests for the association_otf transformer."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from ot_datamodels.polars import to_polars_schema
from ot_datamodels.views.association_otf_disease import AssociationOtfDisease
from ot_datamodels.views.association_otf_target import AssociationOtfTarget

from pts.schemas.dataset.validate import validate_dataset
from pts.task_outputs import OutputValidation
from pts.transformers import association_otf as module
from pts.transformers.association_otf import ascii_lower, association_otf
from pts.transformers.utils.dataset import scan_dataset, sink_dataset

EVIDENCE = {'id': pl.String, 'diseaseId': pl.String, 'targetId': pl.String, 'datasourceId': pl.String,
            'datatypeId': pl.String, 'score': pl.Float64, 'extra': pl.String}  # fmt: skip
MODELS = {'disease': AssociationOtfDisease, 'target': AssociationOtfTarget}
MEASUREMENT = 'EFO_0001444'
ANCESTORS = ['EFO_0000001', 'EFO_0000002']


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
    disease = pl.DataFrame({
        'id': ['EFO_1', 'EFO_2'],
        'name': ['Sjögren SYNDROME Ä', 'other'],
        'ancestors': [ANCESTORS, []],
        'therapeuticAreas': [['EFO_0000002', MEASUREMENT], ['EFO_0000003']],
        'extra': ['x', 'y'],
    })
    target = pl.DataFrame({
        'id': [T1, T2],
        'approvedName': ['Approved Name É', None],
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


def _destination(tmp_path: Path) -> dict[str, str]:
    return {side: str(tmp_path / 'out' / side) for side in MODELS}


def _run(source: dict[str, str], tmp_path: Path) -> dict[str, pl.DataFrame]:
    destination = _destination(tmp_path)
    association_otf(source, destination, None, None)
    return {side: scan_dataset(path).collect() for side, path in destination.items()}


def _row(frame: pl.DataFrame, row_id: str) -> dict:
    return frame.filter(pl.col('rowId') == row_id).row(0, named=True)


def test_ascii_lower_folds_ascii_only() -> None:
    frame = pl.DataFrame({'s': ['Sjögren SYNDROME Ä', 'ÉCOLE Abc', None]})
    assert frame.select(ascii_lower(pl.col('s')))['s'].to_list() == ['sjögren syndrome Ä', 'École abc', None]


def test_schema_and_order(source: dict[str, str], tmp_path: Path) -> None:
    result = _run(source, tmp_path)
    for side, model in MODELS.items():
        assert result[side].schema == to_polars_schema(model)
        assert result[side].columns == list(model.model_fields)


def test_writes_its_models(source: dict[str, str], tmp_path: Path) -> None:
    assert OutputValidation.of(association_otf, module).outputs == MODELS
    destination = _destination(tmp_path)
    association_otf(source, destination, None, None)
    for side, model in MODELS.items():
        validate_dataset(model, destination[side])


def test_evidence_as_pts_writes_it_gives_the_same_views(source: dict[str, str], tmp_path: Path) -> None:
    """Evidence stores its datasource and datatype as the shared Enums; the views keep them."""
    strings = _run(source, tmp_path / 'strings')
    schema = to_polars_schema(AssociationOtfDisease)
    for path in tmp_path.glob('evidence_*/part-0.parquet'):
        frame = pl.read_parquet(path)
        frame.cast({'datasourceId': schema['datasourceId'], 'datatypeId': schema['datatypeId']}).write_parquet(path)

    typed = _run(source, tmp_path / 'typed')
    for side in MODELS:
        assert typed[side].equals(strings[side])


def test_row_count_equals_evidence(source: dict[str, str], tmp_path: Path) -> None:
    assert {side: frame.height for side, frame in _run(source, tmp_path).items()} == {'disease': 4, 'target': 4}


def test_keys_swap_per_side(source: dict[str, str], tmp_path: Path) -> None:
    result = _run(source, tmp_path)
    assert result['disease'].select('A', 'B').rows() == [('EFO_1', T1), ('EFO_2', T2), ('EFO_1', T2), ('EFO_9', T1)]
    assert result['target'].select('A', 'B').rows() == [(T1, 'EFO_1'), (T2, 'EFO_2'), (T2, 'EFO_1'), (T1, 'EFO_9')]


def test_search_labels_are_ascii_lower_cased(source: dict[str, str], tmp_path: Path) -> None:
    result = _run(source, tmp_path)
    disease_label, target_label = 'efo_1 sjögren syndrome Ä', f'{T1.lower()} approved name É symbol'
    disease_side = _row(result['disease'], E1)
    assert (disease_side['searchA'], disease_side['searchB']) == (disease_label, target_label)
    target_side = _row(result['target'], E1)
    assert (target_side['searchA'], target_side['searchB']) == (target_label, disease_label)


def test_novelties(source: dict[str, str], tmp_path: Path) -> None:
    for frame in _run(source, tmp_path).values():
        row = _row(frame, E1)
        assert (row['noveltyDirect'], row['noveltyIndirect']) == (0.15, 0.25)


def test_null_approved_name_gives_null_target_label(source: dict[str, str], tmp_path: Path) -> None:
    result = _run(source, tmp_path)
    assert _row(result['disease'], E2)['searchA'] == 'efo_2 other'
    assert _row(result['disease'], E2)['searchB'] is None
    assert _row(result['target'], E2)['searchA'] is None


def test_disease_side_indirect_is_empty(source: dict[str, str], tmp_path: Path) -> None:
    assert _run(source, tmp_path)['disease']['indirect'].to_list() == [[], [], [], []]


def test_target_side_indirect_and_measurement(source: dict[str, str], tmp_path: Path) -> None:
    target = _run(source, tmp_path)['target']
    assert target.select('rowId', 'indirect', 'isMeasurement').rows() == [
        (E1, ANCESTORS, True),
        (E2, [], False),
        (E3, ANCESTORS, True),
        (E4, [], False),
    ]


def test_unmatched_gets_nulls_or_defaults(source: dict[str, str], tmp_path: Path) -> None:
    result = _run(source, tmp_path)
    e3 = _row(result['target'], E3)
    assert e3['noveltyDirect'] is None
    assert e3['noveltyIndirect'] == 0.35
    e4 = _row(result['target'], E4)
    assert (e4['searchB'], e4['noveltyIndirect'], e4['indirect'], e4['isMeasurement']) == (None, None, [], False)
    assert _row(result['disease'], E4)['searchA'] is None


def test_sorted_by_row_id(source: dict[str, str], tmp_path: Path) -> None:
    for frame in _run(source, tmp_path).values():
        assert frame['rowId'].to_list() == [E1, E2, E3, E4]


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
    disease = pl.DataFrame({
        'id': [f'EFO_{i}' for i in range(500)],
        'name': [f'Name {i}' for i in range(500)],
        'ancestors': [[f'EFO_{j:07d}' for j in range(i % 4)] for i in range(500)],
        'therapeuticAreas': [[MEASUREMENT] if i % 3 else [] for i in range(500)],
    })
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
        association_otf(paths, _destination(tmp_path / name), None, None)
    for side in MODELS:
        dir_a, dir_b = tmp_path / 'a' / 'out' / side, tmp_path / 'b' / 'out' / side
        files_a = sorted(p.name for p in dir_a.rglob('*.parquet'))
        assert len(files_a) > 1
        assert files_a == sorted(p.name for p in dir_b.rglob('*.parquet'))
        for name in files_a:
            assert (dir_a / name).read_bytes() == (dir_b / name).read_bytes()
        assert scan_dataset(str(dir_a)).collect()['rowId'].is_sorted()


def _small_parts(frame: pl.LazyFrame, path: str) -> pl.LazyFrame:
    return sink_dataset(frame, path, approximate_bytes_per_file=200_000)
