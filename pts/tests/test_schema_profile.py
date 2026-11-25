"""Tests for profiling a run's outputs against the dataset models."""

from pathlib import Path
from typing import Annotated, ClassVar

import polars as pl
import pytest
from ot_datamodels import DatasetModel, PrimaryKey
from ot_datamodels.base import StructModel
from ot_datamodels.discovery import discover_datasets, discover_models

from pts.schemas.dataset.profile import _write as upload
from pts.schemas.dataset.profile import field_profile, profile_run, run_folder


class _Item(StructModel):
    v: float | None


class _Toy(DatasetModel):
    dataset_name: ClassVar[str] = '_toy'
    dataset_title: ClassVar[str] = 'toy'
    dataset_description: ClassVar[str] = 'toy'
    example: ClassVar[dict] = {'id': 'a', 'n': 1, 'tags': ['x'], 'items': [{'v': 1.0}]}

    id: Annotated[str, PrimaryKey()]
    n: int | None
    tags: list[str | None] | None
    items: list[_Item]


class _EvidenceA(DatasetModel):
    dataset_name: ClassVar[str] = 'evidence_a'
    dataset_title: ClassVar[str] = 'a'
    dataset_description: ClassVar[str] = 'a'
    example: ClassVar[dict] = {'id': 'a'}

    id: Annotated[str, PrimaryKey()]


class _EvidenceB(_EvidenceA):
    dataset_name: ClassVar[str] = 'evidence_b'


class _ViewC(_EvidenceA):
    dataset_name: ClassVar[str] = 'view_c'


def _write(path: Path, frame: pl.DataFrame, parts: int = 1) -> str:
    path.mkdir(parents=True)
    for index in range(parts):
        frame[index::parts].write_parquet(path / f'{index:08d}.parquet')
    return str(path)


def _toy_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            'id': ['a', 'b', 'c', 'd'],
            'n': [1, 5, None, 3],
            'tags': [['x', 'y', 'z'], None, [], [None, 'x']],
            'items': [[{'v': 1.5}, {'v': None}, {'v': 3.5}], [], [{'v': 2.5}], [{'v': 0.5}]],
        },
        schema={
            'id': pl.String,
            'n': pl.Int64,
            'tags': pl.List(pl.String),
            'items': pl.List(pl.Struct({'v': pl.Float64})),
        },
    )


def test_field_profile_counts(tmp_path: Path) -> None:
    path = _write(tmp_path / '_toy', _toy_frame(), parts=2)

    profile = field_profile(path, _Toy)
    by_path = {row['path']: row for row in profile.iter_rows(named=True)}

    assert list(by_path) == ['id', 'n', 'tags', 'tags[]', 'items', 'items[]', 'items[]/v']
    assert {row['dataset'] for row in by_path.values()} == {'_toy'}

    ident = by_path['id']
    assert (ident['nulls'], ident['rows'], ident['list_nulls'], ident['null_elements']) == (0, 4, None, None)
    assert (ident['min'], ident['max'], ident['distinct']) == (None, None, '["a", "b", "c", "d"]')
    n = by_path['n']
    assert (n['nulls'], n['rows'], n['min'], n['max'], n['distinct']) == (1, 4, '1', '5', '[1, 3, 5]')
    assert (n['list_nulls'], n['null_elements']) == (None, None)
    tags = by_path['tags']
    assert (tags['rows'], tags['list_nulls'], tags['null_elements'], tags['distinct']) == (4, 1, 1, None)
    element = by_path['tags[]']
    assert (element['rows'], element['nulls'], element['distinct']) == (5, 1, '["x", "y", "z"]')
    assert (by_path['items']['list_nulls'], by_path['items']['null_elements']) == (0, 0)
    assert by_path['items[]']['rows'] == 5
    assert by_path['items[]']['nulls'] == 0
    v = by_path['items[]/v']
    assert (v['rows'], v['nulls'], v['min'], v['max'], v['distinct']) == (5, 1, '0.5', '3.5', None)
    assert all(row['present'] for row in by_path.values())


def test_field_profile_distinct_capped(tmp_path: Path) -> None:
    path = _write(tmp_path / 'e', pl.DataFrame({'id': [str(i) for i in range(51)]}), parts=3)

    profile = field_profile(path, _EvidenceA)

    assert profile['distinct'].to_list() == [None]
    assert profile['rows'].to_list() == [51]


def test_profile_run_maps_each_model_to_its_folder(tmp_path: Path, monkeypatch) -> None:
    run = tmp_path / 'run'
    _write(run / 'output' / 'evidence_a', pl.DataFrame({'id': ['a1', 'a2']}))
    _write(run / 'output' / 'evidence_b', pl.DataFrame({'id': ['b1']}))
    monkeypatch.setattr(
        'pts.schemas.dataset.profile.discover_models',
        lambda: {'evidence_a': _EvidenceA, 'evidence_b': _EvidenceB},
    )

    profile_run(str(run), str(tmp_path / 'out'))

    profile = pl.read_parquet(tmp_path / 'out' / 'profile.parquet')
    assert dict(zip(profile['dataset'], profile['rows'], strict=True)) == {'evidence_a': 2, 'evidence_b': 1}
    assert pl.read_parquet(tmp_path / 'out' / 'validation.parquet').is_empty()


def test_profile_run_reports_missing_outputs(tmp_path: Path, monkeypatch) -> None:
    run = tmp_path / 'run'
    (run / 'output' / 'evidence_b').mkdir(parents=True)
    _write(run / 'output' / 'other', pl.DataFrame({'id': ['x']}))
    monkeypatch.setattr(
        'pts.schemas.dataset.profile.discover_models',
        lambda: {'evidence_a': _EvidenceA, 'evidence_b': _EvidenceB},
    )

    profile_run(str(run), str(tmp_path / 'out'))

    validation = pl.read_parquet(tmp_path / 'out' / 'validation.parquet')
    assert sorted(validation.select('dataset', 'check').iter_rows()) == [
        ('evidence_a', 'missing'),
        ('evidence_b', 'missing'),
    ]
    assert validation['failures'].to_list() == [0, 0]
    assert pl.read_parquet(tmp_path / 'out' / 'profile.parquet').is_empty()


def test_profile_run_reads_views_from_the_view_folder(tmp_path: Path, monkeypatch) -> None:
    run = tmp_path / 'run'
    _write(run / 'output' / 'evidence_a', pl.DataFrame({'id': ['a1', 'a2']}))
    _write(run / 'view' / 'evidence_b', pl.DataFrame({'id': ['b1']}))
    _write(run / 'output' / 'view_c', pl.DataFrame({'id': ['c1']}))  # a view's name under output is not read
    monkeypatch.setattr(_EvidenceB, '__module__', 'ot_datamodels.views.evidence_b')
    monkeypatch.setattr(_ViewC, '__module__', 'ot_datamodels.views.view_c')
    monkeypatch.setattr(
        'pts.schemas.dataset.profile.discover_models',
        lambda: {'evidence_a': _EvidenceA, 'evidence_b': _EvidenceB, 'view_c': _ViewC},
    )

    profile_run(str(run), str(tmp_path / 'out'))

    profile = pl.read_parquet(tmp_path / 'out' / 'profile.parquet')
    assert dict(zip(profile['dataset'], profile['rows'], strict=True)) == {'evidence_a': 2, 'evidence_b': 1}
    validation = pl.read_parquet(tmp_path / 'out' / 'validation.parquet')
    assert validation.select('dataset', 'check').rows() == [('view_c', 'missing')]


def test_run_folder_of_the_discovered_models() -> None:
    models = discover_models()
    views = {name for name, model in models.items() if run_folder(model) == 'view'}
    assert views == set(models) - set(discover_datasets())
    assert views == {
        'association_otf_disease',
        'association_otf_target',
        'association_timeseries',
        'coding_variant_by_target',
        'coding_variant_by_variant',
        'disease_phenotype_by_disease',
        'disease_view',
        'drug_mechanism_of_action_by_drug',
        'drug_view',
        'drug_warning_by_drug',
        'interaction_view',
        'mouse_phenotype_by_target',
        'pharmacovigilance_by_drug',
        'study_view',
        'target_by_region',
        'target_prioritisation_view',
        'target_view',
        'transcript_view',
        'variant_view',
    }
    assert run_folder(models['target_view']) == 'view'
    assert run_folder(models['target']) == 'output'


def test_field_profile_marks_absent_columns(tmp_path: Path) -> None:
    path = _write(tmp_path / '_toy', _toy_frame().drop('n', 'items'))

    by_path = {row['path']: row for row in field_profile(path, _Toy).iter_rows(named=True)}

    assert list(by_path) == ['id', 'n', 'tags', 'tags[]', 'items', 'items[]', 'items[]/v']
    assert [p for p, row in by_path.items() if not row['present']] == ['n', 'items', 'items[]', 'items[]/v']
    assert by_path['n']['rows'] is None
    assert by_path['tags[]']['rows'] == 5


def test_field_profile_skips_a_column_of_the_wrong_shape(tmp_path: Path) -> None:
    frame = _toy_frame().with_columns(items=pl.lit(1))
    path = _write(tmp_path / '_toy', frame)

    paths = field_profile(path, _Toy)['path'].to_list()

    assert paths == ['id', 'n', 'tags', 'tags[]']


def test_profile_run_survives_a_failing_dataset(tmp_path: Path, monkeypatch) -> None:
    run = tmp_path / 'run'
    _write(run / 'output' / 'evidence_a', pl.DataFrame({'id': ['a1', 'a2']}))
    _write(run / 'output' / 'evidence_b', pl.DataFrame({'id': ['b1']}))
    monkeypatch.setattr(
        'pts.schemas.dataset.profile.discover_models',
        lambda: {'evidence_a': _EvidenceA, 'evidence_b': _EvidenceB},
    )
    real = field_profile

    def flaky(path: str, model: type[DatasetModel]) -> pl.DataFrame:
        if model is _EvidenceA:
            raise RuntimeError('boom')
        return real(path, model)

    monkeypatch.setattr('pts.schemas.dataset.profile.field_profile', flaky)
    out = tmp_path / 'out'

    profile_run(str(run), str(out))

    validation = pl.read_parquet(out / 'validation.parquet')
    assert validation.select('dataset', 'check', 'examples').rows() == [('evidence_a', 'error', ['RuntimeError: boom'])]
    assert pl.read_parquet(out / 'profile.parquet')['dataset'].to_list() == ['evidence_b']
    assert pl.read_parquet(out / 'profile' / 'evidence_b.parquet')['rows'].to_list() == [1]
    assert pl.read_parquet(out / 'validation' / 'evidence_a.parquet')['check'].to_list() == ['error']


def test_write_uploads_in_one_call_without_flushing(tmp_path: Path, monkeypatch) -> None:
    """A GCS writer rejects `flush` before `close`, which polars calls when given a file handle."""
    written: dict[str, bytes] = {}

    class _Writer:
        def write(self, data: bytes) -> int:
            raise AssertionError('streamed write')

        def flush(self) -> None:
            raise AssertionError('flushed before close')

    class _Handle:
        def __init__(self, path: str) -> None:
            self.path = path

        def open(self, mode: str = 'r') -> _Writer:
            return _Writer()

        def write(self, data: bytes) -> None:
            written[self.path] = data

    monkeypatch.setattr('pts.schemas.dataset.profile.StorageHandle', _Handle)
    frame = pl.DataFrame({'a': [1, 2]})

    upload(frame, 'gs://bucket/x.parquet')

    assert pl.read_parquet(written['gs://bucket/x.parquet']).equals(frame)


def test_write_failure_names_the_path(monkeypatch) -> None:
    class _Handle:
        def __init__(self, path: str) -> None:
            pass

        def write(self, data: bytes) -> None:
            raise OSError('denied')

    monkeypatch.setattr('pts.schemas.dataset.profile.StorageHandle', _Handle)

    with pytest.raises(RuntimeError, match=r'could not write gs://bucket/x.parquet: denied'):
        upload(pl.DataFrame({'a': [1]}), 'gs://bucket/x.parquet')
