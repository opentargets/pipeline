"""Tests for the study view transformer, which replaces the `studies.sql` serving script."""

from pathlib import Path

import polars as pl
from ot_datamodels.outputs.credible_set import CredibleSet
from ot_datamodels.outputs.study import Study
from ot_datamodels.polars import to_polars_schema
from ot_datamodels.views.study_view import StudyView

from pts.schemas.dataset.validate import validate_dataset
from pts.task_outputs import OutputValidation
from pts.transformers import study_view as module
from pts.transformers.study_view import build_study_view, study_view
from tests.credible_set_fixtures import published_credible_set

STUDY = to_polars_schema(Study)


def _studies(*ids: str) -> pl.DataFrame:
    return pl.DataFrame([Study.example | {'studyId': i} for i in ids], schema=STUDY)


def _sets(*pairs: tuple[str, str]) -> pl.DataFrame:
    return published_credible_set(*({'id': i, 'studyId': s} for i, s in pairs)).select('id', 'studyId')


def test_credible_set_ids_sorted_per_study_empty_for_none():
    """Sets of a study not in `study` are ignored; rows keep study order."""
    view = build_study_view(_studies('S2', 'S1', 'S3'), _sets(('c', 'S1'), ('b', 'S2'), ('a', 'S1'), ('z', 'S9')))

    assert view.select('studyId', 'credibleSetIds').rows() == [('S2', ['b']), ('S1', ['a', 'c']), ('S3', [])]


def test_study_columns_pass_through():
    study = _studies('S1', 'S2')
    view = build_study_view(study, _sets(('a', 'S1')))

    assert view.schema == to_polars_schema(StudyView)
    assert view.drop('credibleSetIds').equals(study)


def test_no_credible_sets_gives_empty_lists():
    view = build_study_view(_studies('S1', 'S2'), _sets())

    assert view['credibleSetIds'].to_list() == [[], []]


def test_writes_its_model(tmp_path: Path):
    source = {}
    for name, frame in (
        ('study', pl.DataFrame([Study.example], schema=STUDY)),
        ('credible_set', published_credible_set(CredibleSet.example | {'studyId': Study.example['studyId']})),
    ):
        (tmp_path / name).mkdir()
        frame.write_parquet(tmp_path / name / 'part-0.parquet')
        source[name] = str(tmp_path / name)
    destination = {'view': str(tmp_path / 'out' / 'view')}

    study_view(source, destination, {}, None)

    assert OutputValidation.of(study_view, module).outputs == {'view': StudyView}
    validate_dataset(StudyView, destination['view'])
    written = pl.read_parquet(destination['view'])
    assert written.schema == to_polars_schema(StudyView)
    assert written['credibleSetIds'].to_list() == [[CredibleSet.example['id']]]
